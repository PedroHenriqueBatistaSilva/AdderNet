"""Multi-input and multi-output lookup models for AdderNet."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..preprocessing import UniformQuantizer


def _validate_xy(X, y) -> tuple[np.ndarray, np.ndarray, bool]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have shape (samples, features)")
    one_output = y.ndim == 1
    if one_output:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != X.shape[0] or y.shape[1] == 0:
        raise ValueError("y must have shape (samples,) or (samples, outputs)")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        raise ValueError("X and y must contain only finite values")
    return np.ascontiguousarray(X), np.ascontiguousarray(y), one_output


def _group_means(indices: np.ndarray, values: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Return output-by-group means and group counts."""
    counts = np.bincount(indices, minlength=groups).astype(np.int64)
    table = np.zeros((values.shape[1], groups), dtype=np.float64)
    occupied = counts > 0
    for output in range(values.shape[1]):
        sums = np.bincount(indices, weights=values[:, output], minlength=groups)
        table[output, occupied] = sums[occupied] / counts[occupied]
    return table, counts


class AdderNetMultiInputLayer:
    """Lookup layer accepting N inputs and producing one or more outputs.

    The model combines one LUT per feature with optional pairwise joint LUTs.
    For two inputs, ``interactions='auto'`` creates a joint table, directly
    addressing functions such as ``f(x, y)`` that a scalar ``AdderNetLayer``
    cannot represent. Pair indices use bit shifting because ``bins`` must be a
    power of two; quantized inference is lookup, addition, and masking only.
    """

    FORMAT_VERSION = 1

    def __init__(self, bins: int = 64, *, interactions: str | Sequence[tuple[int, int]] = "auto",
                 max_interactions: int = 8, backfit_rounds: int = 3,
                 interaction_rounds: int = 1, min_samples_per_cell: int = 1):
        bins = int(bins)
        if bins < 2 or bins & (bins - 1):
            raise ValueError("bins must be a power of two >= 2")
        if bins > 4096:
            raise ValueError("bins is too large for pairwise LUTs")
        if int(max_interactions) < 0:
            raise ValueError("max_interactions cannot be negative")
        if int(backfit_rounds) <= 0 or int(interaction_rounds) <= 0:
            raise ValueError("training rounds must be positive")
        if int(min_samples_per_cell) <= 0:
            raise ValueError("min_samples_per_cell must be positive")
        self.bins = bins
        self.interactions = interactions
        self.max_interactions = int(max_interactions)
        self.backfit_rounds = int(backfit_rounds)
        self.interaction_rounds = int(interaction_rounds)
        self.min_samples_per_cell = int(min_samples_per_cell)
        self._bits = bins.bit_length() - 1

        self.quantizer = UniformQuantizer(bins)
        self.intercept_: np.ndarray | None = None
        self.additive_tables_: np.ndarray | None = None
        self.interaction_tables_: np.ndarray | None = None
        self.interaction_pairs_: list[tuple[int, int]] = []
        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None
        self.training_rmse_: float | None = None

    def _resolve_pairs(self, Q: np.ndarray, residual: np.ndarray) -> list[tuple[int, int]]:
        n_features = Q.shape[1]
        all_pairs = list(combinations(range(n_features), 2))
        spec = self.interactions
        if spec in (None, "none"):
            return []
        if isinstance(spec, str):
            if spec not in {"auto", "all"}:
                raise ValueError("interactions must be 'auto', 'all', 'none', or a pair list")
            if spec == "all":
                return all_pairs[:self.max_interactions]
            if n_features == 2 and self.max_interactions > 0:
                return [(0, 1)]
            if self.max_interactions == 0:
                return []
            scored: list[tuple[float, tuple[int, int]]] = []
            groups = self.bins * self.bins
            baseline = float(np.mean(residual * residual))
            for pair in all_pairs:
                idx = (Q[:, pair[0]].astype(np.int64) << self._bits) | Q[:, pair[1]]
                table, counts = _group_means(idx, residual, groups)
                prediction = table[:, idx].T
                score = baseline - float(np.mean((residual - prediction) ** 2))
                occupied = int(np.count_nonzero(counts))
                score *= occupied / max(groups, 1)
                scored.append((score, pair))
            scored.sort(reverse=True)
            return [pair for score, pair in scored[:self.max_interactions] if score > 0]

        pairs = []
        for raw_pair in spec:
            if len(raw_pair) != 2:
                raise ValueError("each interaction must contain two feature indices")
            a, b = sorted((int(raw_pair[0]), int(raw_pair[1])))
            if a < 0 or b >= n_features or a == b:
                raise ValueError(f"invalid interaction pair {(a, b)}")
            if (a, b) not in pairs:
                pairs.append((a, b))
        if len(pairs) > self.max_interactions:
            raise ValueError("interaction list exceeds max_interactions")
        return pairs

    def fit(self, X, y) -> "AdderNetMultiInputLayer":
        X, y, _ = _validate_xy(X, y)
        Q = np.asarray(self.quantizer.fit_transform(X), dtype=np.int64)
        if Q.ndim == 1:
            Q = Q[:, None]
        n_samples, n_features = Q.shape
        n_outputs = y.shape[1]
        self.n_features_in_ = n_features
        self.n_outputs_ = n_outputs
        self.intercept_ = y.mean(axis=0)
        self.additive_tables_ = np.zeros((n_outputs, n_features, self.bins), dtype=np.float64)

        prediction = np.tile(self.intercept_, (n_samples, 1))
        for _ in range(self.backfit_rounds):
            for feature in range(n_features):
                old = self.additive_tables_[:, feature, Q[:, feature]].T
                residual = y - (prediction - old)
                table, counts = _group_means(Q[:, feature], residual, self.bins)
                table[:, counts < self.min_samples_per_cell] = 0.0
                # Center each feature contribution to keep the intercept identifiable.
                center = table[:, Q[:, feature]].mean(axis=1, keepdims=True)
                table -= center
                self.intercept_ += center[:, 0]
                new = table[:, Q[:, feature]].T
                prediction += new - old + center[:, 0]
                self.additive_tables_[:, feature] = table

        residual = y - prediction
        self.interaction_pairs_ = self._resolve_pairs(Q, residual)
        groups = self.bins * self.bins
        self.interaction_tables_ = np.zeros((n_outputs, len(self.interaction_pairs_), groups), dtype=np.float64)
        for _ in range(self.interaction_rounds):
            for pair_index, (a, b) in enumerate(self.interaction_pairs_):
                idx = (Q[:, a] << self._bits) | Q[:, b]
                old = self.interaction_tables_[:, pair_index, idx].T
                residual = y - (prediction - old)
                table, counts = _group_means(idx, residual, groups)
                table[:, counts < self.min_samples_per_cell] = 0.0
                center = table[:, idx].mean(axis=1, keepdims=True)
                table -= center
                self.intercept_ += center[:, 0]
                new = table[:, idx].T
                prediction += new - old + center[:, 0]
                self.interaction_tables_[:, pair_index] = table

        self.training_rmse_ = float(np.sqrt(np.mean((y - prediction) ** 2)))
        return self

    train = fit

    def _check_fitted(self) -> None:
        if self.intercept_ is None or self.additive_tables_ is None or self.n_features_in_ is None:
            raise RuntimeError("fit the model before prediction")

    def _coerce_inputs(self, *inputs) -> tuple[np.ndarray, bool]:
        self._check_fitted()
        if len(inputs) == 1:
            X = np.asarray(inputs[0], dtype=np.float64)
            single = X.ndim == 1 and self.n_features_in_ > 1
            if X.ndim == 0:
                X = X.reshape(1, 1)
                single = True
            elif X.ndim == 1:
                if self.n_features_in_ == 1:
                    X = X[:, None]
                    single = X.shape[0] == 1
                elif X.size == self.n_features_in_:
                    X = X.reshape(1, -1)
                else:
                    raise ValueError(f"expected {self.n_features_in_} features")
        else:
            if len(inputs) != self.n_features_in_:
                raise ValueError(f"expected {self.n_features_in_} input arguments")
            arrays = [np.asarray(value, dtype=np.float64) for value in inputs]
            single = all(array.ndim == 0 for array in arrays)
            arrays = np.broadcast_arrays(*arrays)
            X = np.column_stack([array.reshape(-1) for array in arrays])
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (samples, {self.n_features_in_})")
        if not np.all(np.isfinite(X)):
            raise ValueError("inputs must contain only finite values")
        return X, single

    def predict_quantized(self, Q) -> np.ndarray:
        self._check_fitted()
        Q = np.asarray(Q, dtype=np.int64)
        if Q.ndim == 1:
            if self.n_features_in_ == 1:
                Q = Q[:, None]
            elif Q.size == self.n_features_in_:
                Q = Q.reshape(1, -1)
        if Q.ndim != 2 or Q.shape[1] != self.n_features_in_:
            raise ValueError(f"Q must have shape (samples, {self.n_features_in_})")
        if np.any((Q < 0) | (Q >= self.bins)):
            raise ValueError("quantized inputs must be in [0, bins)")
        result = np.tile(self.intercept_, (Q.shape[0], 1))
        for feature in range(self.n_features_in_):
            result += self.additive_tables_[:, feature, Q[:, feature]].T
        if self.interaction_tables_ is not None:
            for pair_index, (a, b) in enumerate(self.interaction_pairs_):
                idx = (Q[:, a] << self._bits) | Q[:, b]
                result += self.interaction_tables_[:, pair_index, idx].T
        return result

    def predict_batch(self, X) -> np.ndarray:
        X, _ = self._coerce_inputs(X)
        Q = self.quantizer.transform(X)
        result = self.predict_quantized(Q)
        return result[:, 0] if self.n_outputs_ == 1 else result

    def predict(self, *inputs):
        X, single = self._coerce_inputs(*inputs)
        result = self.predict_quantized(self.quantizer.transform(X))
        if single or len(result) == 1:
            return float(result[0, 0]) if self.n_outputs_ == 1 else result[0].copy()
        return result[:, 0] if self.n_outputs_ == 1 else result

    def score(self, X, y) -> float:
        _, y, _ = _validate_xy(X, y)
        prediction = np.asarray(self.predict_batch(X), dtype=np.float64)
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        residual = np.sum((y - prediction) ** 2)
        total = np.sum((y - y.mean(axis=0)) ** 2)
        return float(1.0 - residual / total) if total > 0 else float(residual == 0)

    def explain(self, *inputs) -> dict[str, np.ndarray]:
        X, _ = self._coerce_inputs(*inputs)
        Q = np.asarray(self.quantizer.transform(X), dtype=np.int64)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1) if self.n_features_in_ > 1 else Q[:, None]
        additive = np.stack([
            self.additive_tables_[:, feature, Q[:, feature]].T
            for feature in range(self.n_features_in_)
        ], axis=1)
        if self.interaction_pairs_:
            interaction = np.stack([
                self.interaction_tables_[:, i, (Q[:, a] << self._bits) | Q[:, b]].T
                for i, (a, b) in enumerate(self.interaction_pairs_)
            ], axis=1)
        else:
            interaction = np.empty((len(Q), 0, self.n_outputs_), dtype=np.float64)
        prediction = self.predict_quantized(Q)
        return {
            "intercept": np.tile(self.intercept_, (len(Q), 1)),
            "additive": additive,
            "interaction": interaction,
            "prediction": prediction,
        }

    @property
    def memory_bytes_(self) -> int:
        self._check_fitted()
        total = self.intercept_.nbytes + self.additive_tables_.nbytes
        if self.interaction_tables_ is not None:
            total += self.interaction_tables_.nbytes
        return int(total)

    def save(self, path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "addernet.multi_input",
            "format_version": self.FORMAT_VERSION,
            "bins": self.bins,
            "interactions": self.interactions if isinstance(self.interactions, str) else [list(p) for p in self.interactions],
            "max_interactions": self.max_interactions,
            "backfit_rounds": self.backfit_rounds,
            "interaction_rounds": self.interaction_rounds,
            "min_samples_per_cell": self.min_samples_per_cell,
            "interaction_pairs": [list(pair) for pair in self.interaction_pairs_],
            "n_features_in": self.n_features_in_,
            "n_outputs": self.n_outputs_,
            "training_rmse": self.training_rmse_,
            "quantizer": self.quantizer.to_dict(),
        }
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as fh:
            np.savez_compressed(
                fh,
                metadata=json.dumps(metadata),
                intercept=self.intercept_,
                additive_tables=self.additive_tables_,
                interaction_tables=self.interaction_tables_,
            )
        temp.replace(path)

    @classmethod
    def load(cls, path) -> "AdderNetMultiInputLayer":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format") != "addernet.multi_input" or metadata.get("format_version") != 1:
                raise ValueError("unsupported multi-input model format")
            obj = cls(
                metadata["bins"],
                interactions=metadata["interactions"],
                max_interactions=metadata["max_interactions"],
                backfit_rounds=metadata["backfit_rounds"],
                interaction_rounds=metadata["interaction_rounds"],
                min_samples_per_cell=metadata["min_samples_per_cell"],
            )
            obj.quantizer = UniformQuantizer.from_dict(metadata["quantizer"])
            obj.intercept_ = np.asarray(archive["intercept"], dtype=np.float64)
            obj.additive_tables_ = np.asarray(archive["additive_tables"], dtype=np.float64)
            obj.interaction_tables_ = np.asarray(archive["interaction_tables"], dtype=np.float64)
        obj.interaction_pairs_ = [tuple(pair) for pair in metadata["interaction_pairs"]]
        obj.n_features_in_ = int(metadata["n_features_in"])
        obj.n_outputs_ = int(metadata["n_outputs"])
        obj.training_rmse_ = metadata["training_rmse"]
        return obj


AdderNetRegressor = AdderNetMultiInputLayer
