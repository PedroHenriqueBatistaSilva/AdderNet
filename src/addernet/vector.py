"""High-level multivariate additive regression built from scalar AdderNet LUTs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .addernet import AdderNetLayer
from .preprocessing import UniformQuantizer


class AdderNetAdditiveRegressor:
    """Fit y as an intercept plus a sum of one LUT contribution per feature.

    This is a generalized additive model, not a dense neural network. It is a
    practical bridge from the scalar ``AdderNetLayer`` to multivariate tasks.
    Interactions can be supplied explicitly as extra input features.
    """
    def __init__(self, table_size=256, lr=0.05, backfit_rounds=2,
                 epochs_raw=40, epochs_expanded=80, training_method="direct"):
        if table_size <= 0 or table_size & (table_size - 1):
            raise ValueError("table_size must be a positive power of two")
        self.table_size = int(table_size)
        self.lr = float(lr)
        self.backfit_rounds = int(backfit_rounds)
        self.epochs_raw = int(epochs_raw)
        self.epochs_expanded = int(epochs_expanded)
        if training_method not in {"direct", "iterative"}:
            raise ValueError("training_method must be direct or iterative")
        self.training_method = training_method
        self.quantizer = UniformQuantizer(self.table_size)
        self.layers_: list[list[AdderNetLayer]] | None = None
        self.intercept_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must have shape (samples, features)")
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != X.shape[0]:
            raise ValueError("y must have shape (samples,) or (samples, outputs)")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("X and y must be finite")

        Q = self.quantizer.fit_transform(X)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        self.intercept_ = y.mean(axis=0)
        self.layers_ = [
            [AdderNetLayer(size=self.table_size, bias=0, input_min=0,
                           input_max=self.table_size - 1, lr=self.lr)
             for _ in range(self.n_features_in_)]
            for _ in range(self.n_outputs_)
        ]

        contributions = np.zeros((X.shape[0], self.n_outputs_, self.n_features_in_), dtype=np.float64)
        for _round in range(max(self.backfit_rounds, 1)):
            for output in range(self.n_outputs_):
                for feature in range(self.n_features_in_):
                    other = contributions[:, output, :].sum(axis=1) - contributions[:, output, feature]
                    residual = y[:, output] - self.intercept_[output] - other
                    layer = self.layers_[output][feature]
                    if self.training_method == "direct":
                        layer.fit(Q[:, feature], residual, interpolate=True)
                    else:
                        layer.train(Q[:, feature], residual,
                                    epochs_raw=self.epochs_raw,
                                    epochs_expanded=self.epochs_expanded)
                    contributions[:, output, feature] = layer.predict_batch(Q[:, feature])
        return self

    def _check(self, X):
        if self.layers_ is None or self.intercept_ is None:
            raise RuntimeError("fit the model before prediction")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        return X

    def predict(self, X):
        X = self._check(X)
        Q = self.quantizer.transform(X)
        result = np.tile(self.intercept_, (len(X), 1))
        for output, output_layers in enumerate(self.layers_):
            for feature, layer in enumerate(output_layers):
                result[:, output] += layer.predict_batch(Q[:, feature])
        return result[:, 0] if self.n_outputs_ == 1 else result

    def save(self, directory):
        if self.layers_ is None:
            raise RuntimeError("fit the model before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "table_size": self.table_size, "lr": self.lr,
            "backfit_rounds": self.backfit_rounds,
            "epochs_raw": self.epochs_raw, "epochs_expanded": self.epochs_expanded,
            "training_method": self.training_method,
            "n_features_in": self.n_features_in_, "n_outputs": self.n_outputs_,
            "intercept": self.intercept_.tolist(),
            "quantizer_min": self.quantizer.minimum_.tolist(),
            "quantizer_max": self.quantizer.maximum_.tolist(),
        }
        for output, output_layers in enumerate(self.layers_):
            for feature, layer in enumerate(output_layers):
                layer.save(str(directory / f"layer_o{output}_f{feature}.bin"))
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        obj = cls(manifest["table_size"], manifest["lr"], manifest["backfit_rounds"],
                  manifest["epochs_raw"], manifest["epochs_expanded"],
                  manifest.get("training_method", "iterative"))
        obj.n_features_in_ = manifest["n_features_in"]
        obj.n_outputs_ = manifest["n_outputs"]
        obj.intercept_ = np.asarray(manifest["intercept"], dtype=np.float64)
        obj.quantizer.minimum_ = np.asarray(manifest["quantizer_min"], dtype=np.float64)
        obj.quantizer.maximum_ = np.asarray(manifest["quantizer_max"], dtype=np.float64)
        obj.quantizer.n_features_in_ = obj.n_features_in_
        obj.layers_ = [
            [AdderNetLayer.load(str(directory / f"layer_o{o}_f{f}.bin"))
             for f in range(obj.n_features_in_)]
            for o in range(obj.n_outputs_)
        ]
        return obj
