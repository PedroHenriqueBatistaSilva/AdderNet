"""Readable NumPy reference implementations used for learning and debugging."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class UniformQuantizer:
    """Map continuous features to integer bins.

    The transform is intentionally explicit: native ``AdderNetLayer`` truncates
    inputs to integers, so quantization should be a conscious preprocessing step.
    """
    bins: int = 256
    minimum_: np.ndarray | None = None
    maximum_: np.ndarray | None = None

    def fit(self, X) -> "UniformQuantizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if X.size == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be non-empty and finite")
        if self.bins < 2:
            raise ValueError("bins must be >= 2")
        self.minimum_ = X.min(axis=0)
        self.maximum_ = X.max(axis=0)
        return self

    def transform(self, X) -> np.ndarray:
        if self.minimum_ is None or self.maximum_ is None:
            raise RuntimeError("fit the quantizer before transform")
        X = np.asarray(X, dtype=np.float64)
        one_dim = X.ndim == 1
        if one_dim:
            X = X[:, None]
        span = self.maximum_ - self.minimum_
        span = np.where(span > 0, span, 1.0)
        q = np.rint((X - self.minimum_) * (self.bins - 1) / span)
        q = np.clip(q, 0, self.bins - 1).astype(np.int32)
        return q[:, 0] if one_dim else q

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


class ReferenceAdderNetLayer:
    """Pure-Python/NumPy equivalent of the scalar LUT learning idea.

    It is slower than the C implementation but exposes every step and supports
    an optional trace callback for teaching tools.
    """
    def __init__(self, size=256, bias=0, input_min=0, input_max=255, lr=0.1):
        if size <= 0 or size & (size - 1):
            raise ValueError("size must be a positive power of two")
        if input_max < input_min:
            raise ValueError("input_max must be >= input_min")
        if lr <= 0 or not np.isfinite(lr):
            raise ValueError("lr must be positive and finite")
        self.size = int(size)
        self.mask = self.size - 1
        self.bias = int(bias)
        self.input_min = int(input_min)
        self.input_max = int(input_max)
        self.lr = float(lr)
        self.offset_table = np.zeros(self.size, dtype=np.float64)

    def _index(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64).astype(np.int64) + self.bias) & self.mask

    @staticmethod
    def _dense_samples(inputs: np.ndarray, targets: np.ndarray, lo: int, hi: int):
        # Average duplicate integer inputs before interpolation.
        integer_x = inputs.astype(np.int64)
        unique = np.unique(integer_x)
        means = np.array([targets[integer_x == value].mean() for value in unique])
        grid = np.arange(lo, hi + 1, dtype=np.int64)
        if len(unique) == 1:
            dense = np.full(grid.shape, means[0], dtype=np.float64)
        else:
            dense = np.interp(grid, unique, means)
            left_slope = (means[1] - means[0]) / max(unique[1] - unique[0], 1)
            right_slope = (means[-1] - means[-2]) / max(unique[-1] - unique[-2], 1)
            left = grid < unique[0]
            right = grid > unique[-1]
            dense[left] = means[0] + left_slope * (grid[left] - unique[0])
            dense[right] = means[-1] + right_slope * (grid[right] - unique[-1])
        return grid.astype(np.float64), dense

    def _train_samples(self, inputs, targets, epochs, trace=None):
        for epoch in range(int(epochs)):
            for sample_index, (x, target) in enumerate(zip(inputs, targets)):
                index = int(self._index([x])[0])
                current = self.offset_table[index]
                up = current + self.lr
                down = current - self.lr
                errors = (abs(current - target), abs(up - target), abs(down - target))
                choice = int(np.argmin(errors))
                if choice == 1:
                    self.offset_table[index] = up
                elif choice == 2:
                    self.offset_table[index] = down
                if trace is not None:
                    trace({
                        "epoch": epoch, "sample": sample_index, "input": float(x),
                        "target": float(target), "index": index,
                        "before": float(current), "after": float(self.offset_table[index]),
                    })

    def train(self, inputs, targets, epochs_raw=100, epochs_expanded=400, trace=None):
        inputs = np.asarray(inputs, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if len(inputs) == 0 or len(inputs) != len(targets):
            raise ValueError("inputs and targets must have equal non-zero length")
        if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(targets))):
            raise ValueError("inputs and targets must be finite")
        self._train_samples(inputs, targets, epochs_raw, trace)
        if epochs_expanded > 0:
            dense_x, dense_y = self._dense_samples(inputs, targets, self.input_min, self.input_max)
            self._train_samples(dense_x, dense_y, epochs_expanded, trace)
        return self

    def predict(self, x) -> float:
        return float(self.offset_table[int(self._index([x])[0])])

    def predict_batch(self, inputs) -> np.ndarray:
        return self.offset_table[self._index(inputs)]

    def save(self, path):
        path = Path(path)
        np.savez_compressed(path, size=self.size, bias=self.bias,
                            input_min=self.input_min, input_max=self.input_max,
                            lr=self.lr, table=self.offset_table)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls(int(data["size"]), int(data["bias"]), int(data["input_min"]),
                  int(data["input_max"]), float(data["lr"]))
        obj.offset_table[:] = data["table"]
        return obj
