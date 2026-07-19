"""Readable NumPy reference implementation of the scalar AdderNet LUT."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .preprocessing import UniformQuantizer


class ReferenceAdderNetLayer:
    """Pure NumPy scalar LUT for debugging, education, and portability."""

    def __init__(self, size=256, bias=0, input_min=0, input_max=255, lr=0.1):
        size = int(size)
        if size <= 0 or size & (size - 1):
            raise ValueError("size must be a positive power of two")
        if int(input_max) < int(input_min):
            raise ValueError("input_max must be >= input_min")
        if not np.isfinite(lr) or float(lr) <= 0:
            raise ValueError("lr must be positive and finite")
        self.size = size
        self.mask = size - 1
        self.bias = int(bias)
        self.input_min = int(input_min)
        self.input_max = int(input_max)
        self.lr = float(lr)
        self.offset_table = np.zeros(size, dtype=np.float64)

    def _index(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("inputs must be finite")
        return (x.astype(np.int64) + self.bias) & self.mask

    @staticmethod
    def _unique_means(inputs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        integer_x = inputs.astype(np.int64)
        unique, inverse = np.unique(integer_x, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=targets)
        return unique, sums / counts

    @classmethod
    def _dense_samples(cls, inputs: np.ndarray, targets: np.ndarray, lo: int, hi: int):
        unique, means = cls._unique_means(inputs, targets)
        grid = np.arange(lo, hi + 1, dtype=np.int64)
        if unique.size == 1:
            dense = np.full(grid.shape, means[0], dtype=np.float64)
        else:
            dense = np.interp(grid, unique, means)
            left_slope = (means[1] - means[0]) / (unique[1] - unique[0])
            right_slope = (means[-1] - means[-2]) / (unique[-1] - unique[-2])
            left = grid < unique[0]
            right = grid > unique[-1]
            dense[left] = means[0] + left_slope * (grid[left] - unique[0])
            dense[right] = means[-1] + right_slope * (grid[right] - unique[-1])
        return grid.astype(np.float64), dense

    @staticmethod
    def _validate(inputs, targets) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(inputs, dtype=np.float64).reshape(-1)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if x.size == 0 or x.size != y.size:
            raise ValueError("inputs and targets must have equal non-zero length")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("inputs and targets must be finite")
        return x, y

    def fit(self, inputs, targets, *, interpolate=True, blend=1.0):
        x, y = self._validate(inputs, targets)
        blend = float(blend)
        if not np.isfinite(blend) or not 0 < blend <= 1:
            raise ValueError("blend must be in (0, 1]")
        if interpolate:
            dense_x, dense_y = self._dense_samples(x, y, self.input_min, self.input_max)
            idx = self._index(dense_x)
            self.offset_table[idx] = (1 - blend) * self.offset_table[idx] + blend * dense_y
        else:
            idx = self._index(x)
            unique_idx, inverse = np.unique(idx, return_inverse=True)
            counts = np.bincount(inverse)
            means = np.bincount(inverse, weights=y) / counts
            self.offset_table[unique_idx] = (
                (1 - blend) * self.offset_table[unique_idx] + blend * means
            )
        return self

    partial_fit = fit

    def _train_samples(self, inputs, targets, epochs, trace=None):
        epochs = int(epochs)
        if epochs < 0:
            raise ValueError("epochs cannot be negative")
        for epoch in range(epochs):
            for sample_index, (x, target) in enumerate(zip(inputs, targets)):
                index = int(self._index([x])[0])
                before = self.offset_table[index]
                candidates = np.array([before, before + self.lr, before - self.lr])
                after = candidates[np.argmin(np.abs(candidates - target))]
                self.offset_table[index] = after
                if trace is not None:
                    trace({
                        "epoch": epoch, "sample": sample_index, "input": float(x),
                        "target": float(target), "index": index,
                        "before": float(before), "after": float(after),
                    })

    def train(self, inputs, targets, epochs_raw=100, epochs_expanded=400, trace=None):
        x, y = self._validate(inputs, targets)
        self._train_samples(x, y, epochs_raw, trace)
        if int(epochs_expanded) > 0:
            dense_x, dense_y = self._dense_samples(x, y, self.input_min, self.input_max)
            self._train_samples(dense_x, dense_y, epochs_expanded, trace)
        return self

    def predict(self, x) -> float:
        return float(self.offset_table[int(self._index([x])[0])])

    def predict_batch(self, inputs) -> np.ndarray:
        return self.offset_table[self._index(inputs)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez_compressed(
                fh, size=self.size, bias=self.bias, input_min=self.input_min,
                input_max=self.input_max, lr=self.lr, table=self.offset_table,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            obj = cls(int(data["size"]), int(data["bias"]), int(data["input_min"]),
                      int(data["input_max"]), float(data["lr"]))
            table = np.asarray(data["table"], dtype=np.float64)
        if table.shape != (obj.size,) or not np.all(np.isfinite(table)):
            raise ValueError("invalid reference model file")
        obj.offset_table[:] = table
        return obj


__all__ = ["ReferenceAdderNetLayer", "UniformQuantizer"]
