"""Quantization utilities shared by high-level AdderNet models."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class UniformQuantizer:
    """Map continuous features to uniformly spaced integer bins.

    The fitted feature count removes the common ambiguity between a vector of
    samples for one feature and one sample containing multiple features.
    """

    bins: int = 256
    minimum_: np.ndarray | None = None
    maximum_: np.ndarray | None = None
    n_features_in_: int | None = None

    def __post_init__(self) -> None:
        self.bins = int(self.bins)
        if self.bins < 2:
            raise ValueError("bins must be >= 2")

    @staticmethod
    def _fit_array(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 1-D or 2-D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _transform_array(self, X) -> tuple[np.ndarray, str]:
        if self.n_features_in_ is None:
            raise RuntimeError("fit the quantizer before transform")
        X = np.asarray(X, dtype=np.float64)
        original = "matrix"
        if X.ndim == 0:
            if self.n_features_in_ != 1:
                raise ValueError(f"a scalar cannot represent {self.n_features_in_} features")
            X = X.reshape(1, 1)
            original = "scalar"
        elif X.ndim == 1:
            if self.n_features_in_ == 1:
                X = X[:, None]
                original = "single_feature_vector"
            elif X.size == self.n_features_in_:
                X = X.reshape(1, -1)
                original = "single_sample"
            else:
                raise ValueError(f"a 1-D input must contain exactly {self.n_features_in_} features")
        elif X.ndim != 2:
            raise ValueError("X must be a scalar, 1-D, or 2-D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X, original

    def fit(self, X) -> "UniformQuantizer":
        X = self._fit_array(X)
        self.n_features_in_ = int(X.shape[1])
        self.minimum_ = X.min(axis=0)
        self.maximum_ = X.max(axis=0)
        return self

    def transform(self, X, *, clip: bool = True) -> np.ndarray:
        if self.minimum_ is None or self.maximum_ is None:
            raise RuntimeError("fit the quantizer before transform")
        X, original = self._transform_array(X)
        span = self.maximum_ - self.minimum_
        safe_span = np.where(span > 0, span, 1.0)
        q = np.rint((X - self.minimum_) * (self.bins - 1) / safe_span)
        if clip:
            q = np.clip(q, 0, self.bins - 1)
        elif np.any((q < 0) | (q >= self.bins)):
            raise ValueError("X contains values outside the fitted quantization range")
        q = q.astype(np.int32)
        if original in {"scalar", "single_feature_vector"}:
            return q[:, 0]
        if original == "single_sample":
            return q[0]
        return q

    def fit_transform(self, X, *, clip: bool = True) -> np.ndarray:
        return self.fit(X).transform(X, clip=clip)

    def inverse_transform(self, Q) -> np.ndarray:
        if self.minimum_ is None or self.maximum_ is None:
            raise RuntimeError("fit the quantizer before inverse_transform")
        Q, original = self._transform_array(Q)
        if np.any((Q < 0) | (Q >= self.bins)):
            raise ValueError("quantized values must be inside [0, bins)")
        span = self.maximum_ - self.minimum_
        X = self.minimum_ + Q * span / (self.bins - 1)
        if original in {"scalar", "single_feature_vector"}:
            return X[:, 0]
        if original == "single_sample":
            return X[0]
        return X

    def to_dict(self) -> dict:
        if self.minimum_ is None or self.maximum_ is None:
            raise RuntimeError("fit the quantizer before serialization")
        return {
            "bins": self.bins,
            "minimum": self.minimum_.tolist(),
            "maximum": self.maximum_.tolist(),
            "n_features_in": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UniformQuantizer":
        obj = cls(int(data["bins"]))
        obj.minimum_ = np.asarray(data["minimum"], dtype=np.float64)
        obj.maximum_ = np.asarray(data["maximum"], dtype=np.float64)
        obj.n_features_in_ = int(data["n_features_in"])
        if obj.minimum_.shape != (obj.n_features_in_,) or obj.maximum_.shape != (obj.n_features_in_,):
            raise ValueError("invalid quantizer metadata")
        return obj
