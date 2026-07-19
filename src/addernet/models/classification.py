"""Classification wrapper built on the multi-input AdderNet regressor."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .multi_input import AdderNetMultiInputLayer


class AdderNetClassifier:
    def __init__(self, bins: int = 64, **model_kwargs):
        self.bins = int(bins)
        self.model_kwargs = dict(model_kwargs)
        self.model_: AdderNetMultiInputLayer | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y) -> "AdderNetClassifier":
        y = np.asarray(y)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1-D array")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        targets = -np.ones((len(y), len(self.classes_)), dtype=np.float64)
        targets[np.arange(len(y)), encoded] = 1.0
        self.model_ = AdderNetMultiInputLayer(self.bins, **self.model_kwargs).fit(X, targets)
        return self

    def _check(self) -> None:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("fit the classifier before prediction")

    def decision_function(self, X) -> np.ndarray:
        self._check()
        scores = np.asarray(self.model_.predict_batch(X), dtype=np.float64)
        return scores if scores.ndim == 2 else scores[:, None]

    def predict(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def save(self, directory) -> None:
        self._check()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model_.save(directory / "model.npz")
        with (directory / "classifier.json").open("w", encoding="utf-8") as fh:
            json.dump({"bins": self.bins, "model_kwargs": self.model_kwargs}, fh, indent=2)
        np.save(directory / "classes.npy", self.classes_, allow_pickle=False)

    @classmethod
    def load(cls, directory) -> "AdderNetClassifier":
        directory = Path(directory)
        metadata = json.loads((directory / "classifier.json").read_text(encoding="utf-8"))
        obj = cls(metadata["bins"], **metadata["model_kwargs"])
        obj.model_ = AdderNetMultiInputLayer.load(directory / "model.npz")
        obj.classes_ = np.load(directory / "classes.npy", allow_pickle=False)
        return obj
