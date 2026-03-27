"""
AdderBoost — Gradient Boosting with AdderNetLayer as base learners.

Each estimator learns the residual of the ensemble so far.
Inference: sum of lookup tables — O(n_estimators) memory reads.
"""

import warnings
import numpy as np
from .addernet import AdderNetLayer


class AdderBoost:
    """
    Gradient Boosting with AdderNetLayer.

    Each step trains one AdderNetLayer per feature on the current residual.
    Inference is a sum of LUT lookups — zero multiplication.
    """

    def __init__(self, n_estimators: int = 10, learning_rate: float = 0.1,
                 lr_boost: float = None, **layer_kwargs):
        """
        Args:
            n_estimators: number of boosting rounds
            learning_rate: boosting learning rate / shrinkage (0.0 to 1.0)
            lr_boost: DEPRECATED alias for learning_rate (backward compatibility)
            **layer_kwargs: passed to AdderNetLayer (size, bias, input_min, input_max, lr, etc.)
        """
        self.n_estimators = n_estimators
        if lr_boost is not None:
            warnings.warn("lr_boost is deprecated, use learning_rate instead",
                          DeprecationWarning, stacklevel=2)
            self.learning_rate = lr_boost
        else:
            self.learning_rate = learning_rate
        self.layer_kwargs = layer_kwargs  # does NOT include learning_rate
        self.estimators = []
        self.base_prediction = 0.0
        self.n_features = None

    # Backward compatibility property
    @property
    def lr_boost(self):
        return self.learning_rate

    @lr_boost.setter
    def lr_boost(self, value):
        self.learning_rate = value

    def fit(self, X: np.ndarray, y_cont: np.ndarray, verbose: bool = False) -> 'AdderBoost':
        """
        Train on data.

        Args:
            X: (n_samples, n_features) — normalized
            y_cont: (n_samples,) — continuous target values
            verbose: print residual progress
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y_cont = np.ascontiguousarray(y_cont, dtype=np.float64)
        self.n_features = X.shape[1]

        residual = y_cont.copy()
        self.base_prediction = float(y_cont.mean())
        residual -= self.base_prediction

        self.estimators = []

        for step in range(self.n_estimators):
            step_nets = []
            step_preds = np.zeros(len(X))

            for i in range(self.n_features):
                net = AdderNetLayer(**self.layer_kwargs)
                net.train(X[:, i], residual)
                step_preds += net.predict_batch(X[:, i])
                step_nets.append(net)

            step_preds /= self.n_features
            residual -= self.learning_rate * step_preds
            self.estimators.append(step_nets)

            if verbose:
                mean_res = np.abs(residual).mean()
                print(f"  Boost step {step + 1}/{self.n_estimators} | mean residual: {mean_res:.4f}")

        return self

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch inference — sum of all estimator LUT lookups.

        Args:
            X: (n_samples, n_features)
        Returns:
            (n_samples,) continuous predictions
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        pred = np.full(len(X), self.base_prediction)

        for step_nets in self.estimators:
            step_pred = np.zeros(len(X))
            for i, net in enumerate(step_nets):
                step_pred += net.predict_batch(X[:, i])
            pred += self.learning_rate * step_pred / len(step_nets)

        return pred

    def predict(self, x) -> float:
        """Single-sample inference."""
        return float(self.predict_batch(np.array(x).reshape(1, -1))[0])
