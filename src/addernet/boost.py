"""AdderBoost — additive gradient boosting with LUT base learners."""

import warnings
import numpy as np
from .addernet import AdderNetLayer


class AdderBoost:
    """Gradient boosting with one :class:`AdderNetLayer` per feature."""

    def __init__(self, n_estimators=10, learning_rate=0.1, lr_boost=None,
                 epochs_raw=1000, epochs_expanded=4000, training_method="direct", **layer_kwargs):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if lr_boost is not None:
            warnings.warn("lr_boost is deprecated, use learning_rate instead",
                          DeprecationWarning, stacklevel=2)
            learning_rate = lr_boost
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        if epochs_raw < 0 or epochs_expanded < 0:
            raise ValueError("training epochs cannot be negative")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.epochs_raw = int(epochs_raw)
        self.epochs_expanded = int(epochs_expanded)
        if training_method not in {"direct", "iterative"}:
            raise ValueError("training_method must be direct or iterative")
        self.training_method = training_method
        self.layer_kwargs = layer_kwargs
        self.estimators = []
        self.base_prediction = 0.0
        self.n_features = None

    @property
    def lr_boost(self):
        return self.learning_rate

    @lr_boost.setter
    def lr_boost(self, value):
        self.learning_rate = float(value)

    @staticmethod
    def _validate_X(X, n_features=None):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2-D array")
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError(f"X has {X.shape[1]} features; expected {n_features}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X, y_cont, verbose=False):
        X = self._validate_X(X)
        y_cont = np.ascontiguousarray(y_cont, dtype=np.float64).reshape(-1)
        if len(y_cont) != len(X):
            raise ValueError("X and y_cont must have the same number of samples")
        if not np.all(np.isfinite(y_cont)):
            raise ValueError("y_cont must contain only finite values")

        self.n_features = X.shape[1]
        self.base_prediction = float(y_cont.mean())
        residual = y_cont - self.base_prediction
        self.estimators = []

        for step in range(self.n_estimators):
            step_nets = []
            step_preds = np.zeros(len(X), dtype=np.float64)
            for i in range(self.n_features):
                net = AdderNetLayer(**self.layer_kwargs)
                feature_residual = residual - step_preds
                if self.training_method == "direct":
                    net.fit(X[:, i], feature_residual, interpolate=True)
                else:
                    net.train(X[:, i], feature_residual,
                              epochs_raw=self.epochs_raw,
                              epochs_expanded=self.epochs_expanded)
                step_preds += net.predict_batch(X[:, i])
                step_nets.append(net)
            residual -= self.learning_rate * step_preds
            self.estimators.append(step_nets)
            if verbose:
                print(f"  Boost step {step + 1}/{self.n_estimators} | mean residual: {np.abs(residual).mean():.4f}")
        return self

    def predict_batch(self, X):
        if self.n_features is None or not self.estimators:
            raise RuntimeError("model is not fitted")
        X = self._validate_X(X, self.n_features)
        pred = np.full(len(X), self.base_prediction, dtype=np.float64)
        for step_nets in self.estimators:
            step_pred = np.zeros(len(X), dtype=np.float64)
            for i, net in enumerate(step_nets):
                step_pred += net.predict_batch(X[:, i])
            pred += self.learning_rate * step_pred
        return pred

    def predict(self, x):
        return float(self.predict_batch(np.asarray(x).reshape(1, -1))[0])
