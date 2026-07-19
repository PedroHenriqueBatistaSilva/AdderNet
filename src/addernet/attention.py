"""Multiplication-free additive attention."""

import numpy as np


class AdderAttention:
    """Attention based on negative L1 distance and additive value pooling.

    This layer is stateless. It accepts batched arrays ``Q``, ``K`` and ``V``
    and selects keys whose L1 score is above a fixed or per-query mean
    threshold. It intentionally does not expose ``fit`` or ``predict`` because
    it is an attention operation, not a classifier.
    """

    def __init__(self, threshold=None, normalize=False):
        self.threshold = threshold
        self.normalize = bool(normalize)

    def __call__(self, Q, K, V):
        return self.forward(Q, K, V)

    @staticmethod
    def _validate(Q, K, V):
        Q = np.asarray(Q)
        K = np.asarray(K)
        V = np.asarray(V)
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise ValueError("Q, K and V must be 3-D arrays: (batch, sequence, features)")
        if Q.shape[0] != K.shape[0] or K.shape[0] != V.shape[0]:
            raise ValueError("Q, K and V must have the same batch size")
        if Q.shape[2] != K.shape[2]:
            raise ValueError("Q and K must have the same feature dimension")
        if K.shape[1] != V.shape[1]:
            raise ValueError("K and V must have the same sequence length")
        if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(K)) and np.all(np.isfinite(V))):
            raise ValueError("Q, K and V must contain only finite values")
        return Q, K, V

    def scores(self, Q, K):
        Q = np.asarray(Q)
        K = np.asarray(K)
        if Q.ndim != 3 or K.ndim != 3 or Q.shape[0] != K.shape[0] or Q.shape[2] != K.shape[2]:
            raise ValueError("Q and K must be compatible 3-D arrays")
        return -np.sum(np.abs(Q[:, :, None, :] - K[:, None, :, :]), axis=-1)

    def forward(self, Q, K, V):
        Q, K, V = self._validate(Q, K, V)
        score = self.scores(Q, K)
        if self.threshold is None:
            mask = score >= np.mean(score, axis=-1, keepdims=True)
        else:
            mask = score >= self.threshold
        output = np.sum(np.where(mask[..., None], V[:, None, :, :], 0), axis=2)
        if self.normalize:
            count = np.maximum(mask.sum(axis=-1, keepdims=True), 1)
            output = output / count
        return output
