"""Train a two-input, three-output AdderNet model."""
import numpy as np
from addernet import AdderNetMultiInputLayer

values = np.arange(16, dtype=np.float64)
x, y = np.meshgrid(values, values, indexing="ij")
X = np.column_stack([x.ravel(), y.ravel()])
targets = np.column_stack([
    (x + y).ravel(),
    (x - y).ravel(),
    (x * y).ravel(),
])

model = AdderNetMultiInputLayer(bins=16, interactions="auto").fit(X, targets)
print("f(7, 9) =", model.predict(7, 9))
print("training RMSE =", model.training_rmse_)
