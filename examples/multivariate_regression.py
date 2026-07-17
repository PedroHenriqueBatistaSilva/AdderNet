"""Learn a multivariate additive function using scalar AdderNet layers."""
import numpy as np
from addernet import AdderNetAdditiveRegressor

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(500, 3))
y = 1.8 * X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 2])

model = AdderNetAdditiveRegressor(table_size=128, backfit_rounds=3,
                                  epochs_raw=50, epochs_expanded=100)
model.fit(X[:400], y[:400])
pred = model.predict(X[400:])
print("MAE:", np.mean(np.abs(pred - y[400:])))
