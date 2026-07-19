# AdderNet 1.6 development fork

This directory contains an audited and extended version of the `addernet` 1.5.0
package. It keeps the original scalar, HDC, attention, boosting, clustering, and
additive APIs while adding safer native bindings and genuine multi-input models.

## Main additions

- `AdderNetLayer.fit(...)`: direct O(n + range) scalar LUT fitting.
- `AdderNetLayer.partial_fit(...)`: blended streaming updates.
- Explicit `close()` and context-manager lifecycle.
- Atomic native saves and portable versioned `.npz` saves.
- Hardened native file loading and full validation of C arguments.
- `AdderNetMultiInputLayer`: N inputs, one or more outputs, additive and pairwise LUTs.
- `AdderNetRegressor`: alias for the multi-input regression model.
- `AdderNetClassifier`: classification wrapper with arbitrary class labels.
- Improved `UniformQuantizer` with feature-shape validation and inverse transform.
- OpenMP changed to opt-in because it slowed the memory-bound scalar lookup on the test host.

## Two-input example

```python
import numpy as np
from addernet import AdderNetMultiInputLayer

x = np.arange(32)
y = np.arange(32)
xx, yy = np.meshgrid(x, y, indexing="ij")
X = np.column_stack([xx.ravel(), yy.ravel()])
target = (xx * yy).ravel()

model = AdderNetMultiInputLayer(bins=32, interactions="auto")
model.fit(X, target)

print(model.predict(7, 9))  # approximately 63
print(model.predict_batch([[2, 3], [4, 5]]))
```

## Build and test

```bash
python -m pip install -e .
addernet-selftest
python -m pytest
```

Set `ADDERNET_OPENMP=1` before building only after benchmarking on the target
hardware. The default single-threaded native loop is often faster for this LUT.
