# AdderNet — Inference Without Multiplication

**Version 1.5.0** — Official Release

A machine learning library that performs inference without floating-point multiplications. Uses table lookups (LUT), integer addition, and Hyperdimensional Computing (HDC) — targeting embedded systems without FPUs (ESP32, STM32, Raspberry Pi).

> **Technical honesty**: a trained `AdderNetLayer` inference is `result = offset_table[(int(input) + bias) & mask]` — one indexed load, zero arithmetic. Quantization, HDC encoding, attention, preprocessing, and Python glue may use other operations. "Zero multiplication" is a property of the core LUT path, not of every possible pipeline.

---

## Components

| Component | Description | API |
|---|---|---|
| `AdderNetLayer` | Scalar LUT regressor — C native, ctypes binding | `train()`, `predict()`, `predict_batch()`, `save()`, `load()` |
| `ReferenceAdderNetLayer` | Pure NumPy equivalent for learning/debugging | Same API, optional `trace` callback |
| `UniformQuantizer` | Explicit continuous-to-integer binning | `fit()`, `transform()`, `fit_transform()` |
| `AdderNetAdditiveRegressor` | Multivariate GAM — one LUT per feature, backfitting | `fit()`, `predict()`, `save()`, `load()` |
| `AdderNetHDC` | Multivariate HDC classifier with GPU training | `train()`, `predict_batch()`, retraining |
| `AdderCluster` | Ensemble of LUT nodes (random/range/feature/boosting split) | `fit()`, `predict_batch()`, `predict_single_fast()` |
| `AdderBoost` | Gradient boosting with LUT base learners | `fit()`, `predict_batch()`, `predict()` |
| `AdderAttention` | Stateless L1-distance attention | `forward(Q, K, V)`, `scores(Q, K)` |

---

## Installation

```bash
pip install .
addernet-selftest
```

Auto-builds CPU libraries with `gcc`/`clang`/`cc` on import. Disable with:

```bash
ADDERNET_AUTOBUILD=0 python your_script.py
```

For native CPU optimizations (AVX2 on x86_64, NEON on ARM):

```bash
ADDERNET_NATIVE=1 addernet-build
```

---

## Quick Start

### Scalar LUT Regression

```python
from addernet import AdderNetLayer

layer = AdderNetLayer(size=256, bias=0, input_min=0, input_max=100, lr=0.1)
layer.train([0, 10, 20, 30], [32, 50, 68, 86])
print(layer.predict(25))  # 59.0
```

Inference is a single table lookup: `offset_table[(int(x) + bias) & mask]`.

### Reference Implementation (Pure Python)

```python
from addernet import ReferenceAdderNetLayer

layer = ReferenceAdderNetLayer(size=64, input_min=0, input_max=31, lr=0.25)
layer.train([0, 8, 16, 24, 31], [2, 14, 26, 38, 48.5])
print(layer.offset_table)
```

Slower but transparent — no `ctypes`, no C. Useful for studying the algorithm.

### Multivariate Additive Regression

```python
import numpy as np
from addernet import AdderNetAdditiveRegressor

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(500, 3))
y = 1.8 * X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 2])

model = AdderNetAdditiveRegressor(
    table_size=128, backfit_rounds=3,
    epochs_raw=50, epochs_expanded=100,
).fit(X[:400], y[:400])

print(np.mean(np.abs(model.predict(X[400:]) - y[400:])))
```

A **Generalized Additive Model** — one LUT per feature, fitted via backfitting. Provide interactions as explicit features.

### HDC Classification

```python
import numpy as np
from addernet import AdderNetHDC

X = np.array([[0, 0], [0, 1], [9, 10], [10, 9]], dtype=np.float64)
y = np.array([0, 0, 1, 1], dtype=np.int32)

model = AdderNetHDC(n_vars=2, n_classes=2, hv_dim=1024, seed=7)
model.train(X, y, n_iter=5, patience=0)
print(model.predict_batch(X))
```

Hyperdimensional Computing with OnlineHD training, AdaptHD retraining, and early-exit Hamming distance. Non-multiple-of-64 dimensions supported with bounds checks in AVX.

### Ensemble (AdderCluster)

```python
from addernet import AdderCluster
import numpy as np

X = np.random.rand(100, 4) * 100
y = np.random.randint(0, 3, size=100)

cluster = AdderCluster(n_nodes=3, strategy='range', combination='vote')
cluster.fit(X, y)
print(cluster.predict_batch(X[:5]))
```

Multiple partitioning strategies (`random`, `range`, `feature`, `boosting`) and combination methods (`vote`, `mean`, `stack`).

### Gradient Boosting (AdderBoost)

```python
from addernet import AdderBoost
import numpy as np

X = np.random.rand(100, 3) * 50
y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.3 * X[:, 2]

boost = AdderBoost(n_estimators=5, learning_rate=0.1,
                   size=256, bias=0, input_min=0, input_max=50, lr=0.05)
boost.fit(X, y)
print(boost.predict_batch(X[:5]))
```

### L1 Attention

```python
import numpy as np
from addernet import AdderAttention

Q = np.random.randn(2, 3, 5)
K = np.random.randn(2, 4, 5)
V = np.random.randn(2, 4, 6)

attn = AdderAttention(normalize=True)
output = attn(Q, K, V)  # shape: (2, 3, 6)
scores = attn.scores(Q, K)
```

Stateless attention based on negative L1 distance — no `fit()` or `predict()`.

### Quantization Preprocessor

```python
from addernet import UniformQuantizer
import numpy as np

q = UniformQuantizer(bins=256)
X = np.array([[1.5, 2.7], [3.1, 4.2], [5.0, 6.0]])
q.fit(X)
Xq = q.transform(X)  # integer indices [0..255]
```

---

## CUDA 2026 Backend

The HDC classifier includes GPU-accelerated training and inference:

| Variant | Architecture | Shared Memory | Kernel |
|---|---|---|---|
| `ampere` | sm_80+ (A100, H100, RTX 3090+) | 100 KB | Cooperative, warp-level ops |
| `turing` | sm_70–75 (V100, RTX 2080) | 64 KB | Standard parallel |
| `legacy` | sm_61 and below (Pascal, GTX 1060) | — | Basic CUDA |

- `use_gpu=True` enables GPU batch prediction
- `use_gpu_training=True` enables GPU-accelerated AdaptHD retraining
- Automatic fallback to CPU (AVX2/NEON/scalar) when CUDA is unavailable
- Multi-architecture compilation with `make cuda_2026`
- Capability-based kernel selection via `CUDADetector`

```python
from addernet import AdderNetHDC, hdc_detect_backend

model = AdderNetHDC(n_vars=4, n_classes=3, hv_dim=1024, use_gpu=True)
print(hdc_detect_backend())  # 'cpu', 'cuda', or 'cuda_2026'
```

### CUDA Detection

```python
from addernet import get_cuda_info

info = get_cuda_info()
if info:
    print(f"GPU: {info['gpu_name']} (sm_{info['capability_int']})")
    print(f"Kernel variant: {info['kernel_variant']}")
```

---

## CLI Tools

| Command | Description |
|---|---|
| `addernet-build` | Compile C/CUDA libraries at runtime |
| `addernet-selftest` | Run smoke test to verify installation |

---

## Serialization

| Component | Format | Methods |
|---|---|---|
| `AdderNetLayer` | Binary `.bin` | `save(path)`, `load(path)` |
| `AdderNetAdditiveRegressor` | JSON manifest + per-layer `.bin` | `save(directory)`, `load(directory)` |
| `ReferenceAdderNetLayer` | NumPy `.npz` | `save(path)`, `load(path)` |

---

## Testing

```bash
python -m pytest -q                    # fast unit tests
python benchmarks/validation_suite.py  # long validation suite
python test_validation.py              # scikit-learn dataset validation
```

C unit tests (requires Makefile):

```bash
make test          # build + run all C tests
make test_addernet # AdderNetLayer C tests only
make test_hdc      # HDC C tests only
```

---

## Project Structure

```text
addernet/
  __init__.py          public API, auto-build, version
  addernet.py          AdderNetLayer ctypes binding
  addernet_hdc.py      AdderNetHDC ctypes binding (+ GPU)
  reference.py         ReferenceAdderNetLayer + UniformQuantizer
  vector.py            AdderNetAdditiveRegressor
  attention.py         AdderAttention
  cluster.py           AdderCluster ensemble
  boost.py             AdderBoost gradient boosting
  build_ext.py         portable CPU compiler
  build_ext_2026.py    CUDA 2026 multi-arch compiler
  cuda_detector.py     GPU/capability detection via ctypes
  selftest.py          installation smoke test
  src/                 bundled C/CUDA sources for runtime build
src/                   C/CUDA source tree
  addernet.c/h         scalar LUT layer
  addernet_hdc.c/h     HDC classifier
  hdc_core.c/h         hypervector operations
  hdc_lsh.c/h          locality-sensitive hashing
  addernet_cuda.cu     GPU training kernels (nvcc)
  addernet_hdc_train_cuda.cu  AdaptHD retrain (nvcc)
  cuda_train/          Ampere+ cooperative kernels
examples/              executable demos
tests/                 fast unit tests
benchmarks/            long validation suite
APOSTILA_ADDERNET.html interactive course
```

---

## Platform Support

| Platform | Library | SIMD |
|---|---|---|
| Linux | `.so` | AVX2 (x86_64), NEON (ARM) |
| macOS | `.dylib` | Same as Linux |
| Windows | `.dll` | (manual compilation) |
| CUDA | `.so` | sm_61+ (nvcc optional) |

---

## Limitations

- `AdderNetLayer` accepts one scalar input per layer; fractional parts are truncated.
- `AdderNetAdditiveRegressor` is additive — no automatic interaction discovery.
- HDC trades precision for associative memory and cheap inference.
- Table size is capped at 256 entries (C constant `AN_TABLE_SIZE`).
- CUDA backend requires compatible GPU and toolchain.

---

## License

Apache-2.0. Original project by Pedro Henrique Batista Silva.
