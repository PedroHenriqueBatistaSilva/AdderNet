<div align="center">

# AdderNet

### Fast, explainable LUT-based additive models with multiplication-free inference in the quantized core

[![Version](https://img.shields.io/badge/version-1.6.0.dev0-111111.svg)](#project-status)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)](#requirements)
[![Backend](https://img.shields.io/badge/backend-C%20%2B%20NumPy-00599C.svg)](#architecture)
[![License](https://img.shields.io/badge/license-Apache--2.0-2ea44f.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-21%20passed-2ea44f.svg)](#tests)

**Scalar · multiple inputs · multiple outputs · classification · HDC · additive attention · boosting · ensembles**

[Quick start](#quick-start) · [Installation](#installation) · [Full guide](#full-guide) · [API](#api-reference) · [Performance](#performance) · [Limitations](#limitations)

</div>

---

## What is AdderNet?

AdderNet is a Python library with a native C core for building models based on **lookup tables**, or **LUTs**. Instead of performing a long sequence of matrix multiplications during inference, the model transforms inputs into discrete indices, retrieves previously learned values, and combines those values primarily through addition.

In version `1.6.0.dev0`, the library ranges from an extremely small scalar LUT to models with multiple inputs, multiple outputs, variable interactions, classification, ensembles, and Hyperdimensional Computing.

The core idea of a multivariate model is:

```text
prediction = intercept
           + sum of each feature's individual LUTs
           + sum of the selected interaction LUTs
```

In mathematical form:

```math
\hat{y}(x)=b+\sum_{j=1}^{p}T_j(q_j(x_j))+\sum_{(a,b)\in\mathcal{I}}T_{a,b}(q_a(x_a),q_b(x_b))
```

where:

- `q` quantizes continuous values into integer indices;
- `T_j` is a single-feature LUT;
- `T_{a,b}` is a joint LUT for the interaction between two features;
- `I` is the set of selected interactions;
- `b` is the learned intercept.

> [!IMPORTANT]
> AdderNet **is not a traditional dense neural network**. It belongs to a family of discretized, lookup-oriented additive models. This provides predictable inference, fast direct fitting, and good explainability, but it also imposes resolution and memory limits.

---

## Why use it?

AdderNet is especially useful when you need:

- **low CPU latency** for small and medium-sized models;
- **deterministic inference** based on lookups and sums;
- **explainable models**, with separate contributions for each feature and interaction;
- **incremental updates** for scalar LUTs;
- **multiple inputs and outputs** without immediately resorting to a dense network;
- a **compact implementation** for edge computing, automation, sensors, and embedded systems;
- **HDC classification** with binary hyperdimensional vectors;
- a simple alternative for approximating functions, calibrating sensors, or replacing expensive calculations already represented by data.

### Natural use cases

| Category | Examples |
|---|---|
| Calibration | sensor correction, actuator linearization, thermal compensation |
| Control | thermostats, simple robots, motors, automation rules |
| Regression | battery range, shipping cost, consumption, estimated time |
| Classification | anomalies, operating states, maintenance, discrete signals |
| Edge | local inference with low memory usage and predictable latency |
| Approximation | surrogate models for more expensive functions or simulations |
| HDC | binary multivariate classification with Hamming distance |

### When not to use it

Use another approach when:

- there is a simple, inexpensive, and exact formula for the problem;
- you need to extrapolate far beyond the training range;
- the problem requires high-order interactions among many features;
- the data consists of high-dimensional raw images, audio, or text;
- a GPU will process huge batches and a neural network is already highly optimized;
- the required resolution would make the LUTs too large.

---

## What's new in version 1.6.0.dev0

This release is an audited expansion of `1.5.0` and preserves the existing scalar, HDC, attention, boosting, cluster, and additive-regression APIs.

### Fixed

- Removed silent truncation after 256 samples in native interpolation.
- Duplicate integer inputs are now correctly aggregated by their mean.
- Hardened binary loading against invalid sizes, truncated files, and non-finite values.
- Added validation for Python arguments and C-core return codes.
- Added atomic native saving.
- Added safe `AdderNetLayer` lifecycle management.
- OpenMP is no longer enabled by default, avoiding overhead in small memory-bound loops.

### Added

- `AdderNetLayer.fit()` for direct LUT fitting.
- `AdderNetLayer.partial_fit()` for incremental updates with blending.
- `close()` and context-manager support.
- Portable, versioned `.npz` serialization.
- `AdderNetMultiInputLayer` for N inputs and one or more outputs.
- Automatic, all-pairs, disabled, or manual pairwise interactions.
- `AdderNetClassifier` with arbitrary labels.
- `UniformQuantizer` with shape-safe validation.
- `predict_quantized()`, `score()`, `explain()`, and `memory_bytes_`.
- Organization into `models/`, `preprocessing/`, `tests/`, `benchmarks/`, `examples/`, and `docs/`.

See the [CHANGELOG](CHANGELOG.md) as well.

---

## Project status

`1.6.0.dev0` is a **development release**. It was built and tested in the development environment on Linux x86-64 with CPython 3.13.

- Automated tests: **21/21 passed**.
- Original self-test: **PASS**.
- Included wheel: Linux x86-64, CPython 3.13.
- Automatic source build: Linux and macOS.
- Windows: requires a compatible wheel or manual compilation with an appropriate toolchain.
- CUDA/HDC: preserved and available when the environment provides the required components.

> [!WARNING]
> Before using AdderNet in production, run the tests and benchmarks on the project's actual hardware, compiler, operating system, and data distribution.

---

## Installation

### Requirements

- Python `>= 3.10`
- NumPy `>= 1.24`
- SciPy `>= 1.10`
- scikit-learn `>= 1.3`
- A C compiler such as GCC or Clang when no compatible precompiled native library is available

### Install the included wheel

The wheel in this distribution was built for **CPython 3.13 on Linux x86-64**:

```bash
python -m pip install addernet-1.6.0.dev0-cp313-cp313-linux_x86_64.whl
```

### Install from source

```bash
git clone https://github.com/PedroHenriqueBatistaSilva/AdderNet.git
cd AdderNet
python -m pip install .
```

### Install in development mode

```bash
git clone https://github.com/PedroHenriqueBatistaSilva/AdderNet.git
cd AdderNet
python -m pip install -e ".[dev]"
```

### Build the native libraries manually

```bash
addernet-build
```

When the package is imported, the CPU libraries are compiled automatically if they are missing, provided that a compatible compiler is available.

To disable automatic builds:

```bash
export ADDERNET_AUTOBUILD=0
```

### Verify the installation

```bash
python -c "import addernet; print(addernet.__version__)"
addernet-selftest
```

Expected version output:

```text
1.6.0.dev0
```

---

## Quick start

### Two inputs, one output, and a real interaction

The example below learns `z = x × y`. A purely additive model could not perfectly represent this function using only `f(x) + g(y)`, so the joint `(x, y)` LUT is important.

```python
import numpy as np
from addernet import AdderNetMultiInputLayer

# Complete training grid.
values = np.arange(32, dtype=np.float64)
xx, yy = np.meshgrid(values, values, indexing="ij")

X = np.column_stack([xx.ravel(), yy.ravel()])
y = (xx * yy).ravel()

model = AdderNetMultiInputLayer(
    bins=32,
    interactions="auto",
    max_interactions=1,
).fit(X, y)

print(model.predict(7, 9))
# Approximately 63.0

print(model.predict_batch([[2, 3], [4, 5]]))
# [ 6. 20.]

print(model.score(X, y))
# 1.0 on a fully observed grid
```

### Classification in a few lines

```python
import numpy as np
from addernet import AdderNetClassifier

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=np.float64)

y = np.array(["same", "different", "different", "same"])

classifier = AdderNetClassifier(
    bins=2,
    interactions="auto",
    max_interactions=1,
).fit(X, y)

print(classifier.predict(X))
print(classifier.predict_proba(X))
print(classifier.score(X, y))
```

---

## How inference works

### Scalar path

For `AdderNetLayer`, the input is converted into an index, combined with the bias, and mapped to a cell in the native LUT.

```text
input value
      │
      ▼
conversion to an integer index
      │
      ▼
add bias and apply mask
      │
      ▼
look up a LUT position
      │
      ▼
predicted value
```

### Multivariate path

For `AdderNetMultiInputLayer`:

1. Each continuous feature is quantized.
2. One individual LUT is queried per feature.
3. Selected pairs are combined into a joint index.
4. The contributions and intercept are summed.

The `predict_quantized()` path skips continuous quantization and receives ready-to-use indices. The prediction core then uses lookups, integer operations, and additions.

> [!NOTE]
> Floating-point input still needs to be quantized. Therefore, the claim "multiplication-free" applies to the **quantized inference core**, not necessarily to the entire pipeline that receives continuous numbers.

---

## Full guide

## 1. Native scalar layer: `AdderNetLayer`

Use `AdderNetLayer` when there is exactly **one input and one output**.

```python
from addernet import AdderNetLayer

layer = AdderNetLayer(
    size=256,
    bias=50,
    input_min=-50,
    input_max=200,
    lr=0.1,
)
```

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `size` | `256` | Number of LUT cells. Must be a power of two. |
| `bias` | `50` | Offset applied to the internal index. |
| `input_min` | `-50` | Smallest coordinate expected during fitting/interpolation. |
| `input_max` | `200` | Largest coordinate expected during fitting/interpolation. |
| `lr` | `0.1` | Learning rate used by the legacy iterative training method. |

### Recommended direct fitting

```python
import numpy as np
from addernet import AdderNetLayer

x = np.array([0, 5, 10, 15], dtype=np.float64)
y = np.array([0, 10, 20, 30], dtype=np.float64)

layer = AdderNetLayer(
    size=32,
    bias=0,
    input_min=0,
    input_max=15,
).fit(x, y)

print(layer.predict(7))
print(layer.predict_batch([0, 3, 7, 15]))
```

`fit()`:

- aggregates duplicate coordinates by their mean;
- fills observed cells;
- can interpolate unobserved intervals;
- does not run thousands of epochs;
- returns `self`.

### Disable interpolation

```python
layer.fit(x, y, interpolate=False)
```

With `interpolate=False`, only directly observed cells are updated.

### Incremental updates

`partial_fit` is an alias of `fit` and allows a new batch to be blended with the current LUT.

```python
layer.partial_fit(
    inputs=[15],
    targets=[45],
    interpolate=False,
    blend=0.25,
)
```

The updated value approximately follows:

```text
new = (1 - blend) × old_value + blend × new_target
```

`blend` rules:

- `1.0`: completely replaces the updated cells;
- between `0` and `1`: gradual adaptation;
- `0` or values above `1`: invalid.

### Legacy iterative training

```python
layer.train(
    inputs=x,
    targets=y,
    epochs_raw=1000,
    epochs_expanded=4000,
)
```

Prefer `fit()` for most LUTs. Use `train()` only for compatibility or experiments with the original iterative algorithm.

### Use a context manager

```python
from addernet import AdderNetLayer

with AdderNetLayer(size=64, bias=0, input_min=0, input_max=63) as layer:
    layer.fit([0, 63], [0, 126])
    print(layer.predict(20))

# The native pointer was released when the block exited.
```

After `close()`, any inference call raises `RuntimeError`.

### Read or replace the LUT

```python
table = layer.offset_table
print(table.shape)

layer.set_offset_table(table)
```

The table passed to `set_offset_table()` must contain exactly `layer.size` elements.

### Learned parameters and metadata

```python
print(layer.get_params())
print(layer.size)
print(layer.bias)
print(layer.input_min)
print(layer.input_max)
print(layer.lr)
```

### Native serialization

```python
layer.save("models/calibration.bin")
loaded = AdderNetLayer.load("models/calibration.bin")
```

The native format is compact, but more dependent on the platform and implementation.

### Portable serialization

```python
layer.save_portable("models/calibration.npz")
loaded = AdderNetLayer.load_portable("models/calibration.npz")
```

Prefer `.npz` for model transfer, inspection, and longevity.

---

## 2. Quantization: `UniformQuantizer`

`UniformQuantizer` converts continuous features into indices between `0` and `bins - 1`.

```python
import numpy as np
from addernet import UniformQuantizer

X = np.array([
    [0.0, -10.0],
    [5.0,   5.0],
    [10.0, 20.0],
])

quantizer = UniformQuantizer(bins=32)
Q = quantizer.fit_transform(X)
X_approx = quantizer.inverse_transform(Q)

print(Q)
print(X_approx)
```

### Accepted shapes

| Input | Interpretation |
|---|---|
| scalar | one sample from an already-fitted single feature |
| 1-D vector in a single-feature model | multiple samples |
| 1-D vector with `n_features` values | one multivariate sample |
| 2-D matrix | `(samples, features)` |

### Clipping

By default, values outside the observed range are clipped to the boundaries:

```python
Q = quantizer.transform(X_new, clip=True)
```

To reject them explicitly:

```python
Q = quantizer.transform(X_new, clip=False)
```

### Metadata serialization

```python
metadata = quantizer.to_dict()
restored = UniformQuantizer.from_dict(metadata)
```

---

## 3. Multivariate regression: `AdderNetMultiInputLayer`

Also exported as:

```python
from addernet import AdderNetRegressor
```

Both references point to the same class.

### Constructor

```python
model = AdderNetMultiInputLayer(
    bins=64,
    interactions="auto",
    max_interactions=8,
    backfit_rounds=3,
    interaction_rounds=1,
    min_samples_per_cell=1,
)
```

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `bins` | `64` | Resolution per feature. Must be a power of two between 2 and 4096. |
| `interactions` | `"auto"` | Pair selection: `auto`, `all`, `none`, or a manual list. |
| `max_interactions` | `8` | Maximum number of joint LUTs. |
| `backfit_rounds` | `3` | Passes used to fit individual contributions. |
| `interaction_rounds` | `1` | Passes used to fit joint LUTs. |
| `min_samples_per_cell` | `1` | Minimum occupancy required for a cell to contribute. |

### Single output

```python
import numpy as np
from addernet import AdderNetRegressor

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(5000, 3))
y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]

model = AdderNetRegressor(
    bins=64,
    interactions="none",
    backfit_rounds=4,
).fit(X, y)

predictions = model.predict_batch(X[:10])
print(model.score(X, y))
```

### Multiple outputs

```python
x = X[:, 0]
y_feature = X[:, 1]

targets = np.column_stack([
    x + y_feature,
    x - y_feature,
    x * y_feature,
])

model = AdderNetMultiInputLayer(
    bins=64,
    interactions="auto",
).fit(X[:, :2], targets)

print(model.predict(2.0, 3.0))
# Array with three outputs
```

### Prediction forms

```python
# One sample, separate arguments.
model.predict(2.0, 3.0)

# One sample, vector.
model.predict([2.0, 3.0])

# Multiple samples.
model.predict_batch([
    [2.0, 3.0],
    [4.0, 5.0],
])

# Broadcasting with separate arguments.
model.predict(
    np.array([2.0, 4.0]),
    np.array([3.0, 5.0]),
)
```

### Choose interactions

#### Automatic

```python
model = AdderNetMultiInputLayer(
    bins=32,
    interactions="auto",
    max_interactions=4,
)
```

With two features and available capacity, `auto` selects the pair `(0, 1)`. With more features, pairs are evaluated by their approximate residual gain.

#### All allowed interactions

```python
model = AdderNetMultiInputLayer(
    bins=32,
    interactions="all",
    max_interactions=6,
)
```

`max_interactions` still limits the total number.

#### No interactions

```python
model = AdderNetMultiInputLayer(
    bins=64,
    interactions="none",
)
```

Ideal for approximately additive functions and smaller models.

#### Manual list

```python
model = AdderNetMultiInputLayer(
    bins=32,
    interactions=[
        (0, 1),
        (0, 3),
        (2, 4),
    ],
    max_interactions=3,
)
```

The indices refer to the columns of `X`.

### Already-quantized inference

```python
Q = model.quantizer.transform(X_new)
raw_output = model.predict_quantized(Q)
```

The output of `predict_quantized()` is always a `(samples, outputs)` matrix, even when there is only one output.

### Explainability

```python
explanation = model.explain([[2.0, 3.0], [5.0, 6.0]])

print(explanation.keys())
# intercept, additive, interaction, prediction
```

Shapes:

| Field | Shape |
|---|---|
| `intercept` | `(samples, outputs)` |
| `additive` | `(samples, features, outputs)` |
| `interaction` | `(samples, interactions, outputs)` |
| `prediction` | `(samples, outputs)` |

The exact reconstruction is:

```python
reconstructed = (
    explanation["intercept"]
    + explanation["additive"].sum(axis=1)
    + explanation["interaction"].sum(axis=1)
)
```

### Metrics and attributes

```python
print(model.score(X, y))             # Global R²
print(model.training_rmse_)          # Fitting RMSE
print(model.interaction_pairs_)      # Selected pairs
print(model.n_features_in_)
print(model.n_outputs_)
print(model.memory_bytes_)
```

### Save and load

```python
model.save("models/regressor.npz")
loaded = AdderNetMultiInputLayer.load("models/regressor.npz")
```

---

## 4. Classification: `AdderNetClassifier`

`AdderNetClassifier` transforms the problem into multiple outputs, one per class, and selects the highest score.

```python
from addernet import AdderNetClassifier

classifier = AdderNetClassifier(
    bins=64,
    interactions="auto",
    max_interactions=4,
    backfit_rounds=3,
).fit(X_train, y_train)
```

Labels can be:

- non-contiguous integers;
- strings;
- values compatible with `numpy.unique` and pickle-free NumPy serialization.

### Predict classes

```python
y_pred = classifier.predict(X_test)
```

### Raw scores

```python
scores = classifier.decision_function(X_test)
```

### Normalized probabilities

```python
probabilities = classifier.predict_proba(X_test)
```

The probabilities use softmax over the LUT scores. They are useful as a relative measure, but should not automatically be treated as calibrated probabilities.

### Accuracy

```python
accuracy = classifier.score(X_test, y_test)
```

### Save and load

```python
classifier.save("models/network_guard")
loaded = AdderNetClassifier.load("models/network_guard")
```

The directory contains:

```text
network_guard/
├── classifier.json
├── classes.npy
└── model.npz
```

---

## 5. Additive regressor built from scalar layers: `AdderNetAdditiveRegressor`

This model creates one native `AdderNetLayer` for every output-feature combination.

```python
from addernet import AdderNetAdditiveRegressor

model = AdderNetAdditiveRegressor(
    table_size=256,
    lr=0.05,
    backfit_rounds=2,
    epochs_raw=40,
    epochs_expanded=80,
    training_method="direct",
).fit(X, y)

predictions = model.predict(X_test)
```

Main difference:

| Model | Implementation | Automatic interactions | Multiple outputs |
|---|---|---:|---:|
| `AdderNetMultiInputLayer` | NumPy arrays of LUTs | yes | yes |
| `AdderNetAdditiveRegressor` | multiple native `AdderNetLayer` instances | no | yes |

Use `AdderNetMultiInputLayer` as the general-purpose choice. Use `AdderNetAdditiveRegressor` when you explicitly want to compose the model from native scalar layers and do not need internal interactions.

For manual interactions, create additional features:

```python
X_extended = np.column_stack([
    X,
    X[:, 0] * X[:, 1],
])
```

### Serialization

```python
model.save("models/additive")
loaded = AdderNetAdditiveRegressor.load("models/additive")
```

---

## 6. Additive gradient boosting: `AdderBoost`

`AdderBoost` trains multiple residual-correction stages. Each stage contains one `AdderNetLayer` per feature.

```python
from addernet import AdderBoost

model = AdderBoost(
    n_estimators=10,
    learning_rate=0.1,
    training_method="direct",
    size=256,
    bias=50,
    input_min=-50,
    input_max=200,
    lr=0.1,
).fit(X_train, y_train, verbose=True)

batch = model.predict_batch(X_test)
single = model.predict(X_test[0])
```

### Main parameters

| Parameter | Default | Description |
|---|---:|---|
| `n_estimators` | `10` | Number of boosting stages. |
| `learning_rate` | `0.1` | Weight of each residual stage. |
| `training_method` | `direct` | `direct` or `iterative`. |
| `epochs_raw` | `1000` | Used only by iterative training. |
| `epochs_expanded` | `4000` | Used only by iterative training. |
| `**layer_kwargs` | — | Parameters passed to each `AdderNetLayer`. |

`lr_boost` still works, but is deprecated. Use `learning_rate`.

---

## 7. Classification ensemble: `AdderCluster`

`AdderCluster` creates multiple nodes. Each node contains one `AdderNetLayer` per feature, and their results are combined to produce classes.

```python
from addernet import AdderCluster

cluster = AdderCluster(
    n_nodes=4,
    strategy="range",
    combination="vote",
    input_min=0,
    input_max=150,
    size=256,
    bias=50,
    lr=0.05,
    training_method="direct",
    random_state=42,
).fit(X_train, y_train)

predictions = cluster.predict_batch(X_test)
print(cluster.predict(X_test[0]))
cluster.info()
```

### Partitioning strategies

| Strategy | Behavior |
|---|---|
| `random` | randomly divides samples among nodes |
| `range` | partitions by normalized range with approximate class balancing |
| `feature` | gives all data to all nodes |
| `boosting` | creates random partitions guided by the existing boosting structure |

### Combination methods

| Combination | Behavior |
|---|---|
| `vote` | each node selects a class and the majority wins |
| `mean` | averages the scores and selects the nearest target |
| `stack` | currently combines by averaging scores before class snapping |

The model accepts arbitrary labels and preserves them in `classes_`.

---

## 8. Additive attention: `AdderAttention`

`AdderAttention` uses negative L1 distance between queries and keys. It then selects values above a threshold and sums the corresponding `V` vectors.

```python
import numpy as np
from addernet import AdderAttention

Q = np.random.default_rng(0).normal(size=(2, 4, 8))
K = np.random.default_rng(1).normal(size=(2, 6, 8))
V = np.random.default_rng(2).normal(size=(2, 6, 16))

attention = AdderAttention(
    threshold=None,
    normalize=True,
)

scores = attention.scores(Q, K)
output = attention(Q, K, V)

print(scores.shape)  # (2, 4, 6)
print(output.shape)  # (2, 4, 16)
```

### Parameters

- `threshold=None`: uses the mean score of each query as the threshold.
- `threshold=<number>`: uses a fixed threshold.
- `normalize=False`: sums the selected values.
- `normalize=True`: divides the sum by the number of selected values.

Required shapes:

```text
Q: (batch, query_sequence, features)
K: (batch, key_sequence, features)
V: (batch, key_sequence, value_features)
```

This layer is stateless and has no `fit()` method.

---

## 9. Hyperdimensional Computing: `AdderNetHDC`

`AdderNetHDC` is a multivariate classifier that combines AdderNet encoding with Hyperdimensional Computing.

```python
import numpy as np
from addernet import AdderNetHDC

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

model = AdderNetHDC(
    n_vars=4,
    n_classes=2,
    table_size=256,
    hv_dim=2500,
    seed=42,
    use_gpu=False,
    use_gpu_training=False,
)

history = model.train(
    X,
    y,
    n_iter=20,
    lr=1.0,
    margin="5%",
    regenerate=0.02,
    patience=5,
    interactions=2,
    verbose=True,
)

print(history)
print(model.predict(X[0]))
print(model.predict_batch(X[:32]))
print(model.accuracy(X, y))
```

### Constructor

| Parameter | Default | Description |
|---|---:|---|
| `n_vars` | `1` | Number of input variables. |
| `n_classes` | `2` | Number of classes, encoded from `0` to `n_classes - 1`. |
| `table_size` | `256` | Table size per variable; must be a power of two. |
| `bias` | `None` | List containing one bias per variable. |
| `seed` | `42` | Seed for hyperdimensional generation. |
| `use_gpu` | `False` | Requests CUDA for batch inference. |
| `hv_dim` | `2500` | Hypervector dimensionality. |
| `use_gpu_training` | `False` | Requests the CUDA refinement kernel. |

### Training

| Parameter | Default | Description |
|---|---:|---|
| `n_iter` | `0` | Number of correction passes; zero performs single-pass training. |
| `lr` | `1.0` | Learning rate for iterative correction. |
| `margin` | `0` | Margin in bits, as a fraction, as a percentage, or zero. |
| `regenerate` | `0.0` | Dimension-regeneration rate. |
| `patience` | `10` | Early stopping; zero disables it. |
| `verbose` | `False` | `True` or an integer logging interval. |
| `interactions` | `0` | Number of correlated pairs to encode. |

Valid margin forms:

```python
margin=0       # disabled
margin=0.05    # 5% of hv_dim
margin="5%"    # 5% of hv_dim
margin=125     # 125 bits
```

The margin is internally capped at 20% of `hv_dim`.

### Batch inference modes

```python
pred = model.predict_batch(X)
pred_avx = model.predict_batch_avx(X)
pred_mt = model.predict_batch_mt(X, n_threads=0)
```

- `predict_batch`: standard CPU or CUDA path, depending on `use_gpu`.
- `predict_batch_avx`: requests the AVX2 implementation available in the core.
- `predict_batch_mt`: uses multiple threads; `0` lets the backend decide.

Always test on the target hardware. The theoretically more parallel path may not be faster for small batches.

### Cache, Hadamard, and LSH

```python
model.warm_cache()
model.set_cache(True)
model.set_threads(4)
model.set_hadamard(True)

model.build_lsh(k=10, l=8)
model.set_lsh(True)
```

### Top-K

```python
top_classes = model.predict_top_k(X[0], k=5)
```

### Hypervector operations

```python
codebook = model.codebook

bundled = model.bundle_classes([0, 1])
noisy = model.add_noise(bundled, temperature=0.02)
class_index = model.classify_hv(noisy)
```

### Metadata

```python
print(model.n_vars)
print(model.n_classes)
print(model.table_size)
print(model.hv_dim)
print(model.hv_words)
print(model.codebook.shape)
```

### Save and load

```python
model.save("models/hdc.bin")
loaded = AdderNetHDC.load("models/hdc.bin")
```

### Available backend

```python
from addernet import hdc_detect_backend, get_cuda_info

print(hdc_detect_backend())
# SCALAR, AVX2, NEON, or UNKNOWN

print(get_cuda_info())
# Information dictionary or None
```

---

## 10. Reference implementation: `ReferenceAdderNetLayer`

`ReferenceAdderNetLayer` reproduces the scalar layer in pure NumPy. It is useful for:

- understanding the algorithm;
- debugging differences from the native backend;
- comparing results;
- running in environments where a native extension is not desired.

```python
from addernet import ReferenceAdderNetLayer

reference = ReferenceAdderNetLayer(
    size=32,
    bias=0,
    input_min=0,
    input_max=31,
).fit([0, 10, 31], [0, 20, 62])

print(reference.predict(7))
print(reference.predict_batch([0, 7, 31]))
```

The interface includes:

- `fit()` and `partial_fit()`;
- iterative `train()`;
- `predict()` and `predict_batch()`;
- `save()` and `load()`.

It prioritizes readability and portability, not maximum performance.

---

## API reference

### Public imports

```python
from addernet import (
    AdderNetLayer,
    AdderNetMultiInputLayer,
    AdderNetRegressor,
    AdderNetClassifier,
    UniformQuantizer,
    AdderNetAdditiveRegressor,
    AdderBoost,
    AdderCluster,
    AdderAttention,
    AdderNetHDC,
    AnHdcModel,
    ReferenceAdderNetLayer,
    hdc_detect_backend,
    get_cuda_info,
    set_verbose,
    is_verbose,
)
```

`AnHdcModel` is a compatibility alias for `AdderNetHDC`.

### Class summary

| Class | Problem | Input | Output | Main backend |
|---|---|---|---|---|
| `AdderNetLayer` | scalar regression | 1 feature | 1 value | native C |
| `AdderNetMultiInputLayer` | regression | N features | 1 or N values | NumPy + LUTs |
| `AdderNetClassifier` | classification | N features | labels | multivariate model |
| `UniformQuantizer` | preprocessing | continuous | integer indices | NumPy |
| `AdderNetAdditiveRegressor` | additive regression | N features | 1 or N values | multiple C layers |
| `AdderBoost` | residual regression | N features | 1 value | multiple C layers |
| `AdderCluster` | ensemble classification | N features | label | multiple C layers |
| `AdderAttention` | L1 attention | Q, K, V | tensor | NumPy |
| `AdderNetHDC` | HDC classification | N variables | class | C, SIMD, and optional CUDA |
| `ReferenceAdderNetLayer` | scalar reference | 1 feature | 1 value | NumPy |

### Verbosity

```python
from addernet import set_verbose, is_verbose

set_verbose(False)
print(is_verbose())
```

The equivalent environment variable is:

```bash
export ADDERNET_VERBOSE=0
```

---

## Choosing the right model

```text
One input and one output?
├─ Yes → AdderNetLayer
└─ No
   ├─ Regression with N inputs?
   │  ├─ Needs interactions → AdderNetMultiInputLayer
   │  └─ Additive effects only → MultiInputLayer or AdditiveRegressor
   ├─ Standard classification → AdderNetClassifier
   ├─ Ensemble classification → AdderCluster
   ├─ HDC classification → AdderNetHDC
   ├─ Residual correction for regression → AdderBoost
   └─ Attention over sequences → AdderAttention
```

### Rule of thumb

Start with:

```python
AdderNetMultiInputLayer(bins=64, interactions="auto")
```

Then:

1. evaluate validation error;
2. decrease or increase `bins`;
3. compare `interactions="none"` with `"auto"`;
4. monitor `memory_bytes_`;
5. test `predict_quantized()` when the data can arrive already discretized.

---

## Choosing `bins`

`bins` controls quantization resolution.

| Bins | Typical characteristics |
|---:|---|
| `8–16` | tiny models, low resolution |
| `32–64` | good starting point |
| `128–256` | higher precision, more memory, and greater data requirements |
| `512+` | use only when data coverage and memory requirements justify it |

Because it must be a power of two:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512, ...
```

More bins do not guarantee better generalization. With limited data, many cells remain empty.

### Memory impact

For `float64`:

```text
additive LUT ≈ outputs × features × bins × 8 bytes
joint LUT    ≈ outputs × interactions × bins² × 8 bytes
```

Example with 3 outputs, 5 features, 64 bins, and 4 interactions:

```text
additive     = 3 × 5 × 64 × 8      = 7,680 bytes
interactions = 3 × 4 × 64² × 8     = 393,216 bytes
```

Interactions quickly dominate memory consumption.

---

## Data preparation

### Separate training, validation, and test sets

Quantization learns minimum and maximum values during `fit`. Therefore, fit the model only on the training set.

```python
model.fit(X_train, y_train)
validation_score = model.score(X_validation, y_validation)
test_score = model.score(X_test, y_test)
```

### Values outside the range

In the multivariate workflow, the quantizer clips new values to the boundaries by default. This prevents invalid indices, but it also means the model **does not extrapolate freely**.

When extrapolation matters:

- expand the training range;
- apply a domain transformation;
- create an explicit fallback rule;
- consider another model.

### Categorical features

Encode categories as numbers before training. For unordered categories, avoid imposing an artificial ordinal distance without evaluating its effect.

Options include:

- one-hot encoding;
- separate codes per feature;
- controlled hashing;
- HDC for certain discrete representations.

### Missing values

AdderNet rejects `NaN` and infinity. Impute missing values first:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_clean = imputer.fit_transform(X)
```

---

## Performance

The figures below were measured on the local audit host. They are **not universal guarantees**.

| Measurement | 1.5.0 | 1.6.0.dev0 | Local result |
|---|---:|---:|---:|
| Scalar iterative training, 2,000 samples | 5.798 ms | 4.983 ms | 1.16× |
| Scalar direct fitting, 2,000 samples | — | 0.283 ms | 20.45× vs. original training |
| Inference on 1 million values | 5.393 ms | 2.304 ms | 2.34× |
| Scalar throughput | 185.4 million/s | 434.0 million/s | +134% |

Extended benchmark:

- direct fitting with 20,000 samples: **14.44×** faster than the new version's iterative method;
- multivariate fitting with 50,000 samples: **7.70 ms**;
- training RMSE in the multivariate benchmark: **0.0142**.

Raw files:

```text
benchmarks/original_1.5.0.json
benchmarks/enhanced_1.6.0.json
benchmarks/full_benchmark.json
```

### How to measure correctly

```python
import time

# Warm up the inference path.
model.predict_batch(X[:1000])

start = time.perf_counter()
for _ in range(100):
    model.predict_batch(X)
elapsed = time.perf_counter() - start

throughput = 100 * len(X) / elapsed
print(f"{throughput:,.0f} predictions/s")
```

Include the following in a real benchmark:

- quantization;
- dtype conversion;
- CPU ↔ GPU transfer;
- actual batch size;
- data loading;
- concurrency;
- p50, p95, and p99 latency.

### OpenMP

The scalar core is small and memory-bound. On the test host, starting many threads made inference slower. For this reason, OpenMP is opt-in:

```bash
export ADDERNET_OPENMP=1
python -m pip install --force-reinstall .
```

Enable it only after measuring on the target hardware.

### Local CPU optimization

```bash
export ADDERNET_NATIVE=1
python -m pip install --force-reinstall .
```

This may add `-march=native` on x86-64. The resulting binary may no longer work on older CPUs.

---

## Serialization and deployment

### Recommendation by model

| Model | Method | Format |
|---|---|---|
| `AdderNetLayer` | `save_portable` | versioned `.npz` file |
| `AdderNetLayer` | `save` | compact native binary |
| `AdderNetMultiInputLayer` | `save` | `.npz` file |
| `AdderNetClassifier` | `save` | directory containing JSON, NPY, and NPZ files |
| `AdderNetAdditiveRegressor` | `save` | directory containing a manifest and binary layers |
| `AdderNetHDC` | `save` | native binary |

### Best practices

- save the library version with the model;
- preserve the feature schema and order;
- record units and transformations;
- validate a set of predictions after loading;
- use checksums when distributing artifacts;
- do not load files from unknown sources without operational validation.

Example external manifest:

```json
{
  "addernet_version": "1.6.0.dev0",
  "model_type": "AdderNetMultiInputLayer",
  "features": ["temperature", "humidity", "occupancy"],
  "target": "cooling_power",
  "units": {
    "temperature": "celsius",
    "humidity": "percent",
    "cooling_power": "percent"
  }
}
```

---

## Architecture

```text
src/addernet/
├── __init__.py                 # Public API and protected loading
├── addernet.py                 # ctypes bindings for the scalar layer
├── addernet_hdc.py             # HDC wrapper, SIMD, and optional CUDA
├── attention.py                # L1-distance attention
├── boost.py                    # boosting with scalar LUTs
├── cluster.py                  # classification ensemble
├── vector.py                   # additive regressor with native layers
├── reference.py                # reference NumPy implementation
├── cuda_detector.py            # CUDA capability detection
├── build_ext.py                # portable compiler for CPU libraries
├── models/
│   ├── multi_input.py          # N inputs, N outputs, and interactions
│   └── classification.py       # multiclass classifier
├── preprocessing/
│   └── quantization.py         # shape-safe uniform quantizer
└── src/
    ├── addernet.c/.h           # hardened scalar core
    ├── addernet_hdc.c/.h       # HDC core
    ├── hdc_core.c/.h           # hyperdimensional operations
    ├── hdc_lsh.c/.h            # LSH index
    └── *.cu                    # optional CUDA kernels
```

### Dependencies by path

```text
AdderNetLayer
└── ctypes → libaddernet.so/.dylib/.dll

AdderNetMultiInputLayer
└── NumPy → quantization + additive LUTs + joint LUTs

AdderNetClassifier
└── AdderNetMultiInputLayer

AdderNetHDC
├── libaddernet_hdc
├── SIMD detected at runtime
└── optional CUDA
```

---

## Native build

The command:

```bash
addernet-build
```

builds:

```text
libaddernet.so
libaddernet_hdc.so
```

On macOS, the extension is `.dylib`.

### Environment variables

| Variable | Default | Effect |
|---|---:|---|
| `ADDERNET_AUTOBUILD` | `1` | builds missing CPU libraries during import |
| `ADDERNET_VERBOSE` | `1` | controls package messages |
| `ADDERNET_OPENMP` | `0` | enables OpenMP in the CPU build |
| `ADDERNET_NATIVE` | `0` | enables optimizations specific to the local CPU |
| `CC` | automatic | selects the C compiler |

Example:

```bash
CC=clang ADDERNET_NATIVE=1 ADDERNET_VERBOSE=0 addernet-build
```

---

## Tests

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run:

```bash
python -m pytest
```

Run the self-test:

```bash
addernet-selftest
```

The suite covers:

- direct scalar fitting;
- duplicates and interpolation;
- removal of the 256-sample limit;
- incremental updates;
- context-manager use and use after `close()`;
- native and portable serialization;
- rejection of corrupted files;
- equivalence between the native and reference backends;
- quantization of one or multiple features;
- multiple inputs and outputs;
- interactions, XOR, and product;
- explanations that reconstruct the prediction;
- classification with non-contiguous labels;
- additive regression, boosting, and clustering.

---

## Benchmarks

```bash
python benchmarks/benchmark_core.py
```

When publishing results, report:

- CPU and architecture;
- RAM;
- operating system;
- compiler and flags;
- Python and NumPy versions;
- LUT size;
- number of features and interactions;
- batch size;
- whether quantization was included in the measurement.

---

## Error handling

The current version explicitly rejects:

- empty arrays in fitting operations that require data;
- `NaN` and infinity;
- incompatible shapes;
- LUT sizes that are not powers of two;
- `blend` values outside `(0, 1]`;
- negative epoch counts;
- quantized indices outside `[0, bins)`;
- loaded models with invalid metadata;
- use of the scalar layer after `close()`.

Example:

```python
try:
    model.predict_batch([[float("nan"), 2.0]])
except ValueError as error:
    print(error)
```

---

## Troubleshooting

### `Cannot load the AdderNet native library`

Run:

```bash
addernet-build
```

Check whether GCC or Clang is available:

```bash
cc --version
# or
gcc --version
# or
clang --version
```

### `Missing native AdderNet libraries`

Automatic builds may be disabled:

```bash
unset ADDERNET_AUTOBUILD
# or
export ADDERNET_AUTOBUILD=1
```

### The wheel will not install

The included wheel is specific to CPython 3.13 on Linux x86-64. Install from source on other configurations:

```bash
python -m pip install .
```

### The multivariate model uses too much memory

Reduce:

- `bins`;
- `max_interactions`;
- the number of outputs;
- the manual pair list.

Monitor:

```python
print(model.memory_bytes_)
```

### Validation is much worse than training

Possible causes:

- too many bins for too few samples;
- sparse interactions;
- an out-of-range distribution;
- data leakage;
- too many cells observed only once.

Try:

```python
AdderNetMultiInputLayer(
    bins=32,
    interactions="none",
    min_samples_per_cell=3,
)
```

### `predict_batch_avx` fails

Use the standard path and inspect the backend:

```python
from addernet import hdc_detect_backend
print(hdc_detect_backend())
```

AVX2 is not available on every CPU.

### CUDA was requested but did not load

```python
from addernet import get_cuda_info
print(get_cuda_info())
```

Confirm:

- the NVIDIA driver;
- the CUDA toolkit;
- `nvcc` in `PATH`;
- GPU compatibility;
- generated libraries in the package.

---

## Limitations

### Discrete resolution

The model approximates functions on a grid. Differences smaller than one bin's resolution may produce the same output.

### Extrapolation

Values outside the fitted domain are normally clipped to the boundaries. AdderNet is strongest at interpolation and lookup within the known domain.

### Quadratic growth of interactions

Each interaction uses `bins²` cells. With `bins=256`, a single joint LUT with one output uses:

```text
256² × 8 bytes = 512 KiB
```

Ten interactions and four outputs would already require approximately 20 MiB for the joint tables alone.

### Second-order interactions

`AdderNetMultiInputLayer` models pairs. Explicit interactions among three or more features are not created automatically.

You can create composite features manually, but this should be done carefully to avoid increasing dimensionality and introducing information leakage.

### Unobserved cells

Sparse joint LUTs may contain many zero-valued cells. Increasing the number of bins without increasing data coverage can hurt generalization.

### Uncalibrated probabilities

`AdderNetClassifier.predict_proba()` applies softmax to the scores. Use external calibration when absolute probabilities matter.

### Binary compatibility

The native format is more platform-dependent. For the scalar layer, prefer `save_portable()` when portability is a priority.

### A GPU is not automatically faster

For small batches, data-transfer and kernel-launch costs may exceed CPU processing time.

---

## Migrating from 1.5.0

Legacy imports remain available:

```python
from addernet import (
    AdderNetLayer,
    AdderNetHDC,
    AdderAttention,
    AdderBoost,
    AdderCluster,
    AdderNetAdditiveRegressor,
)
```

### Recommendations

Replace iterative training with direct fitting when appropriate:

```python
# Before
layer.train(x, y, epochs_raw=1000, epochs_expanded=4000)

# Now
layer.fit(x, y)
```

For multiple inputs with real interactions:

```python
from addernet import AdderNetMultiInputLayer

model = AdderNetMultiInputLayer(
    bins=64,
    interactions="auto",
).fit(X, y)
```

For portable persistence of the scalar layer:

```python
layer.save_portable("model.npz")
```

Use `close()` or a context manager in long-running services.

---

## Production best practices

1. Define the feature schema and never change the order silently.
2. Validate units, scale, and domain before inference.
3. Save the code version, quantizer, and model together.
4. Compare training, validation, and test results, not only training error.
5. Measure latency using the real batch size.
6. Configure a fallback for inputs outside the domain.
7. Monitor drift and consider `partial_fit()` for scalar LUTs.
8. Use `min_samples_per_cell` to avoid excessively fragile cells.
9. Do not choose `bins` based only on the best training accuracy.
10. Perform a serialization round trip in the CI pipeline.

### Domain-monitoring example

```python
minimum = model.quantizer.minimum_
maximum = model.quantizer.maximum_

outside = np.any((X_live < minimum) | (X_live > maximum), axis=1)
outside_rate = outside.mean()

if outside_rate > 0.05:
    print("Warning: distribution outside the training domain")
```

---

## Project ideas

AdderNet can be used to build:

1. a nonlinear sensor calibrator;
2. a multi-output thermostat;
3. a battery-range estimator;
4. a network-anomaly detector;
5. a predictive-maintenance classifier;
6. a robot speed and direction controller;
7. a shipping price and delivery-time estimator;
8. a surrogate for an expensive physical simulation;
9. an adaptive model for sensor drift;
10. an explainable score with per-feature decomposition.

---

## Suggested roadmap

- native kernels for the multivariate model;
- multivariate `partial_fit()`;
- configurable regularization and smoothing;
- explicit missing-value handling;
- interaction selection with cross-validation;
- sparse or factorized tables;
- integrated classifier calibration;
- an API compatible with scikit-learn estimators;
- compact export to C and microcontrollers;
- multiplatform wheels in CI;
- in-depth CUDA and HDC profiling.

---

## Contributing

Contributions are welcome.

### Set up the environment

```bash
git clone https://github.com/PedroHenriqueBatistaSilva/AdderNet.git
cd AdderNet
python -m pip install -e ".[dev]"
addernet-build
python -m pytest
```

### Before opening a pull request

```bash
ruff check src tests
python -m pytest
addernet-selftest
```

Include:

- a description of the problem;
- before-and-after benchmarks when performance is affected;
- regression tests;
- the platform and compiler used;
- justification for format or API changes.

Avoid optimizations that improve only a synthetic benchmark while harming latency or portability in common use cases.

---

## Security and reliability

- Do not use example models for medical, financial, legal, or safety-critical decisions without expert validation.
- Validate model files and artifact provenance.
- Treat scores and probabilities as statistical outputs, not guarantees.
- Test with out-of-distribution data.
- Monitor drift, bin saturation, and clipping rates.

---

## FAQ

### Does AdderNet use no multiplications?

The scalar core and the already-quantized multivariate path are centered on LUTs, indices, masks, integer operations, and sums. Quantization of floating-point inputs and some auxiliary paths still use conventional arithmetic.

### Is it faster than a neural network?

It can be for low-latency CPU workloads and small models. On GPUs and huge batches, compiled dense networks may achieve higher throughput. Benchmark the full pipeline end to end.

### Can it learn any function?

With sufficient resolution and coverage, LUTs can approximate many functions within the observed domain. However, memory, sparsity, and high-order interactions limit scalability.

### Does `AdderNetMultiInputLayer` accept two separate inputs?

Yes:

```python
model.predict(x, y)
```

It also accepts N arguments, arrays with broadcasting, or a `(samples, features)` matrix.

### Can I have multiple outputs?

Yes. Pass `y` with shape `(samples, outputs)`.

### Can I use text labels?

Yes, with `AdderNetClassifier` and `AdderCluster`.

### How do I obtain explanations?

```python
explanation = model.explain(X)
```

The additive and interaction contributions sum exactly to the prediction.

### How do I update a model with new data?

For the scalar layer:

```python
layer.partial_fit(new_x, new_y, blend=0.1)
```

The multivariate version does not yet have a native `partial_fit()` method.

### Which serialization method should I use?

For `AdderNetLayer`, prefer `save_portable()` when portability matters. For the multivariate model, use `save()` with `.npz`.

### Does AdderNet extrapolate?

Not in a general sense. Quantization tends to clip inputs to the known boundaries.

---

## License

Distributed under the [Apache 2.0](LICENSE) license.

---

<div align="center">

**AdderNet 1.6.0.dev0**

Small models, predictable inference, and contributions you can inspect.

[Back to top](#addernet)

</div>
