# AdderNet — Developer Guide

Architecture, API reference, and internals for contributors. Current version: **1.4.5**.

## Changelog

### v1.4.5 — Controle de Verbose
- Adicionada função `set_verbose(bool)` para habilitar/desabilitar logs
- Variável de ambiente `ADDERNET_VERBOSE` para controle sem código
- Todos os prints de build e detecção CUDA agora respeitam a flag

## Architecture

```
Input (int)
    │
    ▼
┌──────────────────┐
│  idx = (x+bias)  │  ← 1 ADD + 1 AND (compile to LEA on x86)
│       & mask     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  offset[idx]     │  ← 1 memory load (cache-hot, 4-5 cycles)
└──────┬───────────┘
       │
       ▼
   Output (double)
```

**Why it's fast**: A standard neuron does `y = x*w + b` (mulsd ~4 cycles + addsd ~4 cycles). AdderNet precomputes the result into a table indexed by `input`. At inference, it's a single array load — no ALU arithmetic at all.

### Table layout

- Size must be power-of-2 (default 256) to enable `& mask` instead of `- min` subtraction
- The `offset` array stores the **final output** value, not `output - c`. Storing `F - C` and adding `C` at inference costs an extra `vcvtsi2sd` (int→double, ~4 cycles) that kills the speedup
- `alignas(64)` on the offset table for cache-line alignment

### Training algorithm

Trial-and-error with directional search:

```c
err  = tab[idx] - target;
tab[idx] += lr;   eu = tab[idx] - target;
tab[idx] -= 2*lr; ed = tab[idx] - target;
tab[idx] += lr;   // restore

if (|eu| < |err|) tab[idx] += lr;
else if (|ed| < |err|) tab[idx] -= lr;
```

No gradients, no backpropagation. Pure addition/subtraction.

### Data expansion (`expand_data`)

Raw training data (e.g. 18 Celsius→Fahrenheit points) isn't enough to fill the table. The `expand_data` function:

1. Sorts input points
2. For each integer `v` in `[input_min, input_max]`:
   - If `v < first_point`: extrapolate left using slope of first two points
   - If `v > last_point`: extrapolate right using slope of last two points
   - Otherwise: linear interpolation between nearest neighbors

This is essential. Without it, weights for untrained inputs stay at 0 and predictions are wrong outside the training range.

### Training phases

1. **Phase 1** (raw data, ~1000 epochs): Learns the core function from sparse samples
2. **Phase 2** (expanded data, ~4000 epochs): Fills in the gaps via interpolation

## Building

```bash
# Shared library (.so)
make

# Static library (.a)
make static

# Build + run C tests
make test

# Clean
make clean
```

Compiler flags: `-O3 -march=native -fPIC`. The `-march=native` is important — it lets the compiler emit `LEA` instead of `SUB` for the AND-mask index calculation.

## API Reference

### Types

```c
typedef struct {
    int    size;                    // table size (power of 2)
    int    bias;                    // added to input before masking
    int    input_min;               // for data expansion range
    int    input_max;               // for data expansion range
    double lr;                      // learning rate
    alignas(64) double offset[256]; // precomputed outputs
} an_layer;
```

### Functions

#### `an_layer_create(size, bias, input_min, input_max, lr) → an_layer*`

Allocates and zeroes a layer. `size` must be power-of-2 and ≤ `AN_TABLE_SIZE` (256).

```c
// Covers inputs [-50, 205] with table of 256 entries
an_layer *layer = an_layer_create(256, 50, -50, 200, 0.1);
```

#### `an_layer_free(layer)`

Frees a layer allocated by `an_layer_create`. Uses `aligned_free` internally.

#### `an_train(layer, inputs, targets, n_samples, epochs_raw, epochs_expanded) → int`

Two-phase training:
- `epochs_raw`: iterations over the original samples
- `epochs_expanded`: iterations over interpolated/extrapolated data

```c
double x[] = {0, 10, 25, 37, 50, 100};
double y[] = {32, 50, 77, 98.6, 122, 212};
an_train(layer, x, y, 6, 1000, 4000);
```

#### `an_predict(layer, input) → double`

Single prediction. Input is cast to `int` internally.

```c
double f = an_predict(layer, 37.0);  // 98.60
```

#### `an_predict_batch(layer, inputs, outputs, n) → int`

Batch prediction. Avoids per-call FFI overhead when called from Python.

```c
double in[] = {0, 25, 37, 100};
double out[4];
an_predict_batch(layer, in, out, 4);
```

#### `an_save(layer, path) → int`

Binary format (no versioning, no checksums):

```
[int size][int bias][int input_min][int input_max][double lr]
[double offset[0]] ... [double offset[size-1]]
```

#### `an_load(path) → an_layer*`

Reads a file written by `an_save`. Returns `NULL` on failure.

#### `an_get_offset(layer, buf, buf_size) → int`

Copies the offset table into a caller-provided buffer. Used by Python bindings to avoid matching the `alignas(64)` struct layout.

#### Metadata accessors

```c
int    an_get_size(layer);
int    an_get_bias(layer);
int    an_get_input_min(layer);
int    an_get_input_max(layer);
double an_get_lr(layer);
```

These exist so FFI callers (Python ctypes, Lua, etc.) don't need to match the C struct layout.

## Python Bindings

`addernet.py` wraps the C library via `ctypes`. The pointer is treated as opaque (`c_void_p`) — no struct layout matching needed.

### Key design decisions

- **Opaque pointer**: Avoids the `alignas(64)` padding problem in ctypes
- **numpy interop**: `predict_batch()` accepts/returns `np.float64` arrays directly via `ctypeslib.ndpointer`
- **Accessors**: Metadata (`size`, `bias`, etc.) goes through C accessor functions, not direct struct field access

### Class: `AdderNetLayer`

```python
from addernet import AdderNetLayer

layer = AdderNetLayer(size=256, bias=50, input_min=-50, input_max=200, lr=0.1)
layer.train(inputs, targets, epochs_raw=1000, epochs_expanded=4000)
layer.predict(37.0)                          # → float
layer.predict_batch(np.array([0, 37, 100]))  # → np.ndarray
layer.save("model.bin")
loaded = AdderNetLayer.load("model.bin")
layer.offset_table                           # → np.ndarray of table values
```

## AdderNet-HDC internals

### Encoding

Each input variable `v` with value `x` is encoded as:
```c
int bin = ((int)x + bias[v]) & table_mask;
hv_t val_hv = hv_from_seed(v * 100003 + bin);  // deterministic, zero storage
hv_t pair = bind(position_hvs[v], val_hv);       // XOR
```

All pairs are bundled (majority vote) into a single query hypervector.

### OnlineHD (v1.0.2)

Prevents model saturation by weighting each sample by novelty:

```c
float sim = hv_similarity(sample, class_mean);
float weight = 1.0f - sim;  // novel = 1, redundant ≈ 0
// Samples with weight < 0.05 are skipped entirely
```

This uses weighted bit-count majority vote instead of naive `hv_bundle()`.

### AdaptHD retraining (v1.0.2)

`an_hdc_retrain()` implements iterative error correction:

```c
for each iteration:
    for each sample (x, y_true):
        y_pred = an_hdc_predict(model, x)
        if y_pred != y_true:
            encoded = encode(model, x)
            for each bit in encoded:
                cb_counts[y_true][bit] += lr;   // reinforce correct
                cb_counts[y_pred][bit] -= lr;   // penalize wrong
    rebuild codebook from cb_counts (threshold > 0)
```

Symmetric margin: set bits start at +margin, unset at -margin. Requires `margin/lr` corrections to flip a bit.

### Early-exit Hamming (v1.0.2)

```c
int hv_hamming_early_exit(const hv_t a, const hv_t b, int max_allowed) {
    int dist = 0;
    for each 4-word unrolled block:
        dist += popcount(a[w] ^ b[w]);
        if (dist > max_allowed) return dist;  // bail out early
    return dist;
}
```

Classes are compared sequentially. If a candidate's distance exceeds the current best, the comparison aborts immediately.

### Dimensionality (v1.0.2)

D reduced from 10000 to 2500 bits (40 uint64 words). This gives ~4x speedup on Hamming comparisons with minimal accuracy loss. Compile-time constant in `hdc_core.h`:

```c
#define HDC_DIM   2500
#define HDC_WORDS 40
```

### Performance characteristics

| Operation | Cost | Notes |
|---|---|---|
| `an_predict` | 1 memory load | ~4-5 cycles if cache-hot |
| `an_predict_batch` (1M items) | ~4ms | Limited by memory bandwidth |
| `an_train` (18 samples, 5k epochs) | ~3ms | Trial-and-error, no gradients |
| Python `predict()` (single via ctypes) | ~0.75μs | ctypes call overhead dominates |
| Python `predict_batch()` (1M via numpy) | ~4ms | Same as raw C, no overhead |

### Why not int32 tables?

Int32 (fixed-point) tables are 4 bytes vs 8 bytes for double, so they fit more entries per cache line. But training targets like 98.6 need decimal precision, and the int→double conversion at inference costs ~4 cycles. Double tables are simpler and the performance difference is negligible (< 3%).

## x86 specifics

- `& mask` compiles to `AND reg, 0xFF` — 1 cycle, can use LEA in some cases
- `mulsd` is heavily optimized on modern x86 (~4 cycles, pipelined), so AdderNet's advantage over StdNN is ~1.2x with unroll x8
- The theoretical limit is ~75% of CPU load-port capacity
- `alignas(64)` ensures the offset table doesn't straddle cache lines

## Extending to multi-layer

The current library is single-layer. For multi-layer networks:

- Layer N's output is `double`
- To index into Layer N+1, it must be cast to `int`
- This cast is unavoidable and costs ~4 cycles per layer transition
- Save/load format would need layer count prefix

## Known limitations

1. Input must be castable to `int` (fractional parts are truncated)
2. Table size is capped at 256 entries (`AN_TABLE_SIZE`)
3. Single-variable only
4. No batch training with shuffling (always sequential)
5. Save/load has no versioning or error recovery
