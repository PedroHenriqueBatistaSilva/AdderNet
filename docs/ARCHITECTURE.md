# Architecture

## Compatibility surface

The original top-level modules remain available so existing imports continue to
work: scalar layer, HDC, attention, boost, cluster, vector regressor, CUDA
detection, build command, and self-test.

## Organized extensions

```text
src/addernet/
├── addernet.py                 # safe ctypes scalar API
├── models/
│   ├── multi_input.py         # N-input/N-output additive + pairwise LUTs
│   └── classification.py      # arbitrary-label classifier wrapper
├── preprocessing/
│   └── quantization.py        # shape-safe uniform quantizer
├── reference.py               # readable NumPy scalar implementation
└── src/
    ├── addernet.c/.h          # hardened scalar native core
    └── ...                    # existing HDC and CUDA sources
```

## Inference paths

### Scalar

1. Convert input to integer.
2. Add bias.
3. Apply a power-of-two mask.
4. Read one LUT cell.

### Multi-input

1. Quantize each continuous feature, or accept already quantized integers.
2. Add one LUT value per feature.
3. For selected pairs, combine indices using a left shift and bitwise OR.
4. Add the corresponding pair-LUT values.

`predict_quantized` avoids continuous quantization and uses LUT reads, integer
bit operations, and additions in the model core.

## Memory

The additive portion scales as `outputs × features × bins`. Pairwise tables
scale as `outputs × selected_pairs × bins²`, so `max_interactions` is an
intentional safety limit.
