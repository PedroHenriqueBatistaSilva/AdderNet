# Changelog

## 1.6.0.dev0

### Fixed
- Removed the 256-sample truncation in native interpolation.
- Merged duplicate integer coordinates before interpolation.
- Validated finite values, integer conversion range, epochs, blend, and native return codes.
- Hardened binary loading against invalid sizes, ranges, learning rates, short files, and NaNs.
- Checked all native writes and made Python-side saves atomic.
- Added safe lifecycle management to avoid use-after-close and shutdown-time destructor errors.
- Disabled OpenMP by default after it caused severe overhead in the memory-bound lookup loop.

### Added
- Direct and blended scalar LUT fitting.
- Mutable LUT setter needed for portable model loading.
- Portable versioned scalar serialization.
- Multi-input/multi-output additive and pairwise interaction model.
- Multi-class classifier wrapper.
- Model scoring, explanations, memory reporting, and quantized inference.
- Quantizer inverse transform and metadata serialization.
- Automated tests and performance benchmark.
