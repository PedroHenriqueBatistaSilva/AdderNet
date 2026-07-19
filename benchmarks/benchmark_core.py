"""Small reproducible benchmark; run with `PYTHONPATH=src python benchmarks/benchmark_core.py`."""
from __future__ import annotations

import json
import time
import numpy as np

from addernet import AdderNetLayer, AdderNetMultiInputLayer


def timed(fn, repeats=5):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return min(values)


def main():
    rng = np.random.default_rng(42)
    x = rng.integers(-50, 201, size=20_000).astype(np.float64)
    y = np.sin(x / 20.0) * 50.0 + 75.0

    iterative = AdderNetLayer()
    iterative_seconds = timed(lambda: iterative.train(x, y, epochs_raw=1000, epochs_expanded=4000), repeats=1)
    direct = AdderNetLayer()
    direct_seconds = timed(lambda: direct.fit(x, y), repeats=5)

    batch = rng.uniform(-50, 200, size=1_000_000)
    direct.predict_batch(batch)
    inference_seconds = timed(lambda: direct.predict_batch(batch))

    grid = rng.uniform(-1, 1, size=(50_000, 2))
    target = grid[:, 0] * grid[:, 1] + grid[:, 0] - grid[:, 1]
    multi = AdderNetMultiInputLayer(bins=64, max_interactions=1)
    multi_seconds = timed(lambda: multi.fit(grid, target), repeats=1)

    print(json.dumps({
        "scalar_iterative_fit_seconds": iterative_seconds,
        "scalar_direct_fit_seconds": direct_seconds,
        "direct_fit_speedup": iterative_seconds / direct_seconds,
        "scalar_inference_1m_seconds": inference_seconds,
        "scalar_inference_million_per_second": 1.0 / inference_seconds,
        "multi_input_50k_fit_seconds": multi_seconds,
        "multi_input_training_rmse": multi.training_rmse_,
    }, indent=2))


if __name__ == "__main__":
    main()
