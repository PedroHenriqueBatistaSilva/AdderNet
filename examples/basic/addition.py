#!/usr/bin/env python3
"""
Example 2: Learning addition (x + y = z)
==========================================
Demonstrates AdderNet learning f(x) = x + 5
for a single-variable case, then x + y for two variables.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))

from addernet import AdderNetLayer
import numpy as np

print("=" * 60)
print("  AdderNet: Learning Simple Addition")
print("=" * 60)

# ---- Part 1: f(x) = x + 5 ----
print("\n  Part 1: f(x) = x + 5")
print("  " + "-" * 40)

inputs  = list(range(0, 51))
targets = [x + 5 for x in inputs]

layer1 = AdderNetLayer(size=256, bias=5, input_min=0, input_max=50, lr=0.1)
layer1.train(inputs, targets, epochs_raw=500, epochs_expanded=2000)

print(f"\n  {'x':>6} | {'Real':>8} | {'Pred':>8} | {'Err':>8}")
print("  " + "-" * 36)
for x in [0, 7, 15, 25, 33, 42, 50]:
    real = x + 5
    pred = layer1.predict(x)
    print(f"  {x:>6} | {real:>8.1f} | {pred:>8.2f} | {pred - real:>+8.2f}")

# ---- Part 2: f(x) = 3*x + 7 ----
print("\n\n  Part 2: f(x) = 3*x + 7")
print("  " + "-" * 40)

inputs2  = list(range(0, 51))
targets2 = [3 * x + 7 for x in inputs2]

layer2 = AdderNetLayer(size=256, bias=7, input_min=0, input_max=50, lr=0.1)
layer2.train(inputs2, targets2, epochs_raw=500, epochs_expanded=2000)

print(f"\n  {'x':>6} | {'Real':>8} | {'Pred':>8} | {'Err':>8}")
print("  " + "-" * 36)
for x in [0, 5, 10, 15, 20, 30, 40, 50]:
    real = 3 * x + 7
    pred = layer2.predict(x)
    print(f"  {x:>6} | {real:>8.1f} | {pred:>8.2f} | {pred - real:>+8.2f}")

# ---- Part 3: Batch prediction with numpy ----
print("\n\n  Part 3: Batch prediction (numpy)")
print("  " + "-" * 40)

batch = np.array([0, 10, 20, 30, 40, 50], dtype=np.float64)
results = layer2.predict_batch(batch)
for x, y in zip(batch, results):
    print(f"  f({x:.0f}) = {y:.2f}  (real: {3*x+7:.1f})")
