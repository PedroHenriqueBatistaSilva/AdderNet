#!/usr/bin/env python3
"""
Example 3: Performance benchmark — AdderNet (C) vs Python
============================================================
Compares:
  1. Pure Python AdderNet (dict-based)
  2. C AdderNet via ctypes (table lookup)
  3. Standard NN: c * w + b (for reference)
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet import AdderNetLayer

# ---- Python AdderNet (reference) ----

class PythonAdderNet:
    def __init__(self, lr=0.1):
        self.w = {}
        self.lr = lr

    def predict(self, x):
        if x in self.w:
            return x + self.w[x]
        if not self.w:
            return x
        best = min(self.w.keys(), key=lambda k: abs(k - x))
        return x + self.w[best]

    def train(self, data, epochs):
        for _ in range(epochs):
            random.shuffle(data)
            for inp, tgt in data:
                if inp not in self.w:
                    self.w[inp] = 0.0
                err = self.predict(inp) - tgt
                self.w[inp] += self.lr
                err_up = self.predict(inp) - tgt
                self.w[inp] -= 2 * self.lr
                err_dn = self.predict(inp) - tgt
                self.w[inp] += self.lr
                if abs(err_up) < abs(err):
                    self.w[inp] += self.lr
                elif abs(err_dn) < abs(err):
                    self.w[inp] -= self.lr

# ---- Setup ----

training_data = [
    (0, 32), (5, 41), (10, 50), (15, 59), (20, 68),
    (25, 77), (30, 86), (35, 95), (40, 104), (50, 122),
    (60, 140), (70, 158), (80, 176), (90, 194), (100, 212),
    (-10, 14), (-20, -4), (37, 98.6),
]

N_INF = 1_000_000

print("=" * 60)
print("  AdderNet Benchmark")
print(f"  {N_INF:,} predictions")
print("=" * 60)

# ---- Train C AdderNet ----
print("\n  Training C AdderNet...")
t0 = time.perf_counter()
c_layer = AdderNetLayer(size=256, bias=50, input_min=-50, input_max=200, lr=0.1)
inputs  = [c for c, _ in training_data]
targets = [f for _, f in training_data]
c_layer.train(inputs, targets, epochs_raw=1000, epochs_expanded=4000)
t_c_train = time.perf_counter() - t0
print(f"  Done: {t_c_train:.3f}s")

# ---- Train Python AdderNet ----
print("  Training Python AdderNet...")
random.seed(42)
t0 = time.perf_counter()
py_net = PythonAdderNet(lr=0.1)
py_net.train(training_data, epochs=1000)
# expand data for python too
sorted_data = sorted(training_data, key=lambda x: x[0])
expanded = []
for v in range(-50, 201):
    fv = float(v)
    if fv <= sorted_data[0][0]:
        sl = (sorted_data[1][1] - sorted_data[0][1]) / (sorted_data[1][0] - sorted_data[0][0])
        expanded.append((fv, sorted_data[0][1] + sl * (fv - sorted_data[0][0])))
    elif fv >= sorted_data[-1][0]:
        sr = (sorted_data[-1][1] - sorted_data[-2][1]) / (sorted_data[-1][0] - sorted_data[-2][0])
        expanded.append((fv, sorted_data[-1][1] + sr * (fv - sorted_data[-1][0])))
    else:
        for i in range(len(sorted_data) - 1):
            if sorted_data[i][0] <= fv <= sorted_data[i+1][0]:
                d = sorted_data[i+1][0] - sorted_data[i][0]
                f = (fv - sorted_data[i][0]) / d if d else 0
                expanded.append((fv, sorted_data[i][1] + f * (sorted_data[i+1][1] - sorted_data[i][1])))
                break
py_net.train(expanded, epochs=4000)
t_py_train = time.perf_counter() - t0
print(f"  Done: {t_py_train:.3f}s")

# ---- Generate random inputs ----
random.seed(99)
bench_inputs = [random.uniform(-50, 200) for _ in range(N_INF)]

# ---- Benchmark: C AdderNet (single) ----
t0 = time.perf_counter()
for x in bench_inputs:
    c_layer.predict(x)
t_c_single = time.perf_counter() - t0

# ---- Benchmark: C AdderNet (batch) ----
import numpy as np
bench_np = np.array(bench_inputs, dtype=np.float64)
t0 = time.perf_counter()
c_layer.predict_batch(bench_np)
t_c_batch = time.perf_counter() - t0

# ---- Benchmark: Python AdderNet ----
t0 = time.perf_counter()
for x in bench_inputs:
    py_net.predict(x)
t_py = time.perf_counter() - t0

# ---- Standard NN (pure Python) ----
w, b = 1.8, 32.0
t0 = time.perf_counter()
for x in bench_inputs:
    _ = x * w + b
t_snn = time.perf_counter() - t0

# ---- Results ----
print(f"\n{'='*60}")
print(f"  PERFORMANCE ({N_INF:,} predictions)")
print(f"{'='*60}")
print(f"\n  {'Method':<30} {'Time':>10} {'Pred/s':>12} {'Speedup':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*10}")
print(f"  {'StdNN (x*w+b, Python)':<30} {t_snn:>9.3f}s {N_INF/t_snn:>12,.0f} {'1.00x':>10}")
print(f"  {'Python AdderNet (dict)':<30} {t_py:>9.3f}s {N_INF/t_py:>12,.0f} {t_snn/t_py:>9.2f}x")
print(f"  {'C AdderNet (single)':<30} {t_c_single:>9.3f}s {N_INF/t_c_single:>12,.0f} {t_snn/t_c_single:>9.2f}x")
print(f"  {'C AdderNet (batch np)':<30} {t_c_batch:>9.3f}s {N_INF/t_c_batch:>12,.0f} {t_snn/t_c_batch:>9.2f}x")

# ---- Precision check ----
print(f"\n{'='*60}")
print(f"  PRECISION")
print(f"{'='*60}")
print(f"\n  {'C':>6} | {'Real':>8} | {'C Adder':>10} | {'Py Adder':>10}")
print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*10}")
for c in [-40, 0, 25, 37, 100, 200]:
    real = c * 1.8 + 32
    cp = c_layer.predict(c)
    pp = py_net.predict(c)
    print(f"  {c:>6} | {real:>8.2f} | {cp:>10.2f} | {pp:>10.2f}")
