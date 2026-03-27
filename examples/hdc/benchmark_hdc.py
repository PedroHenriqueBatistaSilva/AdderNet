#!/usr/bin/env python3
"""
Example: Benchmark — AdderNet-HDC vs MLP sklearn
===================================================
Compares inference speed, memory usage, and accuracy on Iris.
"""

import sys
import os
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC

# ---- Load data ----
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

N_INF = 100_000
X_bench = np.tile(X_test, (N_INF // len(X_test) + 1, 1))[:N_INF]

print("=" * 60)
print("  Benchmark: AdderNet-HDC vs MLP sklearn (Iris)")
print(f"  {N_INF:,} predictions for speed test")
print("=" * 60)

# ---- Train AdderNet-HDC ----
print(f"\n  [1/2] AdderNet-HDC")
t0 = time.perf_counter()
hdc = AdderNetHDC(n_vars=4, n_classes=3, table_size=256, seed=42)
hdc.train(X_train, y_train)
t_hdc_train = time.perf_counter() - t0
print(f"  Train: {t_hdc_train*1000:.1f}ms")

# ---- Train MLP ----
print(f"\n  [2/2] MLP sklearn (1 hidden layer, 10 neurons)")
t0 = time.perf_counter()
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
t_mlp_train = time.perf_counter() - t0
print(f"  Train: {t_mlp_train*1000:.1f}ms")

# ---- Accuracy ----
hdc_acc = accuracy_score(y_test, hdc.predict_batch(X_test)) * 100
mlp_acc = accuracy_score(y_test, mlp.predict(X_test)) * 100

# ---- Speed: AdderNet-HDC (batch) ----
t0 = time.perf_counter()
hdc.predict_batch(X_bench)
t_hdc = time.perf_counter() - t0

# ---- Speed: AdderNet-HDC (single) ----
t0 = time.perf_counter()
for i in range(min(N_INF, 10000)):
    hdc.predict(X_bench[i])
n_single = min(N_INF, 10000)
t_hdc_single = time.perf_counter() - t0

# ---- Speed: MLP ----
t0 = time.perf_counter()
mlp.predict(X_bench)
t_mlp = time.perf_counter() - t0

# ---- Memory estimate ----
hdc_bytes = 4 * 256 * (10000 // 8) + 3 * (10000 // 8)  # enc_table + codebook
mlp_params = sum(p.size for p in mlp.coefs_) + sum(p.size for p in mlp.intercepts_)
mlp_bytes = mlp_params * 8  # float64

# ---- Results ----
print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")

print(f"\n  {'Metric':<30} {'AdderNet-HDC':>15} {'MLP sklearn':>15}")
print(f"  {'-'*30} {'-'*15} {'-'*15}")
print(f"  {'Accuracy':<30} {f'{hdc_acc:.1f}%':>15} {f'{mlp_acc:.1f}%':>15}")
print(f"  {'Train time':<30} {f'{t_hdc_train*1000:.1f}ms':>15} {f'{t_mlp_train*1000:.1f}ms':>15}")
print(f"  {'Inference (batch, 100K)':<30} {f'{t_hdc*1000:.1f}ms':>15} {f'{t_mlp*1000:.1f}ms':>15}")
print(f"  {'Throughput (batch)':<30} {f'{N_INF/t_hdc:,.0f}/s':>15} {f'{N_INF/t_mlp:,.0f}/s':>15}")
print(f"  {'Inference (single, 10K)':<30} {f'{t_hdc_single*1000:.1f}ms':>15} {'-':>15}")
print(f"  {'Throughput (single)':<30} {f'{n_single/t_hdc_single:,.0f}/s':>15} {'-':>15}")
print(f"  {'Model memory (est.)':<30} {f'{hdc_bytes/1024:.1f}KB':>15} {f'{mlp_bytes/1024:.1f}KB':>15}")
print(f"  {'Parameters':<30} {'~4×256×157':>15} {f'{mlp_params}':>15}")
print(f"  {'Float mul at inference':<30} {'0 (ZERO)':>15} {'yes':>15}")

speedup = t_mlp / t_hdc if t_hdc > 0 else 0
print(f"\n  Batch speedup: {speedup:.1f}x {'in favor of AdderNet-HDC' if speedup > 1 else 'in favor of MLP'}")
