#!/usr/bin/env python3
"""
Benchmark — AdderNet-HDC Otimizações
====================================
"""

import sys
import os
import time
import platform
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC

def detect_backend():
    import ctypes
    lib = ctypes.CDLL("build/libaddernet_hdc.so")
    lib.hdc_detect_backend.restype = ctypes.c_int
    backends = ["SCALAR", "AVX2", "NEON"]
    return backends[lib.hdc_detect_backend()]

def count_mulsd():
    import subprocess
    result = subprocess.run(
        ["objdump", "-d", "build/libaddernet_hdc.so"],
        capture_output=True, text=True
    )
    return result.stdout.count("mulsd")

def run_single_test(name, hdc, X_warmup, X_bench, n_threads):
    for _ in range(5):
        hdc.predict_batch(X_warmup[:100])
    
    t0 = time.perf_counter()
    hdc.predict_batch_mt(X_bench, 4)
    t_elapsed = time.perf_counter() - t0
    return t_elapsed

def run_benchmark():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    N_BENCH = 30000
    X_bench = np.tile(X_test, (N_BENCH // len(X_test) + 1, 1))[:N_BENCH]
    X_warmup = X_test

    backend = detect_backend()
    arch = platform.machine()
    mulsd = count_mulsd()

    print("=" * 80)
    print("Benchmark AdderNet-HDC — Otimizações (Prompt 1-3)")
    print("=" * 80)
    print(f"Plataforma: {arch}")
    print(f"Backend detectado: {backend}")
    print(f"mulsd: {mulsd} {'OK' if mulsd == 0 else 'FAIL'}")
    print("=" * 80)

    results = []

    configs = [
        ("Baseline (no cache)", False, False, 0, 0, 0),
        ("+ cache", True, False, 0, 0, 0),
        ("+ circulant", False, True, 0, 0, 0),
        ("+ early term margin=50", False, False, 50, 0, 0),
        ("+ early term margin=20", False, False, 20, 0, 0),
        ("+ fold=2", False, False, 0, 2, 0),
        ("+ fold=4", False, False, 0, 4, 0),
        ("+ fold=8", False, False, 0, 8, 0),
        ("+ circulant+early term", False, True, 50, 0, 0),
    ]

    baseline_time = None

    for name, use_cache, use_circulant, early_margin, fold, combined in configs:
        hdc = AdderNetHDC(n_vars=4, n_classes=3, table_size=256, seed=42)
        
        if use_circulant:
            hdc.set_circulant(True)
        hdc.train(X_train, y_train)
        
        if use_cache:
            hdc.warm_cache()
        
        if fold > 0:
            hdc.fold_codebook(fold)
        
        if early_margin > 0:
            hdc.set_early_termination(True, early_margin)
        
        hdc.set_threads(4)

        acc = hdc.accuracy(X_test, y_test) * 100
        
        times = []
        for _ in range(3):
            t = run_single_test(name, hdc, X_warmup, X_bench, 8)
            times.append(t)
        
        t_elapsed = min(times)
        throughput = N_BENCH / t_elapsed

        if baseline_time is None:
            baseline_time = t_elapsed

        speedup = baseline_time / t_elapsed
        results.append((name, throughput, acc, speedup))
        print(f"{name:<28} {throughput:>8,.0f} pred/s  {acc:>5.1f}%  {speedup:>5.2f}x")

    print("=" * 80)
    print(f"{'Config':<28} {'pred/s':>10} {'acc':>7} {'vs baseline':>12}")
    print("-" * 80)
    for name, throughput, acc, speedup in results:
        print(f"{name:<28} {throughput:>10,.0f}  {acc:>5.1f}%  {speedup:>10.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
