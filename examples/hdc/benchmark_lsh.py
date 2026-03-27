#!/usr/bin/env python3
"""
LSH Benchmark — Escala de classes
Testa o throughput com diferentes números de classes, com e sem LSH.
"""

import sys
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC

def create_synthetic_dataset(n_vars, n_classes, n_samples=1000):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_vars)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y

def run_benchmark():
    print("=" * 70)
    print("LSH Benchmark — Escala de classes")
    print("=" * 70)
    
    n_vars = 4
    n_samples = 200
    N_BENCH = 5000
    
    results = []
    
    for C in [10, 100, 1000, 10000]:
        print(f"\n--- C = {C} classes ---")
        
        X, y = create_synthetic_dataset(n_vars, C, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        X_bench = np.tile(X_test, (N_BENCH // len(X_test) + 1, 1))[:N_BENCH]
        
        hdc = AdderNetHDC(n_vars=n_vars, n_classes=C, table_size=256, seed=42)
        hdc.train(X_train, y_train)
        hdc.set_threads(1)
        
        for _ in range(3):
            hdc.predict_batch(X_test)
        
        t0 = time.perf_counter()
        y_pred = hdc.predict_batch(X_bench)
        t_no_lsh = time.perf_counter() - t0
        throughput_no_lsh = N_BENCH / t_no_lsh
        acc_no_lsh = np.mean(y_pred == np.array([y_test[i % len(y_test)] for i in range(N_BENCH)])) * 100
        
        hdc.build_lsh()
        
        for _ in range(3):
            hdc.predict_batch(X_test[:10])
        
        t0 = time.perf_counter()
        y_pred_lsh = hdc.predict_batch(X_bench)
        t_lsh = time.perf_counter() - t0
        throughput_lsh = N_BENCH / t_lsh
        acc_lsh = np.mean(y_pred_lsh == np.array([y_test[i % len(y_test)] for i in range(N_BENCH)])) * 100
        
        speedup = throughput_lsh / throughput_no_lsh if throughput_no_lsh > 0 else 0
        
        results.append((C, throughput_no_lsh, throughput_lsh, acc_no_lsh, acc_lsh))
        print(f"  Sem LSH: {throughput_no_lsh:>8,.0f} pred/s  (acc={acc_no_lsh:.1f}%)")
        print(f"  Com LSH: {throughput_lsh:>8,.0f} pred/s  (acc={acc_lsh:.1f}%)")
        print(f"  Speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print(f"{'C classes':<12} {'Sem LSH':>12} {'Com LSH':>12} {'Recall@1':>10} {'Speedup':>10}")
    print("-" * 70)
    for C, tp_no, tp_lsh, acc_no, acc_lsh in results:
        recall = acc_lsh / acc_no * 100 if acc_no > 0 else 0
        speedup = tp_lsh / tp_no if tp_no > 0 else 0
        print(f"{C:>12} {tp_no:>12,.0f} {tp_lsh:>12,.0f} {recall:>9.1f}% {speedup:>9.2f}x")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
