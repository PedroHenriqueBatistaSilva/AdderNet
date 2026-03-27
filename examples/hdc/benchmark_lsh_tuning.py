#!/usr/bin/env python3
"""
LSH Scaling Benchmark with Different C Classes
"""

import sys
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC

def create_dataset(n_vars, n_classes, n_samples=200):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_vars) * 2
    centers = np.random.randn(n_classes, n_vars) * 3
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        dists = np.sum((X[i] - centers) ** 2, axis=1)
        y[i] = np.argmin(dists)
    return X, y

def run_benchmark():
    print("=" * 70)
    print("LSH Scaling Benchmark")
    print("=" * 70)
    
    n_vars = 4
    n_samples = 200
    N_BENCH = 5000
    
    C_list = [10, 100, 1000, 10000]
    configs = [
        (16, 4),
        (12, 8),
        (10, 8),
        (8, 16),
    ]
    
    results = []
    
    for C in C_list:
        print(f"\n--- C = {C} ---")
        
        X, y = create_dataset(n_vars, C, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        n_repeat = (N_BENCH + len(X_test) - 1) // len(X_test)
        X_bench = np.tile(X_test, (n_repeat, 1))[:N_BENCH]
        y_bench = np.tile(y_test, n_repeat)[:N_BENCH]
        
        hdc = AdderNetHDC(n_vars=n_vars, n_classes=C, table_size=256, seed=42)
        hdc.train(X_train, y_train)
        hdc.set_threads(1)
        
        for _ in range(3):
            hdc.predict_batch(X_test)
        
        t0 = time.perf_counter()
        y_pred = hdc.predict_batch(X_bench)
        t_no_lsh = time.perf_counter() - t0
        tp_no_lsh = N_BENCH / t_no_lsh
        acc_no_lsh = np.mean(y_pred == y_bench) * 100
        
        print(f"  Sem LSH: {tp_no_lsh:>8,.0f} pred/s  acc={acc_no_lsh:.1f}%")
        
        for k, l in configs:
            hdc = AdderNetHDC(n_vars=n_vars, n_classes=C, table_size=256, seed=42)
            hdc.train(X_train, y_train)
            hdc.set_threads(1)
            
            hdc.build_lsh(k=k, l=l)
            
            for _ in range(3):
                hdc.predict_batch(X_test[:5])
            
            t0 = time.perf_counter()
            y_pred_lsh = hdc.predict_batch(X_bench)
            t_lsh = time.perf_counter() - t0
            tp_lsh = N_BENCH / t_lsh
            acc_lsh = np.mean(y_pred_lsh == y_bench) * 100
            
            speedup = tp_lsh / tp_no_lsh
            recall = acc_lsh / acc_no_lsh * 100 if acc_no_lsh > 0 else 0
            
            results.append((C, k, l, tp_no_lsh, tp_lsh, speedup, acc_no_lsh, acc_lsh, recall))
            print(f"  K={k},L={l}: {tp_lsh:>8,.0f} pred/s  acc={acc_lsh:.1f}%  speedup={speedup:.2f}x")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"{'C':>6} {'K':>3} {'L':>3} {'No LSH':>10} {'Com LSH':>10} {'Speedup':>10} {'Acc%':>8} {'Recall':>8}")
    print("-" * 80)
    for C, k, l, tp_no, tp_lsh, sp, acc_no, acc_lsh, rec in results:
        print(f"{C:>6} {k:>3} {l:>3} {tp_no:>10,.0f} {tp_lsh:>10,.0f} {sp:>10.2f}x {acc_no:>7.1f} {rec:>7.1f}%")

if __name__ == "__main__":
    run_benchmark()
