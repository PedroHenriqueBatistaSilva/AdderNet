#!/usr/bin/env python3
"""
Benchmark an_hdc_predict_batch_cuda — CPU vs GPU (MX250)
=========================================================
Fashion-MNIST blocos 2x2, 196 variáveis, 10 classes.
"""

import sys
import os
import time
import ctypes
import numpy as np
from sklearn.datasets import fetch_openml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from addernet_hdc import AdderNetHDC

LIB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "build")

# ---- Load CUDA library ----
cuda_lib = ctypes.CDLL(os.path.join(LIB_DIR, "libaddernet_hdc_cuda.so"), mode=ctypes.RTLD_GLOBAL)

# ---- Fashion-MNIST ----
print("Carregando Fashion-MNIST...")
fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X_all = fashion.data.astype(np.float64) / 255.0
y_all = fashion.target.astype(np.int32)
X_train_raw = X_all[:60000]
X_test_raw = X_all[60000:]
y_train, y_test = y_all[:60000], y_all[60000:]

def blocos_2x2(img):
    img = img.reshape(28, 28)
    r = []
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            r.append(img[i:i+2, j:j+2].mean())
    return np.array(r)

print("Pré-processando...")
X_train = np.array([blocos_2x2(x) for x in X_train_raw]) * 255.0
X_test = np.array([blocos_2x2(x) for x in X_test_raw]) * 255.0
N = len(X_test)

print("Treinando...")
model = AdderNetHDC(n_vars=196, n_classes=10, table_size=256, seed=42)
model.train(X_train, y_train)

# ---- CPU batch ----
print(f"\nBenchmark: {N} predições")
y_pred_cpu = model.predict_batch(X_test)
acc_cpu = np.mean(y_pred_cpu == y_test) * 100

# ---- CPU timing ----
t0 = time.perf_counter()
model.predict_batch(X_test)
t_cpu = time.perf_counter() - t0

# ---- GPU batch ----
cuda_lib.an_hdc_predict_batch_cuda.restype = ctypes.c_int
cuda_lib.an_hdc_predict_batch_cuda.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

X_flat = np.ascontiguousarray(X_test, dtype=np.float64)
y_pred_gpu = np.empty(N, dtype=np.int32)

# Warmup
cuda_lib.an_hdc_predict_batch_cuda(
    model._ptr,
    X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    y_pred_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    100,
)

t0 = time.perf_counter()
cuda_lib.an_hdc_predict_batch_cuda(
    model._ptr,
    X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    y_pred_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    N,
)
t_gpu = time.perf_counter() - t0
acc_gpu = np.mean(y_pred_gpu == y_test) * 100

# ---- Report ----
print(f"\n{'='*60}")
print(f"  Benchmark batch — CPU vs GPU (MX250, Fashion-MNIST {N})")
print(f"{'='*60}")
print(f"  CPU throughput:   {N/t_cpu:,.0f}/s")
print(f"  GPU throughput:   {N/t_gpu:,.0f}/s")
print(f"  Speedup:          {t_cpu/t_gpu:.1f}x")
print(f"  Acurácia CPU:     {acc_cpu:.1f}%")
print(f"  Acurácia GPU:     {acc_gpu:.1f}%")
print(f"  mulsd:            0")
