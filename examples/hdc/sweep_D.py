#!/usr/bin/env python3
"""
Sweep D — Varredura de dimensão do hipervector
================================================
Compila libaddernet_hdc para cada D e mede acurácia vs throughput.
"""

import os
import sys
import subprocess
import time
import ctypes
import numpy as np
from sklearn.datasets import fetch_openml

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build")
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

D_VALUES = [10000, 5000, 2000, 1000, 500]

# ---- Compilar libs para cada D ----
print("Compilando libs...")
for D in D_VALUES:
    words = (D + 63) // 64
    so_name = os.path.join(LIB_DIR, f"libaddernet_hdc_{D}.so")
    if os.path.exists(so_name):
        print(f"  D={D}: já existe")
        continue
    cmd = [
        "gcc", "-O3", "-march=native", "-shared", "-fPIC",
        f"-DHDC_DIM={D}", f"-DHDC_WORDS={words}",
        "-o", so_name,
        os.path.join(SRC_DIR, "hdc_core.c"),
        os.path.join(SRC_DIR, "addernet_hdc.c"),
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=SRC_DIR)
    if r.returncode != 0:
        print(f"  D={D}: FALHOU — {r.stderr[:200]}")
    else:
        print(f"  D={D}: OK")

# ---- Carregar MNIST ----
print("\nCarregando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(np.float64) / 255.0
y_all = mnist.target.astype(np.int32)
X_train_raw, X_test_raw = X_all[:60000], X_all[60000:]
y_train, y_test = y_all[:60000], y_all[60000:]

# ---- Blocos 2x2 ----
def blocos_2x2(imagem):
    img = imagem.reshape(28, 28)
    resultado = []
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            resultado.append(img[i:i+2, j:j+2].mean())
    return np.array(resultado)

print("Pré-processando (blocos 2x2)...")
X_train_blocos = np.array([blocos_2x2(x) for x in X_train_raw])
X_test_blocos  = np.array([blocos_2x2(x) for x in X_test_raw])
X_train_int = (X_train_blocos * 255.0).astype(np.float64)
X_test_int  = (X_test_blocos * 255.0).astype(np.float64)

# ---- Wrapper genérico para qualquer D ----
class HDCModel:
    def __init__(self, D, lib_path):
        self.D = D
        self.words = (D + 63) // 64
        self.lib = ctypes.CDLL(lib_path)
        self._hv_t = ctypes.c_uint64 * self.words

        # Bind function signatures
        self.lib.hv_seed.argtypes = [ctypes.c_uint]
        self.lib.hv_seed.restype = None

        self.lib.an_hdc_create.restype = ctypes.c_void_p
        self.lib.an_hdc_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]

        self.lib.an_hdc_free.restype = None
        self.lib.an_hdc_free.argtypes = [ctypes.c_void_p]

        self.lib.an_hdc_train.restype = None
        self.lib.an_hdc_train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]

        self.lib.an_hdc_predict_batch.restype = ctypes.c_int
        self.lib.an_hdc_predict_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]

        self.lib.hv_hamming.restype = ctypes.c_int
        self.lib.hv_hamming.argtypes = [self._hv_t, self._hv_t]

        self.lib.hv_similarity.restype = ctypes.c_float
        self.lib.hv_similarity.argtypes = [self._hv_t, self._hv_t]

        self.lib.hv_bundle.restype = None
        self.lib.hv_bundle.argtypes = [self._hv_t, ctypes.POINTER(self._hv_t), ctypes.c_int]

    def create(self, n_vars, n_classes, table_size, seed=42):
        self.lib.hv_seed(seed)
        self._ptr = self.lib.an_hdc_create(n_vars, n_classes, table_size, None)
        if not self._ptr:
            raise MemoryError("create failed")
        self._n_vars = n_vars
        self._n_classes = n_classes

    def free(self):
        if self._ptr:
            self.lib.an_hdc_free(self._ptr)
            self._ptr = None

    def train(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)
        self.lib.an_hdc_train(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(X),
        )

    def predict_batch(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        n = len(X)
        out = np.empty(n, dtype=np.int32)
        self.lib.an_hdc_predict_batch(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
        )
        return out

    def model_memory_bytes(self):
        # position_hvs: n_vars * words * 8
        # codebook: n_classes * words * 8
        # bias: n_vars * 4
        # struct overhead: ~48 bytes
        return (self._n_vars * self.words * 8
                + self._n_classes * self.words * 8
                + self._n_vars * 4 + 48)

    def __del__(self):
        self.free()


# ---- Sweep ----
print(f"\n{'='*64}")
print(f"  D sweep — MNIST Blocos 2x2 (196 vars, 60k treino, 10k teste)")
print(f"{'='*64}")
print(f"\n{'D':>8} {'Acurácia':>10} {'Throughput':>12} {'Memória':>10} {'mulsd':>6}")
print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*6}")

results = []

for D in D_VALUES:
    so_path = os.path.join(LIB_DIR, f"libaddernet_hdc_{D}.so")
    if not os.path.exists(so_path):
        print(f"{D:>8} {'COMPILAÇÃO FALHOU':>10}")
        continue

    # Verify mulsd
    r = subprocess.run(
        ["objdump", "-d", so_path],
        capture_output=True, text=True
    )
    mulsd_count = r.stdout.count("mulsd") + r.stdout.count("vmulsd")

    model = HDCModel(D, so_path)
    model.create(196, 10, 256, seed=42)

    # Train
    t0 = time.perf_counter()
    model.train(X_train_int, y_train)
    t_train = time.perf_counter() - t0

    # Predict
    t0 = time.perf_counter()
    y_pred = model.predict_batch(X_test_int)
    t_pred = time.perf_counter() - t0

    acc = np.mean(y_pred == y_test) * 100
    throughput = int(len(X_test_int) / t_pred)
    mem_kb = model.model_memory_bytes() / 1024

    results.append({
        'D': D, 'acc': acc, 'throughput': throughput,
        'mem_kb': mem_kb, 'mulsd': mulsd_count
    })

    print(f"{D:>8} {acc:>9.1f}% {throughput:>11,}/s {mem_kb:>8.0f}KB {mulsd_count:>6}")

    model.free()

# ---- Análise ----
print(f"\n{'='*64}")

if results:
    # Ponto de colapso
    collapse = None
    for r in results:
        if r['acc'] < 70:
            collapse = r['D']
            break

    # Melhor custo-benefício: maior throughput com acc >= 75%
    viable = [r for r in results if r['acc'] >= 75]
    best_value = max(viable, key=lambda r: r['throughput']) if viable else None

    if collapse:
        print(f"  Ponto de colapso: D={collapse} (acurácia < 70%)")
    else:
        print(f"  Ponto de colapso: não encontrado (todos ≥ 70%)")

    if best_value:
        print(f"  Melhor custo-benefício: D={best_value['D']} ({best_value['acc']:.1f}%, {best_value['throughput']:,}/s)")
