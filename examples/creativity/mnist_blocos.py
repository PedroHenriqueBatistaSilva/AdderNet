#!/usr/bin/env python3
"""
MNIST Blocos 4x4 — AdderNet-HDC
=================================
784 pixels → 49 blocos (média 4x4) → HDC.
Zero multiplicação de ponto flutuante na inferência.
"""

import sys
import os
import time
import numpy as np
from sklearn.datasets import fetch_openml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC, _lib, _HV_t

# ---- Função de blocos ----
def blocos_2x2(imagem):
    """784 pixels (28x28) → 196 valores (14x14 blocos de 2x2)."""
    img = imagem.reshape(28, 28)
    resultado = []
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            resultado.append(img[i:i+2, j:j+2].mean())
    return np.array(resultado)

# ---- Carregar MNIST ----
print("Carregando MNIST...")
t0 = time.perf_counter()

cache = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "mnist_cache_60k.npz")
if os.path.exists(cache):
    data = np.load(cache)
    X_train_raw, y_train = data["X_train"], data["y_train"]
    X_test_raw,  y_test  = data["X_test"],  data["y_test"]
else:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_all = mnist.data.astype(np.float64) / 255.0
    y_all = mnist.target.astype(np.int32)
    X_train_raw, X_test_raw = X_all[:60000], X_all[60000:]
    y_train, y_test = y_all[:60000], y_all[60000:]
    np.savez(cache, X_train=X_train_raw, y_train=y_train,
             X_test=X_test_raw, y_test=y_test)

t_load = time.perf_counter() - t0
print(f"  Carregado em {t_load:.1f}s")

# ---- Pré-processar: blocos 4x4 ----
print("Pré-processando (blocos 4x4)...")
t0 = time.perf_counter()
X_train_blocos = np.array([blocos_2x2(x) for x in X_train_raw])
X_test_blocos  = np.array([blocos_2x2(x) for x in X_test_raw])
t_pre = time.perf_counter() - t0
print(f"  {X_train_raw.shape[1]} pixels → {X_train_blocos.shape[1]} blocos em {t_pre:.1f}s")

# Escalar para [0, 255] (a tabela usa int(input) para indexar)
X_train_int = (X_train_blocos * 255.0).astype(np.float64)
X_test_int  = (X_test_blocos * 255.0).astype(np.float64)

# ---- Criar e treinar ----
N_VARS = 196
print(f"\nCriando modelo: {N_VARS} variáveis, 10 classes, table_size=256...")

t0 = time.perf_counter()
model = AdderNetHDC(n_vars=N_VARS, n_classes=10, table_size=256, seed=42)
t_create = time.perf_counter() - t0
print(f"  Criado em {t_create:.1f}s")

print(f"  Treinando em {len(X_train_int)} amostras...")
t0 = time.perf_counter()
model.train(X_train_int, y_train)
t_train = time.perf_counter() - t0
print(f"  Treinado em {t_train*1000:.0f}ms")

# ---- Avaliar ----
print(f"  Avaliando em {len(X_test_int)} amostras...")
t0 = time.perf_counter()
y_pred = model.predict_batch(X_test_int)
t_pred = time.perf_counter() - t0

accuracy = np.mean(y_pred == y_test) * 100

# ---- Relatório ----
print(f"\n{'='*60}")
print(f"  MNIST Blocos 2x2 — AdderNet-HDC (196 variáveis, 10 classes)")
print(f"{'='*60}")
print(f"  Amostras treino:              {len(X_train_int):,}")
print(f"  Amostras teste:               {len(X_test_int):,}")
print(f"  Tempo de treino:              {t_train*1000:.0f}ms")
print(f"  Tempo de inferência ({len(X_test_int):,}):   {t_pred*1000:.1f}ms")
print(f"  Throughput:                   {len(X_test_int)/t_pred:,.0f} pred/s")
print(f"  Acurácia:                     {accuracy:.1f}%")
print(f"  Multiplicações float:         0")
print(f"")
print(f"  vs. MNIST raw (784 variáveis): 10.4%")
print(f"  vs. blocos 4x4 (49 vars):      64.3%")

# Acurácia por dígito
print(f"\n  Acurácia por dígito:")
for d in range(10):
    mask = y_test == d
    if mask.sum() > 0:
        d_acc = np.mean(y_pred[mask] == d) * 100
        hit = int(np.sum(y_pred[mask] == d))
        tot = int(mask.sum())
        bar = "#" * int(d_acc / 3)
        print(f"    {d}: {d_acc:5.1f}%  {hit}/{tot}  {bar}")

# ---- Dígito inventado: 3 BUNDLE 8 ----
print(f"\n{'='*60}")
print(f"  Dígito inventado: 3 BUNDLE 8")
print(f"{'='*60}")

cb = model.codebook

def numpy_to_hv(arr):
    hv = _HV_t()
    for i in range(len(arr)):
        hv[i] = int(arr[i])
    return hv

cbs = [numpy_to_hv(c) for c in cb]

digito_novo = _HV_t()
vecs = (_HV_t * 2)()
vecs[0] = cbs[3]
vecs[1] = cbs[8]
_lib.hv_bundle(digito_novo, vecs, 2)

print(f"\n  Similaridades com todas as 10 classes:")
sims = []
for d in range(10):
    s = _lib.hv_similarity(digito_novo, cbs[d])
    sims.append(s)
    bar = "#" * int(s * 30)
    marker = " <---" if d in (3, 8) else ""
    print(f"    {d}: {s:.3f}  {bar}{marker}")

sorted_d = sorted(range(10), key=lambda i: sims[i], reverse=True)
print(f"\n  Mais similar a: {sorted_d[0]} ({sims[sorted_d[0]]:.3f})")
print(f"  Segundo mais:   {sorted_d[1]} ({sims[sorted_d[1]]:.3f})")

if sorted_d[0] in (3, 8) and sorted_d[1] in (3, 8):
    print(f"  Resultado: dígito inventado ficou entre 3 e 8 ✓")
else:
    print(f"  Resultado: top-2 = {sorted_d[:2]}")
