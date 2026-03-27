#!/usr/bin/env python3
"""
MNIST — AdderNet-HDC
=====================
784 variáveis (pixels), 10 classes (dígitos 0-9).
Zero multiplicação de ponto flutuante na inferência.
"""

import sys
import os
import time
import numpy as np
from sklearn.datasets import fetch_openml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC, _lib, _HV_t

# ---- Carregar MNIST ----
print("Carregando MNIST...")
t0 = time.perf_counter()
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float64) / 255.0
y = mnist.target.astype(np.int32)

X_train, X_test = X[:10000], X[60000:]
y_train, y_test = y[:10000], y[60000:]
t_load = time.perf_counter() - t0

# Escalar pixels para [0, 255] int (a tabela usa int(input) para indexar)
X_train_int = (X_train * 255.0).astype(np.float64)
X_test_int  = (X_test * 255.0).astype(np.float64)

print(f"  Carregado em {t_load:.1f}s")
print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")

# ---- Criar e treinar ----
print(f"\nCriando modelo: 784 variáveis, 10 classes, table_size=256...")
print(f"  (tabela de encoding: ~250MB)")

t0 = time.perf_counter()
model = AdderNetHDC(n_vars=784, n_classes=10, table_size=256, seed=42)
t_create = time.perf_counter() - t0
print(f"  Criado em {t_create:.1f}s")

print(f"  Treinando em {len(X_train)} amostras...")
t0 = time.perf_counter()
model.train(X_train_int, y_train)
t_train = time.perf_counter() - t0
print(f"  Treinado em {t_train:.1f}s")

# ---- Avaliar ----
print(f"  Avaliando em {len(X_test)} amostras...")
t0 = time.perf_counter()
y_pred = model.predict_batch(X_test_int)
t_pred = time.perf_counter() - t0

accuracy = np.mean(y_pred == y_test) * 100

# ---- Relatório ----
print(f"\n{'='*60}")
print(f"  MNIST — AdderNet-HDC (784 variáveis, 10 classes)")
print(f"{'='*60}")
print(f"  Amostras treino:              {len(X_train):,}")
print(f"  Amostras teste:               {len(X_test):,}")
print(f"  Tempo de treino:              {t_train:.1f}s")
print(f"  Tempo de inferência ({len(X_test):,}):   {t_pred*1000:.1f}ms")
print(f"  Throughput:                   {len(X_test)/t_pred:,.0f} pred/s")
print(f"  Acurácia:                     {accuracy:.1f}%")
print(f"  Multiplicações float:         0")

# Matriz de confusão resumida
print(f"\n  Acurácia por dígito:")
for d in range(10):
    mask = y_test == d
    if mask.sum() > 0:
        d_acc = np.mean(y_pred[mask] == d) * 100
        print(f"    {d}: {d_acc:.0f}% ({int(np.sum(y_pred[mask] == d))}/{int(mask.sum())})")

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

# Bundle: voto majoritário de codebook[3] e codebook[8]
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
    print(f"    {d}: {s:.3f}  {bar}")

# Encontrar os dois mais similares
sorted_d = sorted(range(10), key=lambda i: sims[i], reverse=True)
print(f"\n  Mais similar a: {sorted_d[0]} ({sims[sorted_d[0]]:.3f})")
print(f"  Segundo mais:   {sorted_d[1]} ({sims[sorted_d[1]]:.3f})")

if sorted_d[0] in (3, 8) and sorted_d[1] in (3, 8):
    print(f"  Resultado: dígito inventado ficou entre 3 e 8 ✓")
else:
    others = [d for d in sorted_d[:3] if d not in (3, 8)]
    print(f"  Resultado: top-3 inclui {sorted_d[:3]} (esperado: 3 e 8)")
