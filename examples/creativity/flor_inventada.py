#!/usr/bin/env python3
"""
Flor inventada — Misturar codebooks HDC
==========================================
Usa voto majoritário (bundle) para criar uma flor "híbrida" a partir de duas classes
do Iris e verifica se o classificador a posiciona entre as duas originais.
"""

import sys
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC, _lib, _HV_t

# ---- Treinar modelo no Iris ----
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

model = AdderNetHDC(n_vars=4, n_classes=3, table_size=256, seed=42)
model.train(X_train, y_train)

class_names = list(iris.target_names)  # ['setosa', 'versicolor', 'virginica']

# ---- Obter codebook ----
cb = model.codebook  # lista de 3 numpy arrays (cada um 157 × uint64)

# Converter numpy → ctypes para chamar hv_bind / hv_similarity
def numpy_to_hv(arr):
    hv = _HV_t()
    for i in range(len(arr)):
        hv[i] = int(arr[i])
    return hv

def hv_to_numpy(hv):
    return np.array([hv[i] for i in range(157)], dtype=np.uint64)

cb_ctypes = [numpy_to_hv(c) for c in cb]

# ---- Passo 1: Misturar setosa (0) e versicolor (1) ----
flor_nova = _HV_t()
vecs = (_HV_t * 2)()             # array de 2 hv_t
vecs[0] = cb_ctypes[0]           # setosa
vecs[1] = cb_ctypes[1]           # versicolor
_lib.hv_bundle(flor_nova, vecs, 2)  # voto majoritário

# ---- Passo 2: Classificar a flor inventada ----
# O classificador interno faz: argmin_c hamming(query, codebook[c])
# Vamos reproduzir manualmente já que a flor inventada não tem features
best_c = 0
best_d = _lib.hv_hamming(flor_nova, cb_ctypes[0])
for c in range(1, 3):
    d = _lib.hv_hamming(flor_nova, cb_ctypes[c])
    if d < best_d:
        best_d = d
        best_c = c

# ---- Passo 3: Medir similaridades ----
sims = []
for c in range(3):
    sims.append(_lib.hv_similarity(flor_nova, cb_ctypes[c]))

# ---- Relatório ----
print(f"Flor inventada ({class_names[0]} BUNDLE {class_names[1]}):")
print(f"  Similaridade com setosa:     {sims[0]:.2f}")
print(f"  Similaridade com versicolor: {sims[1]:.2f}")
print(f"  Similaridade com virginica:  {sims[2]:.2f}")
print(f"  Classificada como: {class_names[best_c]}")

# ---- Verificar: ficou entre as duas originais? ----
if sims[2] < sims[0] and sims[2] < sims[1]:
    print(f"\n  virginica está mais longe ({sims[2]:.2f}) que as duas originais.")
    print(f"  Conclusão: a flor inventada ficou entre as duas originais e longe da terceira.")
else:
    print(f"\n  Conclusão: virou lixo.")
