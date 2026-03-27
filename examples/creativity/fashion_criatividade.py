#!/usr/bin/env python3
"""
Fashion-MNIST — Criatividade com HDC
======================================
Testa se o modelo consegue "imaginar" itens coerentes
via bundle de classes e ruído controlado (temperatura).
"""

import sys
import os
import time
import numpy as np
from sklearn.datasets import fetch_openml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC, _lib, _HV_t

NOMES = [
    "camiseta", "calça", "suéter", "vestido", "casaco",
    "sandália", "camisa", "tênis", "bolsa", "bota"
]

# ---- Helpers ----
def numpy_to_hv(arr):
    hv = _HV_t()
    for i in range(len(arr)):
        hv[i] = int(arr[i])
    return hv

def blocos_2x2(imagem):
    img = imagem.reshape(28, 28)
    r = []
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            r.append(img[i:i+2, j:j+2].mean())
    return np.array(r)

def classificar_hv(hv, cbs):
    best_c, best_d = 0, _lib.hv_hamming(hv, cbs[0])
    for c in range(1, 10):
        d = _lib.hv_hamming(hv, cbs[c])
        if d < best_d:
            best_d, best_c = d, c
    return best_c

def sims_para_todos(hv, cbs):
    return [_lib.hv_similarity(hv, cbs[c]) for c in range(10)]

# ---- Parte 1: Treinar ----
print("=" * 60)
print("  Fashion-MNIST — Criatividade com HDC")
print("=" * 60)

print("\nCarregando Fashion-MNIST...")
fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X_all = fashion.data.astype(np.float64) / 255.0
y_all = fashion.target.astype(np.int32)
X_train_raw, X_test_raw = X_all[:60000], X_all[60000:]
y_train, y_test = y_all[:60000], y_all[60000:]

print("Pré-processando (blocos 2x2)...")
X_train_blocos = np.array([blocos_2x2(x) for x in X_train_raw])
X_test_blocos  = np.array([blocos_2x2(x) for x in X_test_raw])
X_train_int = (X_train_blocos * 255.0).astype(np.float64)
X_test_int  = (X_test_blocos * 255.0).astype(np.float64)

print("Treinando (196 vars, 10 classes, table=256)...")
t0 = time.perf_counter()
model = AdderNetHDC(n_vars=196, n_classes=10, table_size=256, seed=42)
model.train(X_train_int, y_train)
t_train = time.perf_counter() - t0

y_pred = model.predict_batch(X_test_int)
acc = np.mean(y_pred == y_test) * 100
print(f"  Treino: {t_train:.1f}s")
print(f"  Acurácia base: {acc:.1f}%")

cb = model.codebook
cbs = [numpy_to_hv(c) for c in cb]

# ================================================================
print(f"\n{'='*60}")
print(f"  Experimento A — Bundle camiseta + casaco")
print(f"{'='*60}")

vecs_a = (_HV_t * 2)()
vecs_a[0] = cbs[0]  # camiseta
vecs_a[1] = cbs[4]  # casaco
roupa_nova = _HV_t()
_lib.hv_bundle(roupa_nova, vecs_a, 2)

sims_a = sims_para_todos(roupa_nova, cbs)
classe_a = classificar_hv(roupa_nova, cbs)

print(f"\n  Similaridades:")
for c in range(10):
    bar = "#" * int(sims_a[c] * 30)
    marker = " <---" if c in (0, 4) else ""
    print(f"    {c} {NOMES[c]:>10}: {sims_a[c]:.3f}  {bar}{marker}")
print(f"\n  Classificada como: {classe_a} ({NOMES[classe_a]})")
coerente_a = classe_a in (0, 4)
print(f"  É coerente? {'sim' if coerente_a else 'não'}")

# ================================================================
print(f"\n{'='*60}")
print(f"  Experimento B — Temperatura no tênis")
print(f"{'='*60}")

print(f"\n  {'Temp':>6} | {'Classe':>12} | {'Coerente':>8}")
print(f"  {'-'*6} | {'-'*12} | {'-'*8}")

temp_colapso = None
for temp in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    ruidosa = _HV_t()
    _lib.hv_add_noise(ruidosa, cbs[7], temp)  # tênis = classe 7
    classe_b = classificar_hv(ruidosa, cbs)
    coerente_b = classe_b == 7
    if not coerente_b and temp_colapso is None and temp > 0:
        temp_colapso = temp
    print(f"  {temp:>5.2f} | {classe_b:>2} {NOMES[classe_b]:>9} | {'sim' if coerente_b else 'não':>8}")

if temp_colapso:
    print(f"\n  Ponto de colapso: temp={temp_colapso:.2f}")
else:
    print(f"\n  Não colapsou nas temperaturas testadas")

# ================================================================
print(f"\n{'='*60}")
print(f"  Experimento C — Gradiente sandália → bota")
print(f"{'='*60}")

print(f"\n  {'Alpha':>6} | {'Classe':>12} | {'É bota?':>7} | {'É sandália?':>11}")
print(f"  {'-'*6} | {'-'*12} | {'-'*7} | {'-'*11}")

passos_coerentes = 0
for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
    n_sandalia = int((1.0 - alpha) * 10)
    n_bota = int(alpha * 10)
    n_total = n_sandalia + n_bota
    if n_total == 0:
        n_total = 1
        n_sandalia = 1

    vecs_c = (_HV_t * n_total)()
    idx = 0
    for _ in range(n_sandalia):
        vecs_c[idx] = cbs[5]  # sandália
        idx += 1
    for _ in range(n_bota):
        vecs_c[idx] = cbs[9]  # bota
        idx += 1

    passo = _HV_t()
    _lib.hv_bundle(passo, vecs_c, n_total)
    classe_c = classificar_hv(passo, cbs)

    eh_bota = classe_c == 9
    eh_sandalia = classe_c == 5
    eh_coerente = eh_bota or eh_sandalia
    if eh_coerente:
        passos_coerentes += 1

    print(f"  {alpha:>5.2f} | {classe_c:>2} {NOMES[classe_c]:>9} | {'sim' if eh_bota else 'não':>7} | {'sim' if eh_sandalia else 'não':>11}")

print(f"\n  Passos coerentes: {passos_coerentes}/5")
print(f"  Transição coerente? {'sim' if passos_coerentes >= 3 else 'não'}")

# ================================================================
print(f"\n{'='*60}")
print(f"  RESUMO")
print(f"{'='*60}")
print(f"  Acurácia base:     {acc:.1f}%")
print(f"  A (bundle):        {'✓' if coerente_a else '✗'} classificado como {NOMES[classe_a]}")
print(f"  B (temperatura):   colapso em temp={temp_colapso:.2f}" if temp_colapso else "  B (temperatura):   não colapsou")
print(f"  C (gradiente):     {passos_coerentes}/5 passos coerentes {'✓' if passos_coerentes >= 3 else '✗'}")
