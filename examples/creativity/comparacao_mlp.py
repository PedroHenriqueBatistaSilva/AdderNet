#!/usr/bin/env python3
"""
Comparação MNIST — AdderNet-HDC vs MLP padrão
================================================
Mesmos dados (blocos 2x2, 196 variáveis), mesmas condições.
"""

import sys
import os
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC

# ---- Carregar MNIST ----
print("Carregando MNIST...")
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
t0 = time.perf_counter()
X_train_blocos = np.array([blocos_2x2(x) for x in X_train_raw])
X_test_blocos  = np.array([blocos_2x2(x) for x in X_test_raw])
t_pre = time.perf_counter() - t0
print(f"  {t_pre:.1f}s")

X_train_int = (X_train_blocos * 255.0).astype(np.float64)
X_test_int  = (X_test_blocos * 255.0).astype(np.float64)

# ================================================================
#  AdderNet-HDC
# ================================================================
print("\n[1/2] AdderNet-HDC (196 vars, 10 classes, table=256)")
print("  Treinando com 60k amostras...")

t0 = time.perf_counter()
hdc = AdderNetHDC(n_vars=196, n_classes=10, table_size=256, seed=42)
hdc.train(X_train_int, y_train)
t_hdc_train = time.perf_counter() - t0
print(f"  Treino: {t_hdc_train:.1f}s")

t0 = time.perf_counter()
y_pred_hdc = hdc.predict_batch(X_test_int)
t_hdc_pred = time.perf_counter() - t0
acc_hdc = np.mean(y_pred_hdc == y_test) * 100
print(f"  Inferência: {t_hdc_pred:.1f}s")

# ================================================================
#  MLP
# ================================================================
print("\n[2/2] MLP sklearn (hidden_layer_sizes=(256,), max_iter=20)")
print("  Treinando com 60k amostras...")

t0 = time.perf_counter()
mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, random_state=42)
mlp.fit(X_train_blocos, y_train)
t_mlp_train = time.perf_counter() - t0
print(f"  Treino: {t_mlp_train:.1f}s")

t0 = time.perf_counter()
y_pred_mlp = mlp.predict(X_test_blocos)
t_mlp_pred = time.perf_counter() - t0
acc_mlp = np.mean(y_pred_mlp == y_test) * 100
print(f"  Inferência: {t_mlp_pred:.3f}s")

# ================================================================
#  Estimativa de memória
# ================================================================
# AdderNet-HDC: enc_table = 196*256*1250 bytes, position = 196*1250, codebook = 10*1250
hdc_bytes = 196 * 256 * 1250 + 196 * 1250 + 10 * 1250
# MLP: coefs + biases
mlp_params = sum(p.size for p in mlp.coefs_) + sum(p.size for p in mlp.intercepts_)
mlp_bytes = mlp_params * 8  # float64

# ================================================================
#  Piores dígitos
# ================================================================
hdc_by_digit = []
mlp_by_digit = []
for d in range(10):
    mask = y_test == d
    hdc_by_digit.append(np.mean(y_pred_hdc[mask] == d) * 100)
    mlp_by_digit.append(np.mean(y_pred_mlp[mask] == d) * 100)

hdc_worst = sorted(range(10), key=lambda i: hdc_by_digit[i])
mlp_worst = sorted(range(10), key=lambda i: mlp_by_digit[i])

# ================================================================
#  Relatório
# ================================================================
print(f"\n{'='*60}")
print(f"  COMPARAÇÃO MNIST — Blocos 2x2 (196 vars)")
print(f"{'='*60}")
print(f"")
print(f"  {'':24} {'AdderNet-HDC':>14} {'MLP padrão':>14}")
print(f"  {'-'*24} {'-'*14} {'-'*14}")
print(f"  {'Acurácia':<24} {f'{acc_hdc:.1f}%':>14} {f'{acc_mlp:.1f}%':>14}")
print(f"  {'Treino':<24} {f'{t_hdc_train:.1f}s':>14} {f'{t_mlp_train:.1f}s':>14}")
print(f"  {'Inferência 10k':<24} {f'{t_hdc_pred:.1f}s':>14} {f'{t_mlp_pred:.3f}s':>14}")
print(f"  {'Throughput':<24} {f'{10000/t_hdc_pred:,.0f}/s':>14} {f'{10000/t_mlp_pred:,.0f}/s':>14}")
print(f"  {'Multiplicações float':<24} {'0':>14} {'~bilhões':>14}")
print(f"  {'Params':<24} {'~50M HV bits':>14} {f'{mlp_params:,}':>14}")
print(f"  {'Memória modelo':<24} {f'{hdc_bytes/1024/1024:.1f}MB':>14} {f'{mlp_bytes/1024:.1f}KB':>14}")
print(f"")
print(f"  Pior dígito AdderNet-HDC: {hdc_worst[0]} ({hdc_by_digit[hdc_worst[0]]:.0f}%), {hdc_worst[1]} ({hdc_by_digit[hdc_worst[1]]:.0f}%)")
print(f"  Pior dígito MLP:          {mlp_worst[0]} ({mlp_by_digit[mlp_worst[0]]:.0f}%), {mlp_worst[1]} ({mlp_by_digit[mlp_worst[1]]:.0f}%)")
print(f"{'='*60}")

if acc_mlp > 90:
    print(f"\n  MLP > 90%: AdderNet-HDC perde em acurácia, ganha em princípio (zero mul)")
elif acc_mlp < 85:
    print(f"\n  MLP < 85%: AdderNet-HDC é competitivo")
