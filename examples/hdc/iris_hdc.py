#!/usr/bin/env python3
"""
Example: Iris classification with AdderNet-HDC
================================================
Demonstrates multivariate classification (4 variables, 3 classes)
using Hyperdimensional Computing — zero multiplication at inference.
"""

import sys
import os
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
from addernet_hdc import AdderNetHDC

# ---- Load data ----
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("=" * 60)
print("  AdderNet-HDC: Iris Classification")
print("  4 variables, 3 classes, zero multiplication at inference")
print("=" * 60)

print(f"\n  Dataset: {len(X)} samples, {X.shape[1]} features, {len(class_names)} classes")
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print(f"  Features: {', '.join(feature_names)}")

# ---- Train ----
print(f"\n  Training AdderNet-HDC (table_size=256, seed=42)...")
t0 = time.perf_counter()
model = AdderNetHDC(n_vars=4, n_classes=3, table_size=256, seed=42)
model.train(X_train, y_train)
t_train = time.perf_counter() - t0
print(f"  Training time: {t_train*1000:.1f}ms")

# ---- Predict ----
t0 = time.perf_counter()
y_pred = model.predict_batch(X_test)
t_pred = time.perf_counter() - t0

accuracy = np.mean(y_pred == y_test) * 100

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"\n  Accuracy: {accuracy:.1f}% ({np.sum(y_pred == y_test)}/{len(y_test)})")
print(f"  Inference time: {t_pred*1000:.2f}ms for {len(y_test)} samples")
print(f"  Throughput: {len(y_test)/t_pred:,.0f} predictions/second")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print(f"  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':>15} {'Pred 0':>8} {'Pred 1':>8} {'Pred 2':>8}")
for i, name in enumerate(class_names):
    print(f"  {name:>15} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")

# ---- Show some predictions ----
print(f"\n  Sample predictions:")
print(f"  {'Features':>40} | {'Real':>10} | {'Pred':>10} | {'OK':>4}")
print(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*4}")
for i in range(0, min(len(X_test), 15)):
    feats = ", ".join(f"{v:.1f}" for v in X_test[i])
    real = class_names[y_test[i]]
    pred = class_names[y_pred[i]]
    ok = "OK" if y_test[i] == y_pred[i] else "ERR"
    print(f"  {feats:>40} | {real:>10} | {pred:>10} | {ok:>4}")

# ---- Save/load ----
model.save("/tmp/iris_hdc.bin")
loaded = AdderNetHDC.load("/tmp/iris_hdc.bin")
y_loaded = loaded.predict_batch(X_test)
loaded_acc = np.mean(y_loaded == y_test) * 100
print(f"\n  Save/load: accuracy preserved = {loaded_acc == accuracy} ({loaded_acc:.1f}%)")
