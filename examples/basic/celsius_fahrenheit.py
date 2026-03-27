#!/usr/bin/env python3
"""
Example 1: Celsius → Fahrenheit conversion
============================================
Demonstrates AdderNet learning F = C * 1.8 + 32
using only addition (zero multiplication at inference).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))

from addernet import AdderNetLayer

# Training data: known Celsius→Fahrenheit pairs
training_data = [
    (0, 32), (5, 41), (10, 50), (15, 59), (20, 68),
    (25, 77), (30, 86), (35, 95), (40, 104), (50, 122),
    (60, 140), (70, 158), (80, 176), (90, 194), (100, 212),
    (-10, 14), (-20, -4), (37, 98.6),
]

inputs  = [c for c, _ in training_data]
targets = [f for _, f in training_data]

# Create and train
print("=" * 60)
print("  AdderNet: Celsius -> Fahrenheit")
print("  Inference: single table lookup (zero multiplication)")
print("=" * 60)

layer = AdderNetLayer(size=256, bias=50, input_min=-50, input_max=200, lr=0.1)
print(f"\n  {layer}")
print(f"  Training on {len(inputs)} samples...")

layer.train(inputs, targets, epochs_raw=1000, epochs_expanded=4000)

# Test predictions
print(f"\n  {'C':>6} | {'Real':>10} | {'AdderNet':>10} | {'Error':>10}")
print("  " + "-" * 44)

test_cases = [-40, -20, -10, 0, 10, 20, 25, 30, 37, 50, 80, 100, 150, 200]
for c in test_cases:
    real = c * 1.8 + 32
    pred = layer.predict(c)
    err  = pred - real
    print(f"  {c:>6.1f} | {real:>10.2f} | {pred:>10.2f} | {err:>+10.2f}")

# Save and reload
layer.save("/tmp/celsius_fahrenheit.bin")
loaded = AdderNetLayer.load("/tmp/celsius_fahrenheit.bin")
print(f"\n  Save/load: 37C -> {loaded.predict(37):.2f} F")
