"""Small reproducible smoke test for installed AdderNet builds."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np


def main() -> int:
    from . import AdderAttention, AdderNetHDC, AdderNetLayer, hdc_detect_backend

    layer = AdderNetLayer(size=256, bias=0, input_min=0, input_max=100, lr=1.0)
    x = np.array([0, 25, 50, 75, 100], dtype=np.float64)
    y = 0.5 * x + 3
    layer.train(x, y, epochs_raw=80, epochs_expanded=100)
    mae = float(np.mean(np.abs(layer.predict_batch(x) - y)))

    X = np.array([[0, 0], [0, 1], [10, 9], [9, 10], [1, 0], [8, 8]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
    hdc = AdderNetHDC(n_vars=2, n_classes=2, hv_dim=512, seed=7)
    hdc.train(X, labels, n_iter=2, patience=0)
    acc = float(np.mean(hdc.predict_batch(X) == labels))

    Q = np.zeros((1, 1, 2)); K = np.zeros((1, 2, 2)); V = np.ones((1, 2, 3))
    attention_shape = AdderAttention(normalize=True)(Q, K, V).shape

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layer.bin"
        layer.save(str(path))
        restored = AdderNetLayer.load(str(path))
        roundtrip = bool(np.allclose(restored.predict_batch(x), layer.predict_batch(x)))

    print(f"backend={hdc_detect_backend()}")
    print(f"layer_mae={mae:.6f}")
    print(f"hdc_training_accuracy={acc:.3f}")
    print(f"attention_shape={attention_shape}")
    print(f"save_load_roundtrip={roundtrip}")
    ok = mae < 1.0 and acc >= 0.8 and attention_shape == (1, 1, 3) and roundtrip
    print("SELFTEST: PASS" if ok else "SELFTEST: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
