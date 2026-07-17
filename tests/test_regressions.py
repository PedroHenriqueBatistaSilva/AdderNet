import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from addernet import AdderAttention, AdderBoost, AdderNetHDC, AdderNetLayer


def small_classification(seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-50, 50, size=(90, 3))
    y = ((X[:, 0] + X[:, 1] * 0.4 - X[:, 2] * 0.2) > 0).astype(np.int32)
    return X, y


def test_layer_single_sample_and_duplicate_inputs_are_safe():
    layer = AdderNetLayer(size=256, bias=128, input_min=-20, input_max=20, lr=0.5)
    layer.train([3], [7.5], epochs_raw=10, epochs_expanded=20)
    assert np.all(np.isfinite(layer.predict_batch([-10, 0, 3, 10])))

    dup = AdderNetLayer(size=256, bias=128, input_min=-10, input_max=10, lr=0.1)
    dup.train([2, 2, 2], [1, 1, 1], epochs_raw=5, epochs_expanded=5)
    assert np.isfinite(dup.predict(2))


@pytest.mark.parametrize("hv_dim", [511, 512, 1025, 2500])
def test_hdc_avx_non_multiple_dimensions_and_roundtrip(tmp_path, hv_dim):
    X, y = small_classification()
    model = AdderNetHDC(n_vars=3, n_classes=2, hv_dim=hv_dim, seed=123)
    model.train(X, y, n_iter=2, patience=0)

    normal = model.predict_batch(X[:17])
    fast = model.predict_batch_avx(X[:17])
    threaded = model.predict_batch_mt(X[:17], n_threads=1000)
    assert np.array_equal(normal, fast)
    assert np.array_equal(normal, threaded)

    before_cache = model.predict_batch(X[:17])
    model.warm_cache()
    after_cache = model.predict_batch(X[:17])
    assert np.array_equal(before_cache, after_cache)

    path = tmp_path / f"model_{hv_dim}.bin"
    model.save(str(path))
    loaded = AdderNetHDC.load(str(path))
    assert loaded.n_vars == 3
    assert loaded.n_classes == 2
    assert loaded.hv_dim == hv_dim
    assert loaded.hv_words == (hv_dim + 63) // 64
    assert loaded.codebook.shape == (2, loaded.hv_words)
    assert np.array_equal(normal, loaded.predict_batch(X[:17]))
    assert loaded.classify_hv(loaded.codebook[0]) == 0


def test_hdc_validates_shapes_and_labels():
    model = AdderNetHDC(n_vars=3, n_classes=2, hv_dim=512)
    with pytest.raises(ValueError):
        model.train([[1, 2]], [0])
    with pytest.raises(ValueError):
        model.train([[1, 2, 3]], [2])
    with pytest.raises(ValueError):
        model.predict([1, 2])


def test_attention_api_and_validation():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(2, 3, 5))
    K = rng.normal(size=(2, 4, 5))
    V = rng.normal(size=(2, 4, 6))
    attn = AdderAttention(normalize=True)
    assert attn(Q, K, V).shape == (2, 3, 6)
    assert attn.scores(Q, K).shape == (2, 3, 4)
    with pytest.raises(ValueError):
        attn(Q, K[:, :, :-1], V)


def test_adderboost_fast_epoch_controls_and_validation():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 100, size=(80, 2))
    y = 0.4 * X[:, 0] - 0.2 * X[:, 1] + 3
    model = AdderBoost(
        n_estimators=2,
        learning_rate=0.8,
        epochs_raw=15,
        epochs_expanded=20,
        size=256,
        bias=0,
        input_min=0,
        input_max=100,
        lr=0.2,
    ).fit(X, y)
    pred = model.predict_batch(X)
    assert np.mean(np.abs(pred - y)) < 8
    with pytest.raises(ValueError):
        model.predict_batch(np.zeros((3, 3)))


