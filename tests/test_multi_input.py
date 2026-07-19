from __future__ import annotations

import numpy as np
import pytest

from addernet import AdderNetClassifier, AdderNetMultiInputLayer


def full_grid(bins=16):
    x = np.arange(bins, dtype=np.float64)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    X = np.column_stack([xx.ravel(), yy.ravel()])
    return X, xx.ravel(), yy.ravel()


def test_two_inputs_and_multiple_outputs_are_learned():
    X, x, y = full_grid(16)
    targets = np.column_stack([x + y, x - y, x * y])
    model = AdderNetMultiInputLayer(
        bins=16,
        interactions="auto",
        max_interactions=1,
        backfit_rounds=3,
        interaction_rounds=2,
    ).fit(X, targets)

    pred = model.predict_batch(X)
    np.testing.assert_allclose(pred, targets, atol=1e-10)
    np.testing.assert_allclose(model.predict(7, 9), [16, -2, 63], atol=1e-10)
    assert model.interaction_pairs_ == [(0, 1)]
    assert model.score(X, targets) == pytest.approx(1.0)


def test_additive_only_model_handles_many_inputs():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 32, size=(2000, 4)).astype(np.float64)
    y = X[:, 0] - 2 * X[:, 1] + 3 * X[:, 2] + 0.5 * X[:, 3]
    model = AdderNetMultiInputLayer(bins=32, interactions="none", backfit_rounds=4).fit(X, y)
    assert model.score(X, y) > 0.999
    assert model.predict(1, 2, 3, 4) == pytest.approx(8.0, abs=0.01)


def test_quantized_inference_matches_float_inference():
    X, x, y = full_grid(8)
    target = x * y
    model = AdderNetMultiInputLayer(bins=8).fit(X, target)
    Q = model.quantizer.transform(X)
    np.testing.assert_allclose(model.predict_quantized(Q)[:, 0], model.predict_batch(X))


def test_explanations_sum_to_prediction():
    X, x, y = full_grid(8)
    model = AdderNetMultiInputLayer(bins=8).fit(X, x * y)
    explanation = model.explain([[2, 3], [5, 6]])
    reconstructed = (
        explanation["intercept"]
        + explanation["additive"].sum(axis=1)
        + explanation["interaction"].sum(axis=1)
    )
    np.testing.assert_allclose(reconstructed, explanation["prediction"])


def test_multi_input_roundtrip(tmp_path):
    X, x, y = full_grid(8)
    target = np.column_stack([x + y, x * y])
    model = AdderNetMultiInputLayer(bins=8).fit(X, target)
    path = tmp_path / "multi.npz"
    model.save(path)
    loaded = AdderNetMultiInputLayer.load(path)
    np.testing.assert_allclose(loaded.predict_batch(X), target, atol=1e-10)
    assert loaded.memory_bytes_ == model.memory_bytes_


def test_classifier_supports_non_contiguous_labels_and_xor(tmp_path):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([10, 20, 20, 10])
    clf = AdderNetClassifier(bins=2, interactions="auto", max_interactions=1).fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)
    assert clf.score(X, y) == 1.0
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    directory = tmp_path / "classifier"
    clf.save(directory)
    loaded = AdderNetClassifier.load(directory)
    np.testing.assert_array_equal(loaded.predict(X), y)
