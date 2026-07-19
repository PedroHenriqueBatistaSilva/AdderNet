from __future__ import annotations

import numpy as np

from addernet import AdderBoost, AdderCluster, AdderNetAdditiveRegressor


def test_additive_regressor_direct_training_and_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.integers(0, 16, size=(1000, 3)).astype(float)
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    model = AdderNetAdditiveRegressor(table_size=16, backfit_rounds=4).fit(X, y)
    pred = model.predict(X)
    assert np.mean((pred - y) ** 2) < 0.05
    path = tmp_path / "additive"
    model.save(path)
    loaded = AdderNetAdditiveRegressor.load(path)
    np.testing.assert_allclose(loaded.predict(X[:20]), model.predict(X[:20]))


def test_boost_direct_training_reduces_error():
    x = np.arange(32, dtype=float)
    X = np.column_stack([x, x[::-1]])
    y = 2 * X[:, 0] - X[:, 1]
    model = AdderBoost(
        n_estimators=4,
        learning_rate=0.5,
        training_method="direct",
        size=64,
        bias=0,
        input_min=0,
        input_max=31,
    ).fit(X, y)
    baseline = np.mean((y - y.mean()) ** 2)
    mse = np.mean((model.predict_batch(X) - y) ** 2)
    assert mse < baseline * 0.2


def test_cluster_supports_arbitrary_labels():
    X = np.array([[0], [1], [2], [20], [21], [22]], dtype=float)
    y = np.array(["low", "low", "low", "high", "high", "high"])
    cluster = AdderCluster(
        n_nodes=2,
        strategy="feature",
        combination="mean",
        size=32,
        bias=0,
        input_min=0,
        input_max=22,
        random_state=1,
    ).fit(X, y)
    pred = cluster.predict_batch(X)
    assert set(pred) <= {"low", "high"}
    assert np.mean(pred == y) >= 5 / 6
