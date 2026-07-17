import numpy as np

from addernet import AdderNetAdditiveRegressor, ReferenceAdderNetLayer, UniformQuantizer


def test_reference_layer_learns_linear_and_roundtrips(tmp_path):
    layer = ReferenceAdderNetLayer(size=64, input_min=0, input_max=31, lr=0.25)
    x = np.array([0, 8, 16, 24, 31], dtype=float)
    y = 1.5 * x + 2
    layer.train(x, y, epochs_raw=100, epochs_expanded=150)
    assert np.mean(np.abs(layer.predict_batch(x) - y)) < 1.0
    path = tmp_path / "reference.npz"
    layer.save(path)
    restored = ReferenceAdderNetLayer.load(path)
    assert np.array_equal(layer.offset_table, restored.offset_table)


def test_uniform_quantizer_constant_column_is_safe():
    q = UniformQuantizer(16)
    X = np.array([[3.0, 0.0], [3.0, 5.0], [3.0, 10.0]])
    out = q.fit_transform(X)
    assert out[:, 0].tolist() == [0, 0, 0]
    assert out.min() >= 0 and out.max() <= 15


def test_additive_regressor_multivariate_save_load(tmp_path):
    rng = np.random.default_rng(12)
    X = rng.uniform(-2, 2, size=(160, 3))
    y = 2.0 * X[:, 0] - 0.7 * X[:, 1] + np.sin(X[:, 2])
    model = AdderNetAdditiveRegressor(table_size=64, lr=0.05,
                                      backfit_rounds=2,
                                      epochs_raw=30, epochs_expanded=60).fit(X, y)
    pred = model.predict(X)
    assert np.mean(np.abs(pred - y)) < 0.65
    directory = tmp_path / "model"
    model.save(directory)
    restored = AdderNetAdditiveRegressor.load(directory)
    assert np.allclose(pred, restored.predict(X))
