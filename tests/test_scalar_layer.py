from __future__ import annotations

import struct
import numpy as np
import pytest

from addernet import AdderNetLayer, ReferenceAdderNetLayer


def test_direct_fit_averages_duplicates_and_interpolates():
    layer = AdderNetLayer(size=16, bias=0, input_min=0, input_max=15, lr=0.1)
    layer.fit([0, 0, 5, 10], [2, 4, 13, 23])
    assert layer.predict(0) == pytest.approx(3.0)
    assert layer.predict(5) == pytest.approx(13.0)
    assert layer.predict(10) == pytest.approx(23.0)
    assert layer.predict(7) == pytest.approx(17.0)


def test_native_interpolation_no_longer_truncates_after_256_samples():
    x = np.arange(300, dtype=np.float64)
    y = np.zeros(300, dtype=np.float64)
    y[256:] = 100.0
    layer = AdderNetLayer(size=512, bias=0, input_min=0, input_max=299, lr=1.0)
    layer.train(x, y, epochs_raw=0, epochs_expanded=1)
    assert layer.predict(299) == pytest.approx(1.0)


def test_partial_fit_blends_existing_table():
    layer = AdderNetLayer(size=8, bias=0, input_min=0, input_max=7)
    layer.fit([0, 7], [0, 14])
    before = layer.predict(7)
    layer.partial_fit([7], [30], interpolate=False, blend=0.25)
    assert layer.predict(7) == pytest.approx(0.75 * before + 0.25 * 30)


def test_portable_and_native_roundtrip(tmp_path):
    layer = AdderNetLayer(size=32, bias=3, input_min=-3, input_max=20, lr=0.2)
    layer.fit([-3, 4, 20], [1.0, 5.0, 11.0])
    expected = layer.predict_batch(np.arange(-3, 21))

    native_path = tmp_path / "model.bin"
    portable_path = tmp_path / "model.npz"
    layer.save(native_path)
    layer.save_portable(portable_path)

    native = AdderNetLayer.load(native_path)
    portable = AdderNetLayer.load_portable(portable_path)
    np.testing.assert_allclose(native.predict_batch(np.arange(-3, 21)), expected)
    np.testing.assert_allclose(portable.predict_batch(np.arange(-3, 21)), expected)


def test_corrupt_native_header_is_rejected(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(struct.pack("=iiii d", 2**30, 0, 0, 10, 0.1))
    with pytest.raises(OSError):
        AdderNetLayer.load(path)


def test_context_manager_and_closed_state():
    with AdderNetLayer() as layer:
        layer.fit([0, 1], [2, 3])
        assert layer.predict(1) == pytest.approx(3)
    with pytest.raises(RuntimeError, match="closed"):
        layer.predict(1)


def test_native_and_reference_direct_fit_match():
    x = np.array([-3, 0, 0, 4, 12], dtype=np.float64)
    y = np.array([1, 2, 4, 8, 16], dtype=np.float64)
    native = AdderNetLayer(size=32, bias=3, input_min=-3, input_max=12).fit(x, y)
    reference = ReferenceAdderNetLayer(size=32, bias=3, input_min=-3, input_max=12).fit(x, y)
    grid = np.arange(-3, 13)
    np.testing.assert_allclose(native.predict_batch(grid), reference.predict_batch(grid))


def test_input_validation_and_return_codes():
    layer = AdderNetLayer()
    with pytest.raises(ValueError):
        layer.fit([], [])
    with pytest.raises(ValueError):
        layer.train([1], [1], epochs_raw=-1)
    with pytest.raises(ValueError):
        layer.predict(float("nan"))
