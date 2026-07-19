from __future__ import annotations

import numpy as np
import pytest

from addernet import UniformQuantizer


def test_multifeature_single_sample_is_not_misread_as_many_samples():
    q = UniformQuantizer(16).fit([[0, 10], [15, 25]])
    result = q.transform([7.5, 17.5])
    assert result.shape == (2,)
    np.testing.assert_array_equal(result, [8, 8])


def test_one_feature_vector_preserves_vector_shape():
    q = UniformQuantizer(8).fit([0, 7])
    result = q.transform([0, 3.5, 7])
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, [0, 4, 7])


def test_inverse_transform_and_metadata_roundtrip():
    X = np.array([[0, -10], [10, 20]], dtype=np.float64)
    q = UniformQuantizer(32).fit(X)
    restored = q.inverse_transform(q.transform(X))
    np.testing.assert_allclose(restored, X)
    loaded = UniformQuantizer.from_dict(q.to_dict())
    np.testing.assert_array_equal(loaded.transform(X), q.transform(X))


def test_out_of_range_without_clipping_raises():
    q = UniformQuantizer(8).fit([0, 7])
    with pytest.raises(ValueError):
        q.transform([9], clip=False)
