#!/usr/bin/env python3
"""
AdderNet Python Bindings — ctypes interface to libaddernet.so
==============================================================

Usage:
    from addernet import AdderNetLayer

    layer = AdderNetLayer(size=256, bias=50, input_min=-50, input_max=200, lr=0.1)
    layer.train(inputs, targets, epochs_raw=1000, epochs_expanded=4000)
    result = layer.predict(37.0)
    layer.save("model.bin")
"""

import os
import ctypes
import numpy as np

# ---- Locate shared library ----

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(_HERE, "..", "build")
_LIB_NAMES = [
    os.path.join(_BUILD, "libaddernet.so"),
    os.path.join(_HERE, "libaddernet.so"),
    os.path.join(_HERE, "libaddernet.dylib"),
    "libaddernet.so",
]

_lib = None
for _name in _LIB_NAMES:
    try:
        _lib = ctypes.CDLL(_name)
        break
    except OSError:
        continue

if _lib is None:
    raise OSError(
        "Cannot find libaddernet.so. "
        "Build it first: cd addernet_lib && make"
    )

# ---- Opaque pointer type ----

_AnLayerPtr = ctypes.c_void_p

# ---- Function signatures ----

_lib.an_layer_create.restype  = _AnLayerPtr
_lib.an_layer_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]

_lib.an_layer_free.restype  = None
_lib.an_layer_free.argtypes = [_AnLayerPtr]

_lib.an_train.restype  = ctypes.c_int
_lib.an_train.argtypes = [
    _AnLayerPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

_lib.an_predict.restype  = ctypes.c_double
_lib.an_predict.argtypes = [_AnLayerPtr, ctypes.c_double]

_lib.an_predict_batch.restype  = ctypes.c_int
_lib.an_predict_batch.argtypes = [
    _AnLayerPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

_lib.an_save.restype  = ctypes.c_int
_lib.an_save.argtypes = [_AnLayerPtr, ctypes.c_char_p]

_lib.an_load.restype  = _AnLayerPtr
_lib.an_load.argtypes = [ctypes.c_char_p]

_lib.an_get_offset.restype  = ctypes.c_int
_lib.an_get_offset.argtypes = [_AnLayerPtr, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

_lib.an_get_size.restype  = ctypes.c_int
_lib.an_get_size.argtypes = [_AnLayerPtr]

_lib.an_get_bias.restype  = ctypes.c_int
_lib.an_get_bias.argtypes = [_AnLayerPtr]

_lib.an_get_input_min.restype  = ctypes.c_int
_lib.an_get_input_min.argtypes = [_AnLayerPtr]

_lib.an_get_input_max.restype  = ctypes.c_int
_lib.an_get_input_max.argtypes = [_AnLayerPtr]

_lib.an_get_lr.restype  = ctypes.c_double
_lib.an_get_lr.argtypes = [_AnLayerPtr]


# ---- Python wrapper ----

class AdderNetLayer:
    """
    Python wrapper around the C an_layer struct.
    Provides train / predict / predict_batch / save / load using numpy arrays.
    """

    def __init__(self, size=256, bias=50, input_min=-50, input_max=200, lr=0.1,
                 _ptr=None):
        if _ptr is not None:
            self._ptr = _ptr
        else:
            self._ptr = _lib.an_layer_create(size, bias, input_min, input_max, lr)
            if not self._ptr:
                raise MemoryError("an_layer_create failed")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.an_layer_free(self._ptr)
            self._ptr = None

    def train(self, inputs, targets, epochs_raw=1000, epochs_expanded=4000):
        """Train on input→target pairs. Accepts lists or numpy arrays."""
        inputs  = np.ascontiguousarray(inputs,  dtype=np.float64)
        targets = np.ascontiguousarray(targets, dtype=np.float64)
        n = len(inputs)
        if len(targets) != n:
            raise ValueError("inputs and targets must have same length")
        ret = _lib.an_train(
            self._ptr,
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            targets.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n, epochs_raw, epochs_expanded,
        )
        if ret != 0:
            raise RuntimeError("an_train failed")

    def predict(self, x):
        """Single prediction. Returns float."""
        return _lib.an_predict(self._ptr, float(x))

    def predict_batch(self, inputs):
        """Batch prediction. Accepts numpy array, returns numpy array."""
        inputs  = np.ascontiguousarray(inputs, dtype=np.float64)
        n = len(inputs)
        outputs = np.empty(n, dtype=np.float64)
        _lib.an_predict_batch(
            self._ptr,
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
        )
        return outputs

    def save(self, path):
        """Save layer to binary file."""
        ret = _lib.an_save(self._ptr, path.encode("utf-8"))
        if ret != 0:
            raise IOError(f"an_save failed: {path}")

    @classmethod
    def load(cls, path):
        """Load layer from binary file. Returns new AdderNetLayer."""
        ptr = _lib.an_load(path.encode("utf-8"))
        if not ptr:
            raise IOError(f"an_load failed: {path}")
        return cls(_ptr=ptr)

    @property
    def offset_table(self):
        """Return the offset table as a numpy array."""
        n = _lib.an_get_size(self._ptr)
        buf = np.empty(n, dtype=np.float64)
        _lib.an_get_offset(
            self._ptr,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
        )
        return buf

    @property
    def size(self):
        return _lib.an_get_size(self._ptr)

    @property
    def bias(self):
        return _lib.an_get_bias(self._ptr)

    @property
    def input_min(self):
        return _lib.an_get_input_min(self._ptr)

    @property
    def input_max(self):
        return _lib.an_get_input_max(self._ptr)

    @property
    def lr(self):
        return _lib.an_get_lr(self._ptr)

    def __repr__(self):
        return (f"AdderNetLayer(size={self.size}, bias={self.bias}, "
                f"range=[{self.input_min},{self.input_max}], lr={self.lr})")
