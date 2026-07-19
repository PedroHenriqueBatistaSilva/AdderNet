"""Safe ctypes bindings for the native scalar AdderNet lookup layer."""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if sys.platform == "darwin":
    _BASE_NAMES = ("libaddernet.dylib",)
elif os.name == "nt":
    _BASE_NAMES = ("addernet.dll",)
else:
    _BASE_NAMES = ("libaddernet.so",)

_lib = None
for _name in [str(_HERE / n) for n in _BASE_NAMES] + list(_BASE_NAMES):
    try:
        _lib = ctypes.CDLL(_name)
        break
    except OSError:
        continue
if _lib is None:
    raise OSError("Cannot load the AdderNet native library. Run `addernet-build`.")

_AnLayerPtr = ctypes.c_void_p
_DoublePtr = ctypes.POINTER(ctypes.c_double)

_lib.an_layer_create.restype = _AnLayerPtr
_lib.an_layer_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
_lib.an_layer_free.restype = None
_lib.an_layer_free.argtypes = [_AnLayerPtr]
_lib.an_train.restype = ctypes.c_int
_lib.an_train.argtypes = [_AnLayerPtr, _DoublePtr, _DoublePtr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.an_fit_direct.restype = ctypes.c_int
_lib.an_fit_direct.argtypes = [_AnLayerPtr, _DoublePtr, _DoublePtr, ctypes.c_int, ctypes.c_int, ctypes.c_double]
_lib.an_predict.restype = ctypes.c_double
_lib.an_predict.argtypes = [_AnLayerPtr, ctypes.c_double]
_lib.an_predict_batch.restype = ctypes.c_int
_lib.an_predict_batch.argtypes = [_AnLayerPtr, _DoublePtr, _DoublePtr, ctypes.c_int]
_lib.an_save.restype = ctypes.c_int
_lib.an_save.argtypes = [_AnLayerPtr, ctypes.c_char_p]
_lib.an_load.restype = _AnLayerPtr
_lib.an_load.argtypes = [ctypes.c_char_p]
_lib.an_get_offset.restype = ctypes.c_int
_lib.an_get_offset.argtypes = [_AnLayerPtr, _DoublePtr, ctypes.c_int]
_lib.an_set_offset.restype = ctypes.c_int
_lib.an_set_offset.argtypes = [_AnLayerPtr, _DoublePtr, ctypes.c_int]
for _func in ("size", "bias", "input_min", "input_max"):
    fn = getattr(_lib, f"an_get_{_func}")
    fn.restype = ctypes.c_int
    fn.argtypes = [_AnLayerPtr]
_lib.an_get_lr.restype = ctypes.c_double
_lib.an_get_lr.argtypes = [_AnLayerPtr]


def _as_vector(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    int_info = np.iinfo(np.int32)
    if arr.size and (arr.min() < int_info.min or arr.max() > int_info.max):
        raise ValueError(f"{name} values must fit in a signed 32-bit integer after truncation")
    return arr


class AdderNetLayer:
    """One-input/one-output lookup layer backed by optimized C.

    ``train`` preserves the original iterative learning algorithm. ``fit`` is
    the recommended O(n + input_range) direct estimator: duplicate integer
    inputs are averaged and the configured range can be interpolated.
    """

    FORMAT_VERSION = 1

    def __init__(self, size: int = 256, bias: int = 50, input_min: int = -50,
                 input_max: int = 200, lr: float = 0.1, _ptr=None):
        self._ptr = None
        if _ptr is not None:
            self._ptr = _ptr
            return
        size = int(size)
        bias = int(bias)
        input_min = int(input_min)
        input_max = int(input_max)
        lr = float(lr)
        if size <= 0 or size & (size - 1):
            raise ValueError("size must be a positive power of two")
        if size > 1 << 26:
            raise ValueError("size is unreasonably large")
        if input_max < input_min:
            raise ValueError("input_max must be >= input_min")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive and finite")
        ptr = _lib.an_layer_create(size, bias, input_min, input_max, lr)
        if not ptr:
            raise MemoryError("an_layer_create failed")
        self._ptr = ptr

    def _require_open(self) -> None:
        if not self._ptr:
            raise RuntimeError("this AdderNetLayer has been closed")

    def close(self) -> None:
        if self._ptr:
            _lib.an_layer_free(self._ptr)
            self._ptr = None

    def __enter__(self) -> "AdderNetLayer":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _validate_epochs(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
        if value > np.iinfo(np.int32).max:
            raise ValueError(f"{name} is too large")
        return value

    def train(self, inputs, targets, epochs_raw: int = 1000,
              epochs_expanded: int = 4000) -> "AdderNetLayer":
        """Run the legacy iterative optimizer and return ``self``."""
        self._require_open()
        x = _as_vector(inputs, "inputs")
        y = _as_vector(targets, "targets")
        if x.size != y.size:
            raise ValueError("inputs and targets must have the same length")
        raw = self._validate_epochs(epochs_raw, "epochs_raw")
        expanded = self._validate_epochs(epochs_expanded, "epochs_expanded")
        ret = _lib.an_train(
            self._ptr,
            x.ctypes.data_as(_DoublePtr),
            y.ctypes.data_as(_DoublePtr),
            x.size,
            raw,
            expanded,
        )
        if ret != 0:
            raise RuntimeError(f"an_train failed with status {ret}")
        return self

    def fit(self, inputs, targets, *, interpolate: bool = True,
            blend: float = 1.0) -> "AdderNetLayer":
        """Fit the LUT directly without epoch loops.

        ``blend=1`` replaces observed/interpolated cells. Smaller values update
        the existing LUT incrementally, which is useful for streaming batches.
        """
        self._require_open()
        x = _as_vector(inputs, "inputs")
        y = _as_vector(targets, "targets")
        if x.size != y.size:
            raise ValueError("inputs and targets must have the same length")
        blend = float(blend)
        if not np.isfinite(blend) or not 0 < blend <= 1:
            raise ValueError("blend must be in (0, 1]")
        ret = _lib.an_fit_direct(
            self._ptr,
            x.ctypes.data_as(_DoublePtr),
            y.ctypes.data_as(_DoublePtr),
            x.size,
            int(bool(interpolate)),
            blend,
        )
        if ret != 0:
            raise RuntimeError(f"an_fit_direct failed with status {ret}")
        return self

    partial_fit = fit

    def predict(self, x: float) -> float:
        self._require_open()
        x = float(x)
        if not np.isfinite(x):
            raise ValueError("x must be finite")
        value = float(_lib.an_predict(self._ptr, x))
        if not np.isfinite(value):
            raise RuntimeError("native prediction failed")
        return value

    def predict_batch(self, inputs) -> np.ndarray:
        self._require_open()
        x = _as_vector(inputs, "inputs", allow_empty=True)
        if x.size == 0:
            return np.empty(0, dtype=np.float64)
        out = np.empty(x.size, dtype=np.float64)
        ret = _lib.an_predict_batch(
            self._ptr,
            x.ctypes.data_as(_DoublePtr),
            out.ctypes.data_as(_DoublePtr),
            x.size,
        )
        if ret != 0 or not np.all(np.isfinite(out)):
            raise RuntimeError(f"an_predict_batch failed with status {ret}")
        return out

    def set_offset_table(self, values) -> "AdderNetLayer":
        self._require_open()
        table = _as_vector(values, "values")
        if table.size != self.size:
            raise ValueError(f"values must contain exactly {self.size} entries")
        ret = _lib.an_set_offset(self._ptr, table.ctypes.data_as(_DoublePtr), table.size)
        if ret != 0:
            raise RuntimeError(f"an_set_offset failed with status {ret}")
        return self

    @property
    def offset_table(self) -> np.ndarray:
        self._require_open()
        table = np.empty(self.size, dtype=np.float64)
        ret = _lib.an_get_offset(self._ptr, table.ctypes.data_as(_DoublePtr), table.size)
        if ret != 0:
            raise RuntimeError(f"an_get_offset failed with status {ret}")
        return table

    def save(self, path) -> None:
        """Save the compact legacy native binary, atomically where possible."""
        self._require_open()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        ret = _lib.an_save(self._ptr, os.fsencode(temp))
        if ret != 0:
            temp.unlink(missing_ok=True)
            raise OSError(f"an_save failed: {path}")
        temp.replace(path)

    @classmethod
    def load(cls, path) -> "AdderNetLayer":
        path = Path(path)
        ptr = _lib.an_load(os.fsencode(path))
        if not ptr:
            raise OSError(f"invalid or unreadable AdderNet model: {path}")
        return cls(_ptr=ptr)

    def save_portable(self, path) -> None:
        """Save a cross-platform, versioned NumPy archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "addernet.scalar",
            "format_version": self.FORMAT_VERSION,
            "size": self.size,
            "bias": self.bias,
            "input_min": self.input_min,
            "input_max": self.input_max,
            "lr": self.lr,
        }
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as fh:
            np.savez_compressed(fh, metadata=json.dumps(metadata), table=self.offset_table)
        temp.replace(path)

    @classmethod
    def load_portable(cls, path) -> "AdderNetLayer":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format") != "addernet.scalar" or metadata.get("format_version") != 1:
                raise ValueError("unsupported AdderNet portable model format")
            table = np.asarray(archive["table"], dtype=np.float64)
        obj = cls(
            metadata["size"], metadata["bias"], metadata["input_min"],
            metadata["input_max"], metadata["lr"],
        )
        return obj.set_offset_table(table)

    def get_params(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "bias": self.bias,
            "input_min": self.input_min,
            "input_max": self.input_max,
            "lr": self.lr,
        }

    @property
    def size(self) -> int:
        self._require_open()
        return int(_lib.an_get_size(self._ptr))

    @property
    def bias(self) -> int:
        self._require_open()
        return int(_lib.an_get_bias(self._ptr))

    @property
    def input_min(self) -> int:
        self._require_open()
        return int(_lib.an_get_input_min(self._ptr))

    @property
    def input_max(self) -> int:
        self._require_open()
        return int(_lib.an_get_input_max(self._ptr))

    @property
    def lr(self) -> float:
        self._require_open()
        return float(_lib.an_get_lr(self._ptr))

    def __repr__(self) -> str:
        if not self._ptr:
            return "AdderNetLayer(closed=True)"
        return (
            f"AdderNetLayer(size={self.size}, bias={self.bias}, "
            f"range=[{self.input_min}, {self.input_max}], lr={self.lr})"
        )
