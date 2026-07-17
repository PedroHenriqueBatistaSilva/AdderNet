#!/usr/bin/env python3
"""
AdderNet-HDC Python Bindings — ctypes interface to libaddernet_hdc.so
======================================================================

Usage:
    from addernet_hdc import AdderNetHDC

    model = AdderNetHDC(n_vars=4, n_classes=3, table_size=256)
    model.train(X, y)
    pred = model.predict(x)
"""

import os
import sys
import ctypes
import numpy as np

# ---- Verbose flag control (imported from parent) ----
def _log(msg: str):
    """Print message only if verbose mode is enabled."""
    # Try to import from parent package, fallback to env var
    try:
        from . import is_verbose
        if is_verbose():
            print(msg)
    except Exception:
        if os.environ.get("ADDERNET_VERBOSE", "1") == "1":
            print(msg)

# ---- Locate shared library ----

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(_HERE, "..", "build")
if sys.platform == "darwin":
    _hdc_names = ["libaddernet_hdc.dylib"]
    _cuda_names = ["libaddernet_cuda.dylib"]
elif os.name == "nt":
    _hdc_names = ["addernet_hdc.dll"]
    _cuda_names = ["addernet_cuda.dll"]
else:
    _hdc_names = ["libaddernet_hdc.so"]
    _cuda_names = ["libaddernet_cuda.so"]
_LIB_NAMES = ([os.path.join(_BUILD, name) for name in _hdc_names] +
              [os.path.join(_HERE, name) for name in _hdc_names] + _hdc_names)

_CUDA_LIB_NAMES = ([os.path.join(_HERE, name) for name in _cuda_names] +
                   [os.path.join(_BUILD, name) for name in _cuda_names] + _cuda_names)

_lib = None
for _name in _LIB_NAMES:
    try:
        _lib = ctypes.CDLL(_name)
        break
    except OSError:
        continue

if _lib is None:
    raise OSError(
        "Cannot find libaddernet_hdc.so. "
        "Build it first: cd addernet_lib && make hdc"
    )

# ---- Optional CUDA Native library ----

import subprocess as _subprocess
from shutil import which as _which


def _find_sources():
    """Locate the C/CUDA source tree.

    Handles:
      1. site-packages install: addernet/src/ (bundled during build)
      2. repo checkout: ../src/
      3. two levels up: ../../src/
    Returns the src/ directory path or None."""
    _paths = [
        os.path.join(_HERE, "src"),          # site-packages: addernet/src/
        os.path.join(os.path.dirname(_HERE), "src"),  # repo: ../src/
        os.path.join(os.path.dirname(_HERE), "..", "src"),  # ../../src/
    ]
    for _p in _paths:
        if os.path.isdir(_p) and os.path.isfile(os.path.join(_p, "hdc_core.h")):
            return _p
    return None


def _try_build_cuda():
    """Compile libaddernet_cuda.so directly with nvcc+gcc.

    Called when no pre-built CUDA .so is found but nvcc exists on PATH.
    Typical Colab scenario: CUDA toolkit installed after pip install."""
    _nvcc = _which("nvcc") or _which("nvcc.bin")
    if not _nvcc:
        _log("[AdderNet] CUDA: nvcc not found on PATH")
        return False

    _log(f"[AdderNet] CUDA: found nvcc={_nvcc}")

    _src_dir = _find_sources()
    if _src_dir is None:
        _log("[AdderNet] CUDA: source directory not found")
        return False
    _log(f"[AdderNet] CUDA: sources at {_src_dir}")

    cu_src = os.path.join(_src_dir, "addernet_cuda.cu")
    cu_batch_train = os.path.join(_src_dir, "addernet_hdc_train_cuda.cu")
    hdc_core = os.path.join(_src_dir, "hdc_core.c")
    hdc_lsh  = os.path.join(_src_dir, "hdc_lsh.c")
    hdc      = os.path.join(_src_dir, "addernet_hdc.c")

    if not os.path.isfile(cu_src) or not os.path.isfile(hdc_core):
        _log(f"[AdderNet] CUDA: critical files missing")
        _log(f"[AdderNet] CUDA: cu_src={os.path.isfile(cu_src)}, hdc_core={os.path.isfile(hdc_core)}")
        return False

    _log("[AdderNet] Auto-compiling CUDA library...")
    import tempfile as _tmp
    _tmpdir = _tmp.mkdtemp(prefix="addernet_cuda_")
    _out_so = os.path.join(_HERE, "libaddernet_cuda.so")

    def _compile_obj(src, compiler="gcc", flags=None):
        _is_nvcc = "nvcc" in compiler
        _f = ["-O3"]
        if _is_nvcc:
            _f += ["-Xcompiler", "-fPIC", "-Xcompiler", "-ffast-math",
                    "-Xcompiler", "-fopenmp"]
        else:
            _f += ["-fPIC", "-ffast-math", "-fopenmp", "-Wno-error"]
        if flags:
            _f.extend(flags)
        _obj = os.path.join(_tmpdir, os.path.basename(src) + ".o")
        _r = _subprocess.run(
            [compiler] + _f + ["-c", src, "-o", _obj, "-I", _src_dir],
            capture_output=True
        )
        if _r.returncode != 0:
            _err = _r.stderr.decode('utf-8', errors='replace')
            _log(f"[AdderNet] CUDA: compile error ({compiler} {os.path.basename(src)}): {_err[:500]}")
            raise _subprocess.CalledProcessError(_r.returncode, compiler)
        return _obj

    _objs = []
    for _src in (hdc_core, hdc_lsh, hdc):
        if os.path.isfile(_src):
            try:
                _objs.append(_compile_obj(_src))
                _log(f"[AdderNet] CUDA: compiled {os.path.basename(_src)}")
            except _subprocess.CalledProcessError:
                pass

    if not _objs:
        _log("[AdderNet] CUDA: no CPU objects compiled")
        return False

    try:
        _log(f"[AdderNet] CUDA: compiling addernet_cuda.cu with nvcc...")
        _cuda_obj = _compile_obj(cu_src, compiler=_nvcc)
        _objs.append(_cuda_obj)
        _log(f"[AdderNet] CUDA: compiled addernet_cuda.cu")
    except _subprocess.CalledProcessError:
        return False

    try:
        _log(f"[AdderNet] CUDA: compiling addernet_hdc_train_cuda.cu with nvcc...")
        _train_obj = _compile_obj(cu_batch_train, compiler=_nvcc)
        _objs.append(_train_obj)
        _log(f"[AdderNet] CUDA: compiled addernet_hdc_train_cuda.cu")
    except _subprocess.CalledProcessError:
        return False

    _log(f"[AdderNet] CUDA: linking {_out_so} with {len(_objs)} objects...")

    try:
        _r = _subprocess.run(
            [_nvcc, "-shared", "-o", _out_so] + _objs +
            ["-lm", "-lpthread", "-fopenmp", "-ldl"],
            capture_output=True
        )
        if _r.returncode != 0:
            _err = _r.stderr.decode('utf-8', errors='replace')
            _log(f"[AdderNet] CUDA: link error: {_err[:500]}")
            return False
    except _subprocess.CalledProcessError as e:
        _log(f"[AdderNet] CUDA: link exception: {e}")
        return False

    if not os.path.isfile(_out_so):
        _log(f"[AdderNet] CUDA: link completed but {_out_so} does not exist")
        return False

    _log(f"[AdderNet] CUDA library compiled → {_out_so}")

    # Load it immediately
    global _lib_cuda, _LIB_CUDA_READY
    try:
        _lib_cuda = ctypes.CDLL(_out_so)
        _log(f"[AdderNet] CUDA library loaded!")
        _LIB_CUDA_READY = True
    except OSError as e:
        _log(f"[AdderNet] CUDA: compiled but failed to load: {e}")
        return False
    return True


# ---- CUDA 2026: Capability-based kernel selection (lazy) ----
_cuda_detector = None
_capability_int = None
_kernel_variant = 'legacy'

def _ensure_cuda_detected():
    """Lazy CUDA detection, only when needed and if verbose is enabled."""
    global _cuda_detector, _capability_int, _kernel_variant
    if _cuda_detector is not None:
        return
    try:
        from .cuda_detector import CUDADetector
        _cuda_detector = CUDADetector()
        _cuda_detector.detect()
        _capability_int = _cuda_detector.get_capability_int()
        _kernel_variant = _cuda_detector.get_best_kernel_variant()
        _log(f"[AdderNet 2026] Detected: {_kernel_variant} (capability={_capability_int})")
    except Exception as e:
        _cuda_detector = None
        _capability_int = None
        _kernel_variant = 'legacy'

# ---- Library loading with variant support (lazy) ----
_lib_cuda = None
_lib_cuda_2026 = None
_LIB_CUDA_READY = False
_CUDA_VARIANT = None
_cuda_loading_attempted = False

def _ensure_cuda_loaded():
    """Lazy loading of CUDA libraries and detection."""
    global _lib_cuda, _lib_cuda_2026, _LIB_CUDA_READY, _CUDA_VARIANT, _cuda_loading_attempted
    if _cuda_loading_attempted:
        return
    _cuda_loading_attempted = True

    # Ensure CUDA detection runs before checking variant
    _ensure_cuda_detected()

    # Try 2026 cooperative kernel (ampere/turing specific retrain kernel)
    # Also accept 'legacy' if the 2026 .so file exists — it compiles for sm_75+sm_80
    if _kernel_variant in ['ampere', 'turing']:
        _CUDA_2026_NAMES = [
            os.path.join(_HERE, f"libaddernet_cuda_{_kernel_variant}.so"),
            os.path.join(_HERE, "libaddernet_cuda_2026.so"),
        ]
        for _cuda_name in _CUDA_2026_NAMES:
            if os.path.exists(_cuda_name):
                try:
                    _lib_cuda_2026 = ctypes.CDLL(_cuda_name)
                    _log(f"[AdderNet 2026] Loaded cooperative {_kernel_variant} kernel from {_cuda_name}")
                    break
                except OSError:
                    pass

    # If 2026 kernel not loaded but file exists, try loading it anyway (legacy fallback)
    if _lib_cuda_2026 is None:
        _generic_2026 = os.path.join(_HERE, "libaddernet_cuda_2026.so")
        if os.path.exists(_generic_2026):
            try:
                _lib_cuda_2026 = ctypes.CDLL(_generic_2026)
                _log(f"[AdderNet 2026] Loaded generic 2026 kernel (legacy mode) from {_generic_2026}")
            except OSError:
                pass

    # Load generic CUDA (always needed for inference)
    # Fallback to generic CUDA
    if not _LIB_CUDA_READY:
        for _cuda_name in _CUDA_LIB_NAMES:
            if os.path.exists(_cuda_name):
                try:
                    _lib_cuda = ctypes.CDLL(_cuda_name)
                    _log(f"[AdderNet] CUDA library loaded from {_cuda_name}")
                    _LIB_CUDA_READY = True
                    _CUDA_VARIANT = 'generic'
                    break
                except OSError:
                    pass

    # Try building if not found
    if not _LIB_CUDA_READY:
        if _try_build_cuda():
            for _cuda_name in _CUDA_LIB_NAMES:
                if os.path.exists(_cuda_name):
                    try:
                        _lib_cuda = ctypes.CDLL(_cuda_name)
                        _log(f"[AdderNet] Auto-compiled CUDA library loaded from {_cuda_name}")
                        _LIB_CUDA_READY = True
                        _CUDA_VARIANT = 'generic'
                        break
                    except OSError:
                        pass

# ---- Opaque pointer type ----

_AnHdcPtr = ctypes.c_void_p

# ---- Function signatures ----

_lib.an_hdc_create.restype = _AnHdcPtr
_lib.an_hdc_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int
]

_lib.an_hdc_free.restype = None
_lib.an_hdc_free.argtypes = [_AnHdcPtr]

_lib.an_hdc_train.restype = None
_lib.an_hdc_train.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.an_hdc_retrain.restype = ctypes.c_int
_lib.an_hdc_retrain.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.an_hdc_predict.restype = ctypes.c_int
_lib.an_hdc_predict.argtypes = [_AnHdcPtr, ctypes.POINTER(ctypes.c_double)]

_lib.an_hdc_predict_batch.restype = ctypes.c_int
_lib.an_hdc_predict_batch.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

# Melhoria 4: AVX2 batch prediction
_lib.an_hdc_predict_batch_avx.restype = ctypes.c_int
_lib.an_hdc_predict_batch_avx.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

if _lib_cuda is not None:
    _lib_cuda.an_hdc_retrain_cuda.restype = ctypes.c_int
    _lib_cuda.an_hdc_retrain_cuda.argtypes = [
        _AnHdcPtr,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]

    _lib_cuda.an_hdc_predict_batch_cuda.restype = ctypes.c_int
    _lib_cuda.an_hdc_predict_batch_cuda.argtypes = [
        _AnHdcPtr,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

_lib.an_hdc_save.restype = ctypes.c_int
_lib.an_hdc_save.argtypes = [_AnHdcPtr, ctypes.c_char_p]

_lib.an_hdc_load.restype = _AnHdcPtr
_lib.an_hdc_load.argtypes = [ctypes.c_char_p]

for _getter in ("an_hdc_get_n_vars", "an_hdc_get_n_classes", "an_hdc_get_table_size",
                "an_hdc_get_hv_dim", "an_hdc_get_hv_words"):
    getattr(_lib, _getter).restype = ctypes.c_int
    getattr(_lib, _getter).argtypes = [_AnHdcPtr]
_lib.an_hdc_get_codebook.restype = ctypes.c_int
_lib.an_hdc_get_codebook.argtypes = [
    _AnHdcPtr, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int
]

# OPT-1: Cache functions
_lib.an_hdc_warm_cache.restype = None
_lib.an_hdc_warm_cache.argtypes = [_AnHdcPtr]

_lib.an_hdc_set_cache.restype = None
_lib.an_hdc_set_cache.argtypes = [_AnHdcPtr, ctypes.c_int]

# OPT-5: Multithreaded batch prediction
_lib.an_hdc_predict_batch_mt.restype = ctypes.c_int
_lib.an_hdc_predict_batch_mt.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]

_lib.an_hdc_set_threads.restype = None
_lib.an_hdc_set_threads.argtypes = [_AnHdcPtr, ctypes.c_int]

# Melhoria 3: Hadamard encoding
_lib.an_hdc_set_hadamard.restype = None
_lib.an_hdc_set_hadamard.argtypes = [_AnHdcPtr, ctypes.c_int]

# LSH: Locality-Sensitive Hashing
_lib.an_hdc_build_lsh.restype = None
_lib.an_hdc_build_lsh.argtypes = [_AnHdcPtr]

_lib.an_hdc_build_lsh_ex.restype = None
_lib.an_hdc_build_lsh_ex.argtypes = [_AnHdcPtr, ctypes.c_int, ctypes.c_int]

_lib.an_hdc_set_lsh.restype = None
_lib.an_hdc_set_lsh.argtypes = [_AnHdcPtr, ctypes.c_int]

# predict_top_k
_lib.an_hdc_predict_top_k.restype = None
_lib.an_hdc_predict_top_k.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

# Problem 6: Interaction encoding
_lib.an_hdc_set_interactions.restype = None
_lib.an_hdc_set_interactions.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

# OPT-8: Backend detection
_lib.hdc_detect_backend.restype = ctypes.c_int
_lib.hdc_detect_backend.argtypes = []

_lib.hv_seed.restype = None
_lib.hv_seed.argtypes = [ctypes.c_uint]

# Hypervector size is model-specific; use ``model.hv_dim`` and
# ``model.hv_words`` instead of module-level fixed constants.


# ---- Python wrapper ----

class AdderNetHDC:
    """
    Multivariate classifier using AdderNet encoding + Hyperdimensional Computing.
    Zero floating-point multiplication at inference.
    """

    def __init__(self, n_vars=1, n_classes=2, table_size=256, bias=None,
                 seed=42, use_gpu=False, hv_dim=2500, use_gpu_training=False, _ptr=None):
        """
        Create a new model.

        Args:
            n_vars:     number of input variables
            n_classes:  number of output classes
            table_size: encoding table size per variable (power of 2)
            bias:       list of bias values per variable (default: table_size//2)
            seed:       random seed for reproducibility
            use_gpu:    toggle between CPU and CUDA backend
        """
        self.use_gpu = bool(use_gpu)
        self.use_gpu_training = bool(use_gpu_training)

        if _ptr is not None:
            self._ptr = _ptr
            self._n_vars = _lib.an_hdc_get_n_vars(_ptr)
            self._n_classes = _lib.an_hdc_get_n_classes(_ptr)
            self._table_size = _lib.an_hdc_get_table_size(_ptr)
            self.hv_dim = _lib.an_hdc_get_hv_dim(_ptr)
            self.hv_words = _lib.an_hdc_get_hv_words(_ptr)
            if min(self._n_vars, self._n_classes, self._table_size, self.hv_dim, self.hv_words) <= 0:
                raise RuntimeError("loaded HDC model has invalid metadata")
            return

        if n_vars <= 0 or n_classes <= 0 or hv_dim <= 0:
            raise ValueError("n_vars, n_classes and hv_dim must be positive")
        if table_size <= 0 or table_size & (table_size - 1):
            raise ValueError("table_size must be a positive power of two")
        if bias is not None and len(bias) != n_vars:
            raise ValueError("bias must contain one value per input variable")
        self.hv_dim = int(hv_dim)
        self.hv_words = (self.hv_dim + 63) // 64

        _lib.hv_seed(seed)

        bias_arr = None
        if bias is not None:
            bias_arr = (ctypes.c_int * n_vars)(*bias)

        self._ptr = _lib.an_hdc_create(n_vars, n_classes, table_size, bias_arr, hv_dim)
        if not self._ptr:
            raise MemoryError("an_hdc_create failed")
        self._n_vars = n_vars
        self._n_classes = n_classes
        self._table_size = table_size

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.an_hdc_free(self._ptr)
            self._ptr = None

    def _validate_X(self, X, *, allow_empty=False):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            if X.size == self._n_vars:
                X = X.reshape(1, self._n_vars)
            elif X.size % self._n_vars == 0:
                X = X.reshape(-1, self._n_vars)
            else:
                raise ValueError(f"input length must be a multiple of n_vars={self._n_vars}")
        if X.ndim != 2 or X.shape[1] != self._n_vars:
            raise ValueError(f"X must have shape (n_samples, {self._n_vars})")
        if not allow_empty and X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def train(self, X, y, n_iter=0, lr=1.0, margin=0, regenerate=0.0,
              patience=10, verbose=False, interactions=0):
        """
        Train the codebook from labeled data, with optional iterative retraining.

        Uses OnlineHD weighted bundling by default to prevent model saturation.
        When n_iter > 0, applies RefineHD iterative correction with additive margin
        and NeuralHD dimension regeneration.

        Args:
            X: n_samples × n_vars array (list of lists or numpy array)
            y: n_samples class labels (int, 0-indexed)
            n_iter: number of iterative correction passes (0 = single-pass, default)
            lr: learning rate for iterative retraining (default 1.0)
            margin: RefineHD margin — aceita 4 formas:
                    0 / None  → desligado (AdaptHD puro)
                    float 0-1 → fração de D (ex: 0.05 = 5% de D)
                    '5%'      → percentual string (ex: '5%', '10%')
                    int > 0   → distância Hamming absoluta em bits
            regenerate: NeuralHD dimension regeneration rate (0.0 = off, 0.02-0.05 recommended)
            patience: early stopping patience in epochs (0 = disabled, 5 recommended)
            verbose: if True, print epoch progress to stderr every 10 epochs. If int, print every N epochs.
            interactions: number of top correlated feature pairs to encode (0 = disabled, 10 recommended)
        Returns:
            dict with training history:
                'epochs_run': epochs actually executed,
                'best_val_accuracy': best validation accuracy (last 25% of data),
                'best_train_accuracy': best training accuracy (first 75% of data),
                'best_epoch': epoch of best val accuracy,
                'stopped_early': True if early stopping triggered
        """
        X = self._validate_X(X)
        y = np.ascontiguousarray(y, dtype=np.int32).reshape(-1)

        n = X.shape[0]
        if len(y) != n:
            raise ValueError(f"X has {n} samples but y has {len(y)}")
        if np.any(y < 0) or np.any(y >= self._n_classes):
            raise ValueError(f"class labels must be in [0, {self._n_classes - 1}]")

        # Default history for single-pass training
        history = {
            'epochs_run': 0,
            'best_val_accuracy': 0.0,
            'best_train_accuracy': 0.0,
            'best_epoch': 0,
            'stopped_early': False,
        }

        # Problema 6: Detect and set interaction pairs
        if interactions > 0 and X.shape[1] > 1:
            corr = np.corrcoef(X.T)
            pairs = []
            for ii in range(corr.shape[0]):
                for jj in range(ii + 1, corr.shape[1]):
                    pairs.append((abs(corr[ii, jj]), ii, jj))
            pairs.sort(reverse=True)
            top_pairs = pairs[:interactions]
            if top_pairs:
                pairs_i = np.array([p[1] for p in top_pairs], dtype=np.int32)
                pairs_j = np.array([p[2] for p in top_pairs], dtype=np.int32)
                _lib.an_hdc_set_interactions(
                    self._ptr,
                    pairs_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    pairs_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    len(top_pairs),
                )

        # Initial train (encodes samples including interactions)
        _lib.an_hdc_train(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
        )

        if n_iter > 0:
            D = self.hv_dim

            # ── Conversão de margin ──────────────────────────────
            if margin == 0 or margin is None:
                margin_int = 0          # desligado — AdaptHD puro

            elif isinstance(margin, str) and margin.endswith('%'):
                pct = float(margin[:-1]) / 100.0
                margin_int = max(1, int(pct * D))

            elif isinstance(margin, float) and 0.0 < margin < 1.0:
                margin_int = max(1, int(margin * D))

            elif isinstance(margin, int) and margin > 0:
                margin_int = margin

            else:
                raise ValueError(
                    f"margin deve ser: 0 (off), float 0-1 (fração de D), "
                    f"'5%' (percentual), ou int (bits). Recebeu: {margin!r}"
                )

            # Clamp: nunca maior que 20% de D
            margin_int = min(margin_int, int(D * 0.20))

            if verbose:
                if margin_int == 0:
                    print(f"  margin=off (AdaptHD puro)")
                else:
                    print(f"  margin={margin!r} → {margin_int} bits ({margin_int/D*100:.1f}% de D={D})")

            # Verbose level: True → 10, int → as-is, False → 0
            if verbose is True:
                verbose_level = 10
            elif isinstance(verbose, int) and verbose > 0:
                verbose_level = verbose
            else:
                verbose_level = 0

            epochs_run = ctypes.c_int(0)

            # GPU training: variant selection based on capability
            _used_gpu = False
            if self.use_gpu_training:
                _ensure_cuda_loaded()
                _selected_kernel = None

                # Try 2026 cooperative kernel (ampere/turing)
                if _kernel_variant in ('ampere', 'turing') and _lib_cuda_2026 is not None:
                    _selected_kernel = ('2026', _lib_cuda_2026)
                # Fallback: use 2026 kernel even in legacy mode if loaded
                elif _lib_cuda_2026 is not None:
                    _selected_kernel = ('2026_legacy', _lib_cuda_2026)
                # Last fallback: generic CUDA retrain
                elif _lib_cuda is not None and hasattr(_lib_cuda, 'an_hdc_retrain_cuda'):
                    _selected_kernel = ('generic', _lib_cuda)

                if _selected_kernel is not None:
                    variant_name, cuda_lib = _selected_kernel
                    if verbose_level > 0:
                        _log(f"[AdderNet] GPU training active: kernel={variant_name}")
                    cuda_lib.an_hdc_retrain_cuda(
                        self._ptr,
                        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        n,
                        n_iter,
                        ctypes.c_float(lr),
                        ctypes.c_int(margin_int),
                        ctypes.c_int(patience),
                        ctypes.c_int(verbose_level),
                        ctypes.byref(epochs_run),
                    )
                    _used_gpu = True
                else:
                    _log("[AdderNet] Warning: use_gpu_training=True but "
                          "no CUDA training kernel found. Falling back to CPU.")

            if not _used_gpu:
                _lib.an_hdc_retrain(
                    self._ptr,
                    X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    n,
                    n_iter,
                    ctypes.c_float(lr),
                    ctypes.c_int(margin_int),
                    ctypes.c_float(regenerate),
                    ctypes.c_int(patience),
                    ctypes.c_int(verbose_level),
                    ctypes.byref(epochs_run),
                )

            # Compute final accuracies on train/val split
            n_val = max(1, n // 4)
            n_train_split = n - n_val
            train_acc = float(self.accuracy(X[:n_train_split], y[:n_train_split]))
            val_acc = float(self.accuracy(X[n_train_split:], y[n_train_split:]))

            history['epochs_run'] = epochs_run.value
            history['best_val_accuracy'] = val_acc
            history['best_train_accuracy'] = train_acc
            history['best_epoch'] = epochs_run.value
            history['stopped_early'] = (epochs_run.value < n_iter)

        return history

    def predict(self, x):
        """
        Classify one sample.

        Args:
            x: list or array of n_vars input values
        Returns:
            predicted class label (int)
        """
        x = self._validate_X(x)[0]
        return _lib.an_hdc_predict(
            self._ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def predict_batch(self, X):
        """
        Classify multiple samples.

        Args:
            X: n_samples × n_vars array
        Returns:
            numpy array of predicted class labels
        """
        X = self._validate_X(X)
        n = X.shape[0]
        outputs = np.empty(n, dtype=np.int32)
        
        if getattr(self, 'use_gpu', False):
            if _lib_cuda is None:
                raise RuntimeError("CUDA backend requested but libaddernet_cuda.so not found")
            _lib_cuda.an_hdc_predict_batch_cuda(
                self._ptr,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                n,
            )
        else:
            ret = _lib.an_hdc_predict_batch(
                self._ptr,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                n,
            )
            if ret != 0:
                raise RuntimeError("an_hdc_predict_batch failed")
        return outputs

    def predict_batch_avx(self, X):
        """
        Classify multiple samples using AVX2 SIMD (Melhoria 4).
        Processes 4 samples simultaneously for faster batch inference.

        Args:
            X: n_samples × n_vars array
        Returns:
            numpy array of predicted class labels
        """
        X = self._validate_X(X)
        n = X.shape[0]
        outputs = np.empty(n, dtype=np.int32)
        ret = _lib.an_hdc_predict_batch_avx(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
        )
        if ret != 0:
            raise RuntimeError("an_hdc_predict_batch_avx failed")
        return outputs

    def predict_batch_mt(self, X, n_threads=0):
        """
        Classify multiple samples using multiple threads (OPT-5).

        Args:
            X: n_samples × n_vars array
            n_threads: number of threads (0 = auto-detect)
        Returns:
            numpy array of predicted class labels
        """
        X = self._validate_X(X)
        n = X.shape[0]
        outputs = np.empty(n, dtype=np.int32)
        ret = _lib.an_hdc_predict_batch_mt(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
            int(n_threads),
        )
        if ret != 0:
            raise RuntimeError("an_hdc_predict_batch_mt failed")
        return outputs

    def warm_cache(self):
        """Pre-compute encoding cache (OPT-1). Call once before benchmarking."""
        _lib.an_hdc_warm_cache(self._ptr)

    def set_cache(self, use_cache):
        """Enable/disable encoding cache (OPT-1)."""
        _lib.an_hdc_set_cache(self._ptr, 1 if use_cache else 0)

    def set_threads(self, n_threads):
        """Set thread count for batch prediction (OPT-5)."""
        _lib.an_hdc_set_threads(self._ptr, n_threads)

    def set_hadamard(self, enable=True):
        """Enable/disable Hadamard encoding (Melhoria 3 - orthogonal base vectors)."""
        _lib.an_hdc_set_hadamard(self._ptr, 1 if enable else 0)

    def build_lsh(self, k=10, l=8):
        """Build LSH index with K=10, L=8 tables."""
        _lib.an_hdc_build_lsh_ex(self._ptr, k, l)

    def set_lsh(self, enable):
        """Enable/disable LSH for prediction."""
        _lib.an_hdc_set_lsh(self._ptr, 1 if enable else 0)

    def predict_top_k(self, x, k=5):
        """
        Classify one sample and return top K predictions.

        Args:
            x: list or array of n_vars input values
            k: number of top predictions to return
        Returns:
            list of top K predicted class labels (int)
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        out_classes = np.empty(k, dtype=np.int32)
        _lib.an_hdc_predict_top_k(
            self._ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            k,
        )
        return out_classes.tolist()

    def accuracy(self, X, y):
        """
        Compute accuracy on given data.

        Args:
            X: n_samples × n_vars array
            y: true class labels
        Returns:
            accuracy as float
        """
        y_pred = self.predict_batch(X)
        y = np.ascontiguousarray(y, dtype=np.int32)
        return np.mean(y_pred == y)

    def add_noise(self, hv, temperature):
        """Return a copy with each valid bit flipped independently."""
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        src = np.ascontiguousarray(hv, dtype=np.uint64).reshape(-1)
        if len(src) != self.hv_words:
            raise ValueError(f"hypervector must contain {self.hv_words} uint64 words")
        rng = np.random.default_rng()
        dst = src.copy()
        for bit in np.flatnonzero(rng.random(self.hv_dim) < temperature):
            dst[bit // 64] ^= np.uint64(1) << np.uint64(bit % 64)
        if self.hv_dim % 64:
            dst[-1] &= np.uint64((1 << (self.hv_dim % 64)) - 1)
        return dst

    def bundle_classes(self, class_indices):
        """Majority-bundle selected class prototypes."""
        indices = np.asarray(class_indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self._n_classes):
            raise ValueError("class index out of range")
        if len(indices) == 0:
            return np.zeros(self.hv_words, dtype=np.uint64)
        selected = np.stack([self.codebook[int(i)] for i in indices])
        bytes_view = selected.view(np.uint8).reshape(len(selected), -1)
        bits = np.unpackbits(bytes_view, axis=1, bitorder="little")[:, :self.hv_dim]
        majority = bits.sum(axis=0) > (len(selected) / 2)
        packed = np.packbits(majority, bitorder="little")
        padded = np.zeros(self.hv_words * 8, dtype=np.uint8)
        padded[:len(packed)] = packed
        return padded.view(np.uint64).copy()

    def classify_hv(self, hv):
        """Classify a hypervector using true bitwise Hamming distance."""
        hv_arr = np.ascontiguousarray(hv, dtype=np.uint64).reshape(-1)
        if len(hv_arr) != self.hv_words:
            raise ValueError(f"hypervector must contain {self.hv_words} uint64 words")
        cb = self.codebook
        xor = np.bitwise_xor(cb, hv_arr[None, :])
        dist = np.unpackbits(xor.view(np.uint8), axis=1).sum(axis=1)
        return int(np.argmin(dist))

    def save(self, path):
        """Save model to binary file."""
        ret = _lib.an_hdc_save(self._ptr, path.encode("utf-8"))
        if ret != 0:
            raise IOError(f"an_hdc_save failed: {path}")

    @classmethod
    def load(cls, path):
        """Load model from binary file."""
        ptr = _lib.an_hdc_load(path.encode("utf-8"))
        if not ptr:
            raise IOError(f"an_hdc_load failed: {path}")
        return cls(_ptr=ptr)

    @property
    def n_vars(self):
        return self._n_vars

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def table_size(self):
        return self._table_size

    @property
    def codebook(self):
        """Return a copy of the trained codebook as ``(classes, words)``."""
        total = self._n_classes * self.hv_words
        flat = np.empty(total, dtype=np.uint64)
        ret = _lib.an_hdc_get_codebook(
            self._ptr,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            total,
        )
        if ret != 0:
            raise RuntimeError("failed to read HDC codebook")
        return flat.reshape(self._n_classes, self.hv_words)

    def __repr__(self):
        return (f"AdderNetHDC(n_vars={self._n_vars}, "
                f"n_classes={self._n_classes}, "
                f"table_size={self._table_size}, hv_dim={self.hv_dim})")


def hdc_detect_backend():
    """
    Detect the available backend for HDC operations.

    Returns:
        'AVX2', 'NEON', or 'SCALAR'
    """
    backends = {0: "SCALAR", 1: "AVX2", 2: "NEON"}
    return backends.get(_lib.hdc_detect_backend(), "UNKNOWN")
