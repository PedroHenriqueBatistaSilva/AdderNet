"""AdderNet public API with guarded native-library loading."""
from __future__ import annotations

import os
import shutil as _shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ADDERNET_VERBOSE = os.environ.get("ADDERNET_VERBOSE", "1") == "1"


def set_verbose(enabled: bool) -> None:
    global _ADDERNET_VERBOSE
    _ADDERNET_VERBOSE = bool(enabled)
    os.environ["ADDERNET_VERBOSE"] = "1" if enabled else "0"


def is_verbose() -> bool:
    return _ADDERNET_VERBOSE


def _native_names() -> tuple[str, str]:
    if sys.platform == "darwin":
        return "libaddernet.dylib", "libaddernet_hdc.dylib"
    if os.name == "nt":
        return "addernet.dll", "addernet_hdc.dll"
    return "libaddernet.so", "libaddernet_hdc.so"


def _copy_existing_native_files() -> None:
    layer_name, hdc_name = _native_names()
    search_dirs = (_HERE.parent / "build", _HERE / "build", Path("/usr/local/lib"), Path("/usr/lib"))
    for name in (layer_name, hdc_name):
        destination = _HERE / name
        if destination.exists():
            continue
        for directory in search_dirs:
            source = directory / name
            if source.exists():
                try:
                    _shutil.copy2(source, destination)
                except OSError:
                    pass
                break


def _ensure_native_files() -> None:
    layer_name, hdc_name = _native_names()
    _copy_existing_native_files()
    missing = [name for name in (layer_name, hdc_name) if not (_HERE / name).exists()]
    if not missing:
        return

    if os.environ.get("ADDERNET_AUTOBUILD", "1") == "0":
        raise OSError(
            f"Missing native AdderNet libraries: {', '.join(missing)}. "
            "Run `addernet-build` or reinstall a compatible wheel."
        )

    if is_verbose():
        print("[AdderNet] Native CPU libraries not found; compiling locally...")
    try:
        from .build_ext import build
        build(_HERE)
    except Exception as exc:
        raise OSError(
            "AdderNet could not compile its native CPU libraries. Install a C "
            "compiler (gcc/clang), or use a prebuilt wheel. Set "
            "ADDERNET_AUTOBUILD=0 to disable automatic compilation."
        ) from exc

    missing = [name for name in (layer_name, hdc_name) if not (_HERE / name).exists()]
    if missing:
        raise OSError(f"Native build completed without producing: {', '.join(missing)}")


_ensure_native_files()

from .addernet import AdderNetLayer
from .addernet_hdc import AdderNetHDC, hdc_detect_backend
from .attention import AdderAttention
from .boost import AdderBoost
from .cluster import AdderCluster
from .reference import ReferenceAdderNetLayer, UniformQuantizer
from .vector import AdderNetAdditiveRegressor


def get_cuda_info():
    """Return CUDA detector information without forcing CUDA at import time."""
    try:
        from .cuda_detector import CUDADetector
        detector = CUDADetector()
        detector.detect()
        return detector.to_dict()
    except Exception:
        return None


AnHdcModel = AdderNetHDC
__version__ = "1.5.0"
__all__ = [
    "AdderNetLayer", "AdderNetHDC", "AnHdcModel", "hdc_detect_backend",
    "AdderCluster", "AdderBoost", "AdderAttention", "ReferenceAdderNetLayer",
    "UniformQuantizer", "AdderNetAdditiveRegressor", "get_cuda_info",
    "set_verbose", "is_verbose",
]
