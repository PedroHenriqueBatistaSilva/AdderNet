"""Portable native builder for AdderNet CPU libraries.

The function is deliberately dependency-free and is used both by the console
script ``addernet-build`` and by the package's guarded auto-build path.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _source_dir(pkg_dir: Path) -> Path:
    candidates = [pkg_dir / "src", pkg_dir.parent / "src"]
    for candidate in candidates:
        if (candidate / "addernet.c").exists() and (candidate / "addernet_hdc.c").exists():
            return candidate
    raise FileNotFoundError(
        "AdderNet C sources were not found. Reinstall from the source archive "
        "or a wheel that includes addernet/src/*.c and *.h."
    )


def _run(cmd: Iterable[str]) -> None:
    subprocess.run(list(cmd), check=True)


def _compiler() -> str:
    configured = os.environ.get("CC")
    if configured:
        return configured
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("No C compiler found. Install gcc/clang and retry.")


def _platform_config() -> tuple[str, list[str], list[str]]:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        suffix = ".dylib"
        shared = ["-dynamiclib"]
        link = ["-lm"]
    elif os.name == "nt":
        raise RuntimeError(
            "The automatic ctypes builder currently supports Linux and macOS. "
            "On Windows, install a prebuilt wheel or compile with a compatible toolchain."
        )
    else:
        suffix = ".so"
        shared = ["-shared"]
        link = ["-lm", "-lpthread"]

    flags = ["-O3", "-fPIC", "-Wall", "-Wextra"]
    if machine in {"x86_64", "amd64"}:
        # Build a portable x86-64 baseline by default. Native AVX2 can be
        # requested explicitly without producing illegal instructions on older CPUs.
        if os.environ.get("ADDERNET_NATIVE", "0") == "1":
            flags += ["-march=native", "-mpopcnt"]
    elif machine in {"aarch64", "arm64"}:
        flags += ["-march=armv8-a"]
    elif machine.startswith("arm"):
        flags += ["-mfpu=neon-vfpv4"]

    # OpenMP is optional and opt-in. The scalar LUT loop is memory-bound;
    # spawning a large thread pool made common batch inference dramatically slower.
    if sys.platform != "darwin" and os.environ.get("ADDERNET_OPENMP", "0") == "1":
        flags += ["-fopenmp"]
        link += ["-fopenmp"]
    return suffix, shared + flags, link


def build(output_dir: str | os.PathLike[str] | None = None) -> int:
    """Compile the CPU libraries into the Python package directory.

    Returns ``0`` so the function is safe as a console-script entry point.
    """
    pkg_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    pkg_dir.mkdir(parents=True, exist_ok=True)
    src = _source_dir(pkg_dir)
    cc = _compiler()
    suffix, compile_flags, link_flags = _platform_config()

    out_layer = pkg_dir / f"libaddernet{suffix}"
    out_hdc = pkg_dir / f"libaddernet_hdc{suffix}"

    common = [cc, *compile_flags]
    _run([*common, str(src / "addernet.c"), "-o", str(out_layer), *link_flags])
    _run([
        *common,
        str(src / "addernet.c"),
        str(src / "hdc_core.c"),
        str(src / "hdc_lsh.c"),
        str(src / "addernet_hdc.c"),
        "-o", str(out_hdc), *link_flags,
    ])

    print(out_layer)
    print(out_hdc)
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
