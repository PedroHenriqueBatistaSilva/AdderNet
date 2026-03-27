import subprocess
import os
import sys
from pathlib import Path

def build():
    pkg_dir = Path(__file__).parent
    src_dir = pkg_dir / "src"

    if sys.platform == "darwin":
        lib_name = "libaddernet_hdc.dylib"
        lib_name_addernet = "libaddernet.dylib"
    elif sys.platform == "win32":
        lib_name = "addernet_hdc.dll"
        lib_name_addernet = "addernet.dll"
    else:
        lib_name = "libaddernet_hdc.so"
        lib_name_addernet = "libaddernet.so"

    out_hdc = pkg_dir / lib_name
    out_addernet = pkg_dir / lib_name_addernet

    cmd_hdc = [
        "gcc", "-O3", "-march=native", "-fPIC", "-shared",
        "-mavx2", "-mpopcnt", "-D__AVX2__", "-fopenmp",
        str(src_dir / "addernet.c"),
        str(src_dir / "hdc_core.c"),
        str(src_dir / "hdc_lsh.c"),
        str(src_dir / "addernet_hdc.c"),
        "-o", str(out_hdc), "-lm", "-lpthread"
    ]
    subprocess.run(cmd_hdc, check=True)

    cmd_addernet = [
        "gcc", "-O3", "-march=native", "-fPIC", "-shared", "-fopenmp",
        str(src_dir / "addernet.c"),
        "-o", str(out_addernet), "-lm"
    ]
    subprocess.run(cmd_addernet, check=True)

    return str(out_hdc), str(out_addernet)

if __name__ == "__main__":
    build()
