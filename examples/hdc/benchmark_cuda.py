#!/usr/bin/env python3
"""
Benchmark hv_bundle — CPU vs GPU (MX250, inline PTX)
=====================================================
Compara hv_bundle (CPU) vs hv_bundle_cuda (GPU via driver API).
"""

import sys
import os
import time
import ctypes
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

# ---- Load CPU library ----
_LIB_CPU = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "..", "build", "libaddernet_hdc.so"))

# ---- Load CUDA library ----
_LIB_GPU = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "libaddernet_hdc_cuda.so"),
    mode=ctypes.RTLD_GLOBAL
)

D_VALUES = [5000]  # Test with D=5000 as specified
N_VECS = [10, 50, 100, 196]

for D in D_VALUES:
    WORDS = (D + 63) // 64
    _HV_t = ctypes.c_uint64 * WORDS

    # Setup CPU bindings
    _LIB_CPU.hv_bundle.argtypes = [_HV_t, ctypes.POINTER(_HV_t), ctypes.c_int]
    _LIB_CPU.hv_bundle.restype = None
    _LIB_CPU.hv_random.argtypes = [_HV_t]
    _LIB_CPU.hv_random.restype = None
    _LIB_CPU.hv_seed.argtypes = [ctypes.c_uint]
    _LIB_CPU.hv_seed.restype = None
    _LIB_CPU.hv_hamming.argtypes = [_HV_t, _HV_t]
    _LIB_CPU.hv_hamming.restype = ctypes.c_int

    # Setup GPU bindings
    _LIB_GPU.hv_bundle_cuda.argtypes = [_HV_t, ctypes.POINTER(_HV_t), ctypes.c_int]
    _LIB_GPU.hv_bundle_cuda.restype = None
    _LIB_GPU.hv_cuda_shutdown.argtypes = []
    _LIB_GPU.hv_cuda_shutdown.restype = None

    _LIB_CPU.hv_seed(42)

    print(f"\n{'='*60}")
    print(f"  Benchmark hv_bundle — CPU vs GPU (D={D})")
    print(f"{'='*60}")
    print(f"\n{'n_vecs':>8} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>10}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for n in N_VECS:
        # Generate random vectors
        vecs = (_HV_t * n)()
        for v in range(n):
            _LIB_CPU.hv_random(vecs[v])

        out_cpu = _HV_t()
        out_gpu = _HV_t()

        # Warmup CPU
        _LIB_CPU.hv_bundle(out_cpu, vecs, n)

        # Time CPU (10 iterations)
        N_ITERS = 10
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            _LIB_CPU.hv_bundle(out_cpu, vecs, n)
        t_cpu = (time.perf_counter() - t0) / N_ITERS

        # Warmup GPU (first call has init overhead)
        try:
            _LIB_GPU.hv_bundle_cuda(out_gpu, vecs, n)
            # Verify correctness
            d = _LIB_CPU.hv_hamming(out_cpu, out_gpu)
            correct = (d == 0)
        except Exception as e:
            print(f"{n:>8} {'ERROR: ' + str(e)[:30]:>10}")
            continue

        # Time GPU (10 iterations)
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            _LIB_GPU.hv_bundle_cuda(out_gpu, vecs, n)
        t_gpu = (time.perf_counter() - t0) / N_ITERS

        speedup = t_cpu / t_gpu if t_gpu > 0 else 0
        mark = " ✓" if correct else " ✗ MISMATCH"
        print(f"{n:>8} {t_cpu*1000:>9.2f}  {t_gpu*1000:>9.2f}  {speedup:>9.1f}x{mark}")

    _LIB_GPU.hv_cuda_shutdown()
