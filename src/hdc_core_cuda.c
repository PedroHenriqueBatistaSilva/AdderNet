/*
 * HDC Core CUDA — GPU hv_bundle via inline PTX (driver API)
 * ===========================================================
 *   No nvcc required. PTX embedded as string, loaded via cuModuleLoad.
 *   Memory via cuMemAlloc. No runtime API dependency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include "hdc_core.h"

typedef int CUresult;
typedef void *CUmodule;
typedef void *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef void *CUdevice;
#define CUDA_SUCCESS 0

static struct {
    void *h;
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuCtxCreate)(void**, unsigned, CUdevice);
    CUresult (*cuCtxGetCurrent)(void**);
    CUresult (*cuCtxSetCurrent)(void*);
    CUresult (*cuCtxDestroy)(void*);
    CUresult (*cuModuleLoad)(CUmodule*, const char*);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
    CUresult (*cuLaunchKernel)(CUfunction,
        unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
        void*, void**, void*);
    CUresult (*cuCtxSynchronize)(void);
} G;

static int inited = 0;
static CUmodule mod = NULL;
static CUfunction fn = NULL;
static void *ctx = NULL;

#define S_(x) #x
#define S(x) S_(x)

static const char ptx[] =
    ".version 6.5\n.target sm_61\n.address_size 64\n"
    ".visible .entry bundle_kernel("
    ".param .u64 param_out,"
    ".param .u64 param_vecs,"
    ".param .u32 param_n,"
    ".param .u32 param_threshold)\n"
    "{\n"
    "  .reg .u32 %r<15>; .reg .u64 %rd<12>; .reg .pred %p<8>;\n"
    "  mov.u32 %r0,%tid.x;\n"
    "  mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x;\n"
    "  mul.lo.u32 %r1,%r1,%r2; add.u32 %r0,%r0,%r1;\n"
    "  setp.ge.u32 %p0,%r0," S(HDC_WORDS) "; @%p0 ret;\n"
    "  ld.param.u64 %rd1,[param_out];\n"
    "  ld.param.u64 %rd2,[param_vecs];\n"
    "  ld.param.u32 %r3,[param_n];\n"
    "  ld.param.u32 %r4,[param_threshold];\n"
    "  mov.u64 %rd8,0; mov.u32 %r5,0;\n"
    "BIT:\n"
    "  setp.ge.u32 %p1,%r5,64; @%p1 bra ST;\n"
    "  mov.u32 %r6,0;\n"
    "  mov.u64 %rd3,1; shl.b64 %rd3,%rd3,%r5;\n"
    "  mov.u32 %r7,0;\n"
    "VL:\n"
    "  setp.ge.u32 %p2,%r7,%r3; @%p2 bra CHK;\n"
    "  mul.lo.u32 %r10,%r7," S(HDC_WORDS*8) ";\n"
    "  cvt.u64.u32 %rd4,%r10; add.u64 %rd4,%rd2,%rd4;\n"
    "  mul.lo.u32 %r11,%r0,8;\n"
    "  cvt.u64.u32 %rd5,%r11; add.u64 %rd4,%rd4,%rd5;\n"
    "  ld.global.u64 %rd6,[%rd4];\n"
    "  and.b64 %rd7,%rd6,%rd3;\n"
    "  setp.ne.u64 %p3,%rd7,0; @%p3 add.u32 %r6,%r6,1;\n"
    "  add.u32 %r7,%r7,1; bra VL;\n"
    "CHK:\n"
    "  setp.gt.u32 %p4,%r6,%r4; @%p4 or.b64 %rd8,%rd8,%rd3;\n"
    "  add.u32 %r5,%r5,1; bra BIT;\n"
    "ST:\n"
    "  mul.lo.u32 %r12,%r0,8;\n"
    "  cvt.u64.u32 %rd9,%r12; add.u64 %rd9,%rd1,%rd9;\n"
    "  st.global.u64 [%rd9],%rd8;\n"
    "  ret;\n"
    "}\n";

static int init(void) {
    if (inited) return 1;

    G.h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!G.h) G.h = dlopen("libcuda.so", RTLD_LAZY);
    if (!G.h) return 0;

    #define L(s) G.s = dlsym(G.h, #s)
    L(cuInit); L(cuDeviceGet); L(cuCtxCreate); L(cuCtxGetCurrent);
    L(cuCtxSetCurrent); L(cuCtxDestroy); L(cuModuleLoad); L(cuModuleUnload);
    L(cuModuleGetFunction); L(cuMemAlloc); L(cuMemFree);
    L(cuMemcpyHtoD); L(cuMemcpyDtoH); L(cuLaunchKernel); L(cuCtxSynchronize);
    #undef L

    if (!G.cuInit || !G.cuModuleLoad || !G.cuLaunchKernel) return 0;
    G.cuInit(0);

    /* Use existing context or create new */
    if (G.cuCtxGetCurrent) G.cuCtxGetCurrent(&ctx);
    if (!ctx) {
        CUdevice dev;
        G.cuDeviceGet(&dev, 0);
        G.cuCtxCreate(&ctx, 0, dev);
    }
    if (G.cuCtxSetCurrent) G.cuCtxSetCurrent(ctx);

    /* Write PTX to temp file and load */
    size_t plen = strlen(ptx) + 1;
    char path[] = "/tmp/hdc_bundle_XXXXXX.ptx";
    int fd = mkstemps(path, 4);
    if (fd < 0) return 0;
    write(fd, ptx, plen - 1); close(fd);
    CUresult r = G.cuModuleLoad(&mod, path);
    unlink(path);
    if (r != 0) return 0;

    r = G.cuModuleGetFunction(&fn, mod, "bundle_kernel");
    if (r != 0) return 0;

    inited = 1;
    return 1;
}

void hv_bundle_cuda(hv_t out, const hv_t *vecs, int n) {
    if (n <= 0) { hv_zero(out); return; }
    if (n == 1) { hv_copy(out, vecs[0]); return; }
    if (!init()) { hv_bundle(out, vecs, n); return; }

    size_t vb = (size_t)n * sizeof(hv_t);
    size_t ob = sizeof(hv_t);
    int thr = n / 2;

    CUdeviceptr dv, dout;
    CUresult a1 = G.cuMemAlloc(&dv, vb);
    CUresult a2 = G.cuMemAlloc(&dout, ob);
    G.cuMemcpyHtoD(dv, vecs, vb);

    void *args[] = { &dout, &dv, &n, &thr };
    unsigned blk = 256, grd = (HDC_WORDS + blk - 1) / blk;
    CUresult lr = G.cuLaunchKernel(fn, grd,1,1, blk,1,1, 0,NULL,args,NULL);
    CUresult sr = G.cuCtxSynchronize();

    if (lr != 0 || sr != 0) {
        G.cuMemFree(dv); G.cuMemFree(dout);
        hv_bundle(out, vecs, n);
        return;
    }

    G.cuMemcpyDtoH(out, dout, ob);
    G.cuMemFree(dv);
    G.cuMemFree(dout);
}

void hv_cuda_shutdown(void) {
    if (inited) {
        G.cuModuleUnload(mod);
        G.cuCtxDestroy(ctx);
        dlclose(G.h);
        inited = 0;
    }
}
