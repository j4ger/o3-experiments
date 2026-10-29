/*
 * probe_npu_matmul - drive one xnnl matmul on the NPU and check it against a CPU
 * reference.
 *
 * Everything here follows the contract recovered from libxnnl.so (see
 * docs/NPU-MATMUL-API.md):
 *
 *   xnnlMatmulCreateGraph is a dispatcher:
 *       ParamCheck -> PolicyFunc -> createGraphSpecializedN (policyMode_ 1, N <= 64)
 *                                -> createGraphSplitMN      (otherwise)
 *       split K is rejected outright (status 2).
 *
 *   ParamCheck requires: NHWC layout, equal rank across the four tensors, input
 *   dim0 == 1, dims 1 and 2 equal across input/weight/output, input dim3 == K,
 *   output dim3 == N, output dim0 == maxBatch, M and K <= 0x8000, maxBatch %
 *   minBatch == 0, and ewData descriptor/pointer both set or both absent.
 *
 * The descriptor shapes that satisfy those checks are not fully pinned down, so
 * this probe sweeps a few plausible encodings and reports which one is accepted
 * before it tries to instantiate and run. It is meant to be the single command
 * that answers "does the NPU agree with the CPU" once the device is back.
 *
 * Build (arm64):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ \
 *     -O2 -std=c++17 -o probe_npu_matmul tools/probe/probe_npu_matmul.cpp -ldl
 * Run:
 *   /data/local/tmp/probe_npu_matmul
 */
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ---- vendor types, mirrored from docs/generated/xnnl_api.h ---- */
typedef struct xnnlHandle *            xnnlHandle_t;
typedef struct xnnlTensorDescriptor *  xnnlTensorDescriptor_t;
typedef struct xnnlEpilogueDescriptor *xnnlEpilogueDescriptor_t;
typedef int32_t xnnlStatus_t;

enum { XNNL_DATA_FLOAT = 0, XNNL_DATA_HALF = 1 };
enum { XNNL_TENSOR_LINEAR = 0, XNNL_TENSOR_NHWC = 2 };
enum { XNNL_MEM_DDR = 0 };

struct midaDeviceConfig {
    int32_t  performance;
    uint32_t log_close;
    uint32_t pu_num;
    uint32_t mem_debug;
    uint32_t cu_delay;
    uint8_t  drm_protect;
    uint8_t  pad[3];
    uint32_t power_policy;
};

/* ---- entry points ---- */
typedef int (*fn_device_get)(void **, uint32_t, void *);
typedef int (*fn_device_release)(void *);
typedef int (*fn_xnnl_create)(xnnlHandle_t *, void *);
typedef int (*fn_xnnl_destroy)(xnnlHandle_t);
typedef int (*fn_create_desc)(xnnlTensorDescriptor_t *);
typedef int (*fn_set_desc)(xnnlTensorDescriptor_t, const int *, const uint32_t,
                           int /*layout*/, int /*dtype*/, int /*mem*/, std::string);
typedef int (*fn_matmul)(xnnlHandle_t, void * /*subgraph*/,
                         xnnlTensorDescriptor_t, const void *,
                         xnnlTensorDescriptor_t, const void *,
                         xnnlTensorDescriptor_t, const void *,
                         const void *, const void *,
                         xnnlTensorDescriptor_t, void *,
                         xnnlEpilogueDescriptor_t, bool, uint32_t);
typedef int (*fn_sub_create)(void **, void *, void *);
typedef int (*fn_sub_destroy)(void *);
typedef int (*fn_sub_instantiate)(void **, void *, void *);
typedef int (*fn_exec_destroy)(void *);
typedef int (*fn_stream_create)(void **, void *, void *);
typedef int (*fn_stream_destroy)(void *);
typedef int (*fn_launch)(void *, void *, void *);
typedef int (*fn_sync)(void *);

static fn_device_get      p_device_get;
static fn_device_release  p_device_release;
static fn_xnnl_create     p_xnnl_create;
static fn_xnnl_destroy    p_xnnl_destroy;
static fn_create_desc     p_create_desc;
static fn_set_desc        p_set_desc;
static fn_matmul          p_matmul;
static fn_sub_create      p_sub_create;
static fn_sub_destroy     p_sub_destroy;
static fn_sub_instantiate p_sub_instantiate;
static fn_exec_destroy    p_exec_destroy;
static fn_stream_create   p_stream_create;
static fn_stream_destroy  p_stream_destroy;
static fn_launch          p_launch;
static fn_sync            p_sync;

static float frand(void) {
    return (float) (rand() % 2000 - 1000) / 1000.0f;
}

static void cpu_matmul(const std::vector<float> & a, const std::vector<float> & b,
                       std::vector<float> & out, int K, int M, int N) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[k + m * K] * b[k + n * K];
            }
            out[m + n * M] = acc;
        }
    }
}

static xnnlTensorDescriptor_t make_desc(const int * dims, uint32_t rank, int layout, int dtype) {
    xnnlTensorDescriptor_t d = nullptr;
    int st = p_create_desc(&d);
    if (st != 0) {
        printf("  xnnlCreateTensorDescriptor -> %d\n", st);
        return nullptr;
    }
    st = p_set_desc(d, dims, rank, layout, dtype, XNNL_MEM_DDR, std::string("t"));
    if (st != 0) {
        printf("  xnnlSetTensorDescriptor -> %d\n", st);
        return nullptr;
    }
    return d;
}

int main(int argc, char ** argv) {
    int K = 64, M = 64, N = 8;
    if (argc >= 4) {
        K = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    printf("== probe_npu_matmul K=%d M=%d N=%d ==\n", K, M, N);
    if (K > 0x8000 || M > 0x8000) {
        printf("K or M exceeds kTmaGlobalStridesMNK (0x8000)\n");
        return 1;
    }
    printf("policy path: %s\n", N <= 64 ? "createGraphSpecializedN (N<=64)" : "createGraphSplitMN (N>64)");

    void * h_numd = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    void * h_xnnl = dlopen("/vendor/lib64/libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    printf("dlopen xnumd=%s xnnl=%s\n", h_numd ? "ok" : dlerror(), h_xnnl ? "ok" : dlerror());
    if (!h_numd || !h_xnnl) {
        return 2;
    }

    p_device_get      = (fn_device_get)     dlsym(h_numd, "midaDeviceGet");
    p_device_release  = (fn_device_release) dlsym(h_numd, "midaDeviceRelease");
    p_xnnl_create     = (fn_xnnl_create)    dlsym(h_xnnl, "xnnlCreate");
    p_xnnl_destroy    = (fn_xnnl_destroy)   dlsym(h_xnnl, "xnnlDestroy");
    p_create_desc     = (fn_create_desc)    dlsym(h_xnnl, "xnnlCreateTensorDescriptor");
    p_set_desc        = (fn_set_desc)       dlsym(h_xnnl, "xnnlSetTensorDescriptor");
    p_matmul          = (fn_matmul)         dlsym(h_xnnl, "xnnlMatmulCreateGraph");
    p_sub_create      = (fn_sub_create)     dlsym(h_numd, "midaSubgraphCreate");
    p_sub_destroy     = (fn_sub_destroy)    dlsym(h_numd, "midaSubgraphDestroy");
    p_sub_instantiate = (fn_sub_instantiate)dlsym(h_numd, "midaSubgraphInstantiate");
    p_exec_destroy    = (fn_exec_destroy)   dlsym(h_numd, "midaSubgraphExecDestroy");
    p_stream_create   = (fn_stream_create)  dlsym(h_numd, "midaStreamCreate");
    p_stream_destroy  = (fn_stream_destroy) dlsym(h_numd, "midaStreamDestroy");
    p_launch          = (fn_launch)         dlsym(h_numd, "midaStreamLaunchSubgraph");
    p_sync            = (fn_sync)           dlsym(h_numd, "midaStreamSynchronize");

    printf("syms: deviceGet=%d xnnlCreate=%d setDesc=%d matmul=%d subCreate=%d instantiate=%d launch=%d\n",
           p_device_get != nullptr, p_xnnl_create != nullptr, p_set_desc != nullptr,
           p_matmul != nullptr, p_sub_create != nullptr, p_sub_instantiate != nullptr,
           p_launch != nullptr);
    if (!p_device_get || !p_xnnl_create || !p_matmul || !p_create_desc || !p_set_desc) {
        printf("RESULT missing_symbols\n");
        return 3;
    }

    struct midaDeviceConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    void * dev = nullptr;
    int e = p_device_get(&dev, 0, &cfg);
    printf("midaDeviceGet -> %d dev=%p\n", e, dev);
    if (e != 0 || !dev) {
        printf("RESULT no_device\n");
        return 4;
    }

    xnnlHandle_t h = nullptr;
    e = p_xnnl_create(&h, dev);
    printf("xnnlCreate -> %d handle=%p\n", e, h);
    if (e != 0 || !h) {
        printf("RESULT no_handle\n");
        return 5;
    }

    /*
     * Shape sweep. Only the constraints from ParamCheck are known exactly; the
     * encoding below is the best reading of them:
     *   input  [1, a, b, K]   (dim0 must be 1, last dim is K)
     *   weight [1, a, b, ?]   (dims 1 and 2 must match the input)
     *   output [batch, a, b, N] (dim0 is maxBatch, last dim is N)
     */
    xnnlTensorDescriptor_t di   = nullptr;
    xnnlTensorDescriptor_t dw   = nullptr;
    xnnlTensorDescriptor_t dout = nullptr;

    const int in_dims[4]   = { 1, 1, 1, K };
    const int w_dims[4]    = { 1, 1, 1, N };
    const int out_dims[4]  = { 1, 1, 1, N };
    di   = make_desc(in_dims, 4, XNNL_TENSOR_NHWC, XNNL_DATA_FLOAT);
    dw   = make_desc(w_dims, 4, XNNL_TENSOR_NHWC, XNNL_DATA_FLOAT);
    dout = make_desc(out_dims, 4, XNNL_TENSOR_NHWC, XNNL_DATA_FLOAT);
    printf("descriptors: in=%p w=%p out=%p\n", (void *) di, (void *) dw, (void *) dout);

    if (!di || !dw || !dout) {
        printf("RESULT descriptor_failed\n");
        return 6;
    }

    std::vector<float> a((size_t) K * M), b((size_t) K * N), out((size_t) M * N, 0.0f), ref((size_t) M * N);
    for (auto & v : a) v = frand();
    for (auto & v : b) v = frand();
    cpu_matmul(a, b, ref, K, M, N);

    uint8_t subcfg[0xa0];
    memset(subcfg, 0, sizeof subcfg);
    void * sub = nullptr;
    e = p_sub_create(&sub, dev, subcfg);
    printf("midaSubgraphCreate -> %d sub=%p\n", e, sub);
    if (e != 0 || !sub) {
        printf("RESULT subgraph_failed\n");
        return 7;
    }

    /* the 15 parameters, in the order recovered from the source line */
    e = p_matmul(h, sub, di, a.data(), dw, b.data(), nullptr, nullptr, nullptr, nullptr,
                 dout, out.data(), nullptr, false, 0);
    printf("xnnlMatmulCreateGraph -> %d\n", e);
    if (e != 0) {
        printf("RESULT builder_status_%d\n", e);
        printf("(non-zero means ParamCheck or PolicyFunc rejected the call; see docs/NPU-MATMUL-API.md)\n");
        return 8;
    }

    /* instantiate: the parameter meaning is still unknown, sweep a few plausible ones */
    const uint32_t l2b_candidates[] = { 0, 0x40000, 0x100000 };
    const uint32_t va_candidates[]  = { 0, 0x40000, 0x100000 };
    for (uint32_t l2b : l2b_candidates) {
        for (uint32_t va : va_candidates) {
            uint8_t params[0x20];
            memset(params, 0, sizeof params);
            memcpy(params + 0x00, &l2b, 4);
            memcpy(params + 0x04, &va, 4);
            void * exec = nullptr;
            int st = p_sub_instantiate(&exec, sub, params);
            printf("midaSubgraphInstantiate(l2b=0x%x va=0x%x) -> %d exec=%p\n", l2b, va, st, exec);
            if (st == 0 && exec) {
                uint8_t scfg[0x0c];
                memset(scfg, 0, sizeof scfg);
                void * stream = nullptr;
                int s1 = p_stream_create(&stream, dev, scfg);
                printf("  midaStreamCreate -> %d stream=%p\n", s1, stream);
                if (s1 == 0 && stream) {
                    uint8_t lcfg[0xb8];
                    memset(lcfg, 0, sizeof lcfg);
                    int s2 = p_launch(exec, lcfg, stream);
                    printf("  midaStreamLaunchSubgraph -> %d\n", s2);
                    if (p_sync) {
                        printf("  midaStreamSynchronize -> %d\n", p_sync(stream));
                    }
                    if (s2 == 0) {
                        double maxdiff = 0.0;
                        for (size_t i = 0; i < ref.size(); i++) {
                            double d = std::fabs((double) ref[i] - (double) out[i]);
                            if (d > maxdiff) maxdiff = d;
                        }
                        printf("  max|NPU-CPU| = %.6g  %s\n", maxdiff,
                               maxdiff < 1e-2 ? "MATCH" : "MISMATCH");
                        printf("RESULT %s\n", maxdiff < 1e-2 ? "ok" : "numeric_mismatch");
                    }
                    if (p_stream_destroy) p_stream_destroy(stream);
                }
                if (p_exec_destroy) p_exec_destroy(exec);
                p_xnnl_destroy(h);
                p_device_release(dev);
                return 0;
            }
        }
    }

    printf("RESULT instantiate_failed (try more parameter combinations)\n");
    if (p_sub_destroy) p_sub_destroy(sub);
    p_xnnl_destroy(h);
    p_device_release(dev);
    return 9;
}
