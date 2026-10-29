/*
 * probe_openvx_matmul - run a fully-connected layer through the vendor OpenVX
 * stack and compare it with a CPU reference.
 *
 * Why this route instead of xnnl:
 *
 *   libxnnl (xnnlMatmulCreateGraph) is the LLM layer. It builds a mida graph
 *   that the firmware refused to execute (midaErrorFWTimeout, output all zero),
 *   because an executable NPU task needs a compiled program.
 *
 *   libOpenVX.so contains the compiler itself: vxGenerateNBG runs
 *   vxVerifyGraph + vxProcessGraph, and the library exports
 *   compiler_core_graph_optimization_hint, cr_compile_storage,
 *   compiler_reorder_* and friends. So building a graph through the public
 *   OpenVX API and calling vxVerifyGraph lets the vendor compiler produce the
 *   program for us.
 *
 *   vxFullyConnectedLayer(graph, inputs, weights, biases, overflow_policy,
 *                         rounding_policy, outputs)  (Khronos NN extension)
 *
 * Build:
 *   aarch64-linux-android21-clang -O2 -o probe_openvx_matmul \
 *     tools/probe/probe_openvx_matmul.c -ldl
 * Run: probe_openvx_matmul [K N]
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void * vx_context;
typedef void * vx_graph;
typedef void * vx_tensor;
typedef void * vx_node;
typedef void * vx_reference;
typedef int32_t vx_status;
typedef int32_t vx_enum;
typedef uint32_t vx_uint32;
typedef int32_t vx_int32;
typedef int8_t vx_int8;
typedef uint64_t vx_size;

#define VX_STATUS_OK 0

static float frand(void) { return (float) (rand() % 2000 - 1000) / 1000.0f; }

int main(int argc, char ** argv) {
    int K = 64, N = 8;
    if (argc >= 3) { K = atoi(argv[1]); N = atoi(argv[2]); }
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("== probe_openvx_matmul K=%d N=%d ==\n", K, N);

    /* libOpenVX needs libGAL and libVSC loaded first: vxCreateContext calls
     * gcoHAL_GetPLS, which crashes when the HAL libraries are not resident. */
    printf("preload libGAL=%p\n", dlopen("/vendor/lib64/libGAL.so", RTLD_NOW | RTLD_GLOBAL));
    printf("preload libVSC=%p\n", dlopen("/vendor/lib64/libVSC.so", RTLD_NOW | RTLD_GLOBAL));
    void * h = dlopen("/vendor/lib64/libOpenVX.so", RTLD_NOW | RTLD_GLOBAL);
    printf("dlopen libOpenVX=%s\n", h ? "ok" : dlerror());
    if (!h) return 2;

    vx_context (*pCreateContext)(vx_reference)                 = dlsym(h, "vxCreateContext");
    vx_graph   (*pCreateGraph)(vx_context)                      = dlsym(h, "vxCreateGraph");
    vx_tensor  (*pCreateTensor)(vx_context, vx_size, const vx_size *, vx_enum, vx_int8) =
                                                                     dlsym(h, "vxCreateTensor");
    vx_node    (*pFC)(vx_graph, vx_tensor, vx_tensor, vx_tensor, vx_enum, vx_enum, vx_tensor) =
                                                                     dlsym(h, "vxFullyConnectedLayer");
    vx_status  (*pVerify)(vx_graph)                             = dlsym(h, "vxVerifyGraph");
    vx_status  (*pProcess)(vx_graph)                            = dlsym(h, "vxProcessGraph");
    vx_status  (*pCopyPatch)(vx_tensor, vx_size, const vx_size *, const vx_size *,
                             const vx_size *, vx_size, void *)  = dlsym(h, "vxCopyTensorPatch");
    int32_t    (*pTypeSize)(vx_enum)                            = dlsym(h, "vxnneGetTypeSize");

    printf("syms: ctx=%d graph=%d tensor=%d fc=%d verify=%d process=%d copypatch=%d typesize=%d\n",
           pCreateContext != 0, pCreateGraph != 0, pCreateTensor != 0, pFC != 0,
           pVerify != 0, pProcess != 0, pCopyPatch != 0, pTypeSize != 0);
    if (!pCreateContext || !pCreateGraph || !pCreateTensor || !pFC || !pVerify || !pProcess) {
        printf("RESULT missing_symbols\n");
        return 3;
    }

    /*
     * The numeric values of VX_TYPE_* are not documented in the binary, so ask
     * the library: vxnneGetTypeSize maps a type enum to its byte width. Print
     * the table, then pick 2-byte (fp16) and 4-byte (fp32) candidates.
     */
    int t16 = -1, t32 = -1;
    if (pTypeSize) {
        printf("type sizes:");
        for (int t = 0; t <= 24; t++) {
            int sz = pTypeSize(t);
            if (sz > 0) printf(" %d:%d", t, sz);
            if (sz == 2 && t16 < 0) t16 = t;
            if (sz == 4 && t32 < 0) t32 = t;
        }
        printf("\n");
    }
    if (t16 < 0) t16 = 10;   /* fallbacks if the size table is unavailable */
    if (t32 < 0) t32 = 9;
    printf("chosen types: fp16=%d fp32=%d\n", t16, t32);

    vx_context ctx = pCreateContext(NULL);
    printf("vxCreateContext -> %p\n", ctx);
    if (!ctx) { printf("RESULT no_context\n"); return 4; }

    vx_graph g = pCreateGraph(ctx);
    printf("vxCreateGraph -> %p\n", g);
    if (!g) { printf("RESULT no_graph\n"); return 5; }

    /* fully connected: out[M,N] = in[M,K] * W[N,K]^T + b[N] */
    const vx_size M = 1;
    const vx_size in_dims[2]  = { K, M };   /* OpenVX tensors are [width, height] */
    const vx_size w_dims[2]   = { K, N };
    const vx_size b_dims[1]   = { N };
    const vx_size out_dims[2] = { N, M };

    vx_tensor tin  = pCreateTensor(ctx, 2, in_dims,  t32, 0);
    vx_tensor tw   = pCreateTensor(ctx, 2, w_dims,   t32, 0);
    vx_tensor tb   = pCreateTensor(ctx, 1, b_dims,   t32, 0);
    vx_tensor tout = pCreateTensor(ctx, 2, out_dims, t32, 0);
    printf("tensors in=%p w=%p b=%p out=%p\n", tin, tw, tb, tout);
    if (!tin || !tw || !tb || !tout) { printf("RESULT tensor_failed\n"); return 6; }

    /* fill: weights are [N][K] row-major, input is [M][K] */
    float * a = malloc(sizeof(float) * K * M);
    float * w = malloc(sizeof(float) * K * N);
    float * b = malloc(sizeof(float) * N);
    float * o = calloc((size_t) N * M, sizeof(float));
    float * ref = malloc(sizeof(float) * N * M);
    for (int i = 0; i < K * (int) M; i++) a[i] = frand();
    for (int i = 0; i < K * N; i++) w[i] = frand();
    for (int i = 0; i < N; i++) b[i] = frand();
    for (int n = 0; n < N; n++) {
        float acc = b[n];
        for (int k = 0; k < K; k++) acc += a[k] * w[k + n * K];
        ref[n] = acc;
    }

    const vx_size origin[2]   = { 0, 0 };
    const vx_size region_in[2]  = { K, M };
    const vx_size region_w[2]   = { K, N };
    const vx_size region_b[1]   = { N };
    const vx_size region_out[2] = { N, M };
    if (pCopyPatch) {
        printf("copy in -> %d\n", pCopyPatch(tin, 0, origin, region_in, NULL, 0, a));
        printf("copy w  -> %d\n", pCopyPatch(tw,  0, origin, region_w,  NULL, 0, w));
        printf("copy b  -> %d\n", pCopyPatch(tb,  0, origin, region_b,  NULL, 0, b));
    }

    vx_node node = pFC(g, tin, tw, tb, 0 /*overflow: saturate*/, 0 /*rounding*/, tout);
    printf("vxFullyConnectedLayer -> %p\n", node);
    if (!node) { printf("RESULT node_failed\n"); return 7; }

    vx_status st = pVerify(g);
    printf("vxVerifyGraph -> %d%s\n", st, st == VX_STATUS_OK ? " (compiler ran)" : "");
    if (st != VX_STATUS_OK) { printf("RESULT verify_failed\n"); return 8; }

    st = pProcess(g);
    printf("vxProcessGraph -> %d\n", st);
    if (st != VX_STATUS_OK) { printf("RESULT process_failed\n"); return 9; }

    if (pCopyPatch) {
        vx_status rs = pCopyPatch(tout, 0, origin, region_out, NULL, 0, o);
        printf("copy out -> %d\n", rs);
    }

    double maxdiff = 0.0;
    for (int i = 0; i < N * (int) M; i++) {
        double d = fabs((double) ref[i] - (double) o[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("max|NPU-CPU| = %.6g %s\n", maxdiff, maxdiff < 1e-2 ? "MATCH" : "MISMATCH");
    printf("RESULT %s\n", maxdiff < 1e-2 ? "ok" : "numeric_mismatch");
    return 0;
}
