/*
 * probe_npu_matmul - run one xnnl matmul on the NPU and compare it with a CPU
 * reference.
 *
 * Contract recovered from libxnnl.so (docs/NPU-MATMUL-API.md):
 *   - xnnlMatmulCreateGraph is a dispatcher: ParamCheck -> PolicyFunc ->
 *     createGraphSpecializedN (N <= 64) or createGraphSplitMN.
 *   - ParamCheck requires NHWC, rank 4, dims[1] == 1 for all tensors,
 *     input dims[3] == weight dims[2] == K, output dims[3] == weight dims[3] == N,
 *     input dims[2] == output dims[2], and dtype HALF (FLOAT is rejected).
 *   - an epilogue descriptor must be supplied; all modes disabled is accepted.
 *
 * Execution goes through the graph layer:
 *   midaGraphCreate -> midaGraphAddSubgraphNode -> midaGraphInstantiate
 *   -> midaStreamLaunchGraph -> midaStreamSynchronize
 *
 * Build:
 *   aarch64-linux-android21-clang++ -O2 -std=c++17 -o probe_npu_matmul \
 *     tools/probe/probe_npu_matmul.cpp -ldl
 * Run: probe_npu_matmul [K M N]
 */
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef struct xnnlHandle *            xnnlHandle_t;
typedef struct xnnlTensorDescriptor *  xnnlTensorDescriptor_t;
typedef struct xnnlEpilogueDescriptor *xnnlEpilogueDescriptor_t;

enum { XNNL_DATA_FLOAT = 0, XNNL_DATA_HALF = 1 };
enum { XNNL_TENSOR_NHWC = 2 };
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

static void * g_numd;
static void * g_xnnl;

template <typename T> static T sym(void * h, const char * n) {
    T p = (T) dlsym(h, n);
    if (!p) {
        printf("missing symbol %s\n", n);
    }
    return p;
}

static float frand(void) { return (float) (rand() % 2000 - 1000) / 1000.0f; }

static void cpu_matmul(const std::vector<float> & a, const std::vector<float> & b,
                       std::vector<float> & out, int K, int M, int N) {
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += a[k + m * K] * b[k + n * K];
            out[m + n * M] = acc;
        }
}

int main(int argc, char ** argv) {
    int K = 64, M = 64, N = 8;
    if (argc >= 4) { K = atoi(argv[1]); M = atoi(argv[2]); N = atoi(argv[3]); }
    printf("== probe_npu_matmul K=%d M=%d N=%d (%s) ==\n", K, M, N,
           N <= 64 ? "specializedN" : "splitMN");

    g_numd = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    g_xnnl = dlopen("/vendor/lib64/libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    printf("dlopen xnumd=%s xnnl=%s\n", g_numd ? "ok" : dlerror(), g_xnnl ? "ok" : dlerror());
    if (!g_numd || !g_xnnl) return 2;

    auto device_get   = sym<int (*)(void **, uint32_t, void *)>(g_numd, "midaDeviceGet");
    auto device_rel   = sym<int (*)(void *)>(g_numd, "midaDeviceRelease");
    auto mem_alloc    = sym<int (*)(void **, void *, uint64_t, uint32_t, uint32_t, const char *)>(g_numd, "midaMemoryAlloc");
    auto mem_get_ptr  = sym<void * (*)(void *)>(g_numd, "midaMemoryGetPtr");
    auto mem_get_va   = sym<uint64_t (*)(void *)>(g_numd, "midaMemoryGetDeviceVa");
    auto sub_create   = sym<int (*)(void **, void *, void *)>(g_numd, "midaSubgraphCreate");
    auto graph_create = sym<int (*)(void **, void *)>(g_numd, "midaGraphCreate");
    auto graph_add    = sym<int (*)(void **, void *, const void *, uint32_t, void *)>(g_numd, "midaGraphAddSubgraphNode");
    auto graph_inst   = sym<int (*)(void **, void *)>(g_numd, "midaGraphInstantiate");
    auto stream_create= sym<int (*)(void **, void *, void *)>(g_numd, "midaStreamCreate");
    auto launch_graph = sym<int (*)(void *, void *, void *)>(g_numd, "midaStreamLaunchGraph");
    auto launch_async = sym<int (*)(void *, void *, void *)>(g_numd, "midaStreamLaunchGraphAsync");
    auto sync         = sym<int (*)(void *)>(g_numd, "midaStreamSynchronize");
    auto err_name     = sym<const char * (*)(int)>(g_numd, "midaGetErrorName");
    auto err_str      = sym<const char * (*)(int)>(g_numd, "midaGetErrorString");
    auto x_create     = sym<int (*)(xnnlHandle_t *, void *)>(g_xnnl, "xnnlCreate");
    auto x_destroy    = sym<int (*)(xnnlHandle_t)>(g_xnnl, "xnnlDestroy");
    auto d_create     = sym<int (*)(xnnlTensorDescriptor_t *)>(g_xnnl, "xnnlCreateTensorDescriptor");
    auto d_set        = sym<int (*)(xnnlTensorDescriptor_t, const int *, uint32_t, int, int, int, std::string)>(g_xnnl, "xnnlSetTensorDescriptor");
    auto e_create     = sym<int (*)(xnnlEpilogueDescriptor_t *)>(g_xnnl, "xnnlCreateEpilogueDescriptor");
    auto e_set        = sym<int (*)(xnnlEpilogueDescriptor_t, int, int, int, int, int, int, int, int, int, int)>(g_xnnl, "xnnlSetEpilogueDescriptor");
    auto matmul       = sym<int (*)(xnnlHandle_t, void *, xnnlTensorDescriptor_t, const void *,
                                    xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, const void *,
                                    const void *, const void *, xnnlTensorDescriptor_t, void *,
                                    xnnlEpilogueDescriptor_t, bool, uint32_t)>(g_xnnl, "xnnlMatmulCreateGraph");
    if (!device_get || !x_create || !d_create || !d_set || !matmul || !sub_create ||
        !graph_create || !graph_add || !graph_inst || !launch_graph) {
        printf("RESULT missing_symbols\n");
        return 3;
    }

    struct midaDeviceConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    void * dev = nullptr;
    int e = device_get(&dev, 0, &cfg);
    printf("midaDeviceGet -> %d", e);
    if (e && err_name) printf(" [%s: %s]", err_name(e), err_str ? err_str(e) : "?");
    printf(" dev=%p\n", dev);
    if (e || !dev) { printf("RESULT no_device\n"); return 4; }

    xnnlHandle_t h = nullptr;
    e = x_create(&h, dev);
    printf("xnnlCreate -> %d handle=%p\n", e, h);
    if (e || !h) { printf("RESULT no_handle\n"); return 5; }

    auto mk = [&](const int * dims) -> xnnlTensorDescriptor_t {
        xnnlTensorDescriptor_t d = nullptr;
        if (d_create(&d) != 0) return nullptr;
        if (d_set(d, dims, 4, XNNL_TENSOR_NHWC, XNNL_DATA_HALF, XNNL_MEM_DDR, std::string("t")) != 0) return nullptr;
        return d;
    };
    const int in_d[4]  = { 1, 1, 1, K };
    const int w_d[4]   = { 1, 1, K, N };
    const int out_d[4] = { 1, 1, 1, N };
    xnnlTensorDescriptor_t di = mk(in_d), dw = mk(w_d), dout = mk(out_d);
    printf("descriptors in=%p w=%p out=%p\n", (void *) di, (void *) dw, (void *) dout);
    if (!di || !dw || !dout) { printf("RESULT descriptor_failed\n"); return 6; }

    xnnlEpilogueDescriptor_t epl = nullptr;
    if (e_create && e_set) {
        e_create(&epl);
        if (epl) e_set(epl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    printf("epilogue=%p\n", (void *) epl);

    std::vector<float> a((size_t) K * M), b((size_t) K * N), ref((size_t) M * N);
    std::vector<_Float16> ah((size_t) K * M), bh((size_t) K * N), outh((size_t) M * N, (_Float16) 0);
    for (auto & v : a) v = frand();
    for (auto & v : b) v = frand();
    cpu_matmul(a, b, ref, K, M, N);
    for (size_t i = 0; i < a.size(); i++) ah[i] = (_Float16) a[i];
    for (size_t i = 0; i < b.size(); i++) bh[i] = (_Float16) b[i];

    uint8_t subcfg[0xa0];
    memset(subcfg, 0, sizeof subcfg);
    /* the builder accepted plain host pointers, so allow both modes */
    const bool host_mode = getenv("O3_HOST_PTR") != nullptr;
    void * m_in = nullptr, * m_w = nullptr, * m_out = nullptr;
    void * p_in = nullptr, * p_w = nullptr, * p_out = nullptr;
    if (mem_alloc && !host_mode) {
        int r1 = mem_alloc(&m_in,  dev, (uint64_t) K * M * 2 + 4096, 1, 0, "in");
        int r2 = mem_alloc(&m_w,   dev, (uint64_t) K * N * 2 + 4096, 1, 0, "w");
        int r3 = mem_alloc(&m_out, dev, (uint64_t) M * N * 2 + 4096, 1, 0, "out");
        printf("midaMemoryAlloc tensors -> %d/%d/%d in=%p w=%p out=%p\n", r1, r2, r3, m_in, m_w, m_out);
        if (r1 || r2 || r3) { printf("RESULT tensor_alloc_failed\n"); return 8; }
        if (!mem_get_ptr || !mem_get_va) { printf("RESULT missing_memory_accessors\n"); return 8; }
        p_in = mem_get_ptr(m_in); p_w = mem_get_ptr(m_w); p_out = mem_get_ptr(m_out);
        uint64_t va_in = mem_get_va(m_in), va_w = mem_get_va(m_w), va_out = mem_get_va(m_out);
        printf("cpu ptr in=%p w=%p out=%p\n", p_in, p_w, p_out);
        printf("device va in=0x%llx w=0x%llx out=0x%llx\n",
               (unsigned long long) va_in, (unsigned long long) va_w, (unsigned long long) va_out);
        memcpy(p_in,  ah.data(), (size_t) K * M * 2);
        memcpy(p_w,   bh.data(), (size_t) K * N * 2);
        memset(p_out, 0, (size_t) M * N * 2);
        m_in = (void *) va_in; m_w = (void *) va_w; m_out = (void *) va_out;
    } else {
        m_in = ah.data(); m_w = bh.data(); m_out = outh.data();
        p_in = ah.data(); p_w = bh.data(); p_out = outh.data();
    }

    void * sub = nullptr;
    e = sub_create(&sub, dev, subcfg);
    printf("midaSubgraphCreate -> %d sub=%p\n", e, sub);
    if (e || !sub) { printf("RESULT subgraph_failed\n"); return 7; }

    e = matmul(h, sub, di, m_in, dw, m_w, nullptr, nullptr, nullptr, nullptr,
               dout, m_out, epl, false, 0);
    printf("xnnlMatmulCreateGraph -> %d\n", e);
    if (e != 0) { printf("RESULT builder_status_%d\n", e); return 8; }

    void * stack_mem = nullptr;
    if (mem_alloc) {
        int r = mem_alloc(&stack_mem, dev, 4u * 1024u * 1024u, 1, 0, "stack");
        printf("midaMemoryAlloc(stack 4MB) -> %d mem=%p\n", r, stack_mem);
        if (r != 0) stack_mem = nullptr;
    }

    const uint32_t cand[] = { 0, 0x100000, 0x200000 };
    for (uint32_t l2b : cand)
    for (uint32_t va : cand)
    for (uint32_t ss : cand) {
        if (ss && !stack_mem) continue;
        uint8_t snp[0x28];              /* midaSubgraphNodeParams: subgraph + execParams */
        memset(snp, 0, sizeof snp);
        memcpy(snp + 0x00, &sub, 8);
        memcpy(snp + 0x08, &l2b, 4);
        memcpy(snp + 0x0c, &va, 4);
        if (ss) {
            memcpy(snp + 0x18, &ss, 4);         /* stackInfo.stack_size */
            memcpy(snp + 0x20, &stack_mem, 8);  /* stackInfo.stack_mem  */
        }

        void * graph = nullptr;
        if (graph_create(&graph, dev) != 0 || !graph) { printf("graph create failed\n"); break; }
        void * node = nullptr;
        int ra = graph_add(&node, graph, nullptr, 0, snp);
        void * gexec = nullptr;
        int ri = (ra == 0) ? graph_inst(&gexec, graph) : -1;
        printf("graph_add=%d graph_inst=%d (l2b=0x%x va=0x%x stack=0x%x)\n", ra, ri, l2b, va, ss);
        fflush(stdout);
        if (ra != 0 || ri != 0 || !gexec) continue;

        for (uint32_t stype = 0; stype <= 2; stype++) {
            uint8_t scfg[0x0c];
            memset(scfg, 0, sizeof scfg);
            memcpy(scfg + 0x04, &stype, 4);
            void * stream = nullptr;
            int rs = stream_create(&stream, dev, scfg);
            if (rs != 0 || !stream) { printf("  stream(type=%u) create -> %d\n", stype, rs); continue; }

            uint8_t lcfg[0xb8];
            memset(lcfg, 0, sizeof lcfg);
            uint32_t cu = 1, pf = 0;
            uint32_t to = getenv("O3_FW_TO") ? (uint32_t) strtoul(getenv("O3_FW_TO"), nullptr, 0) : 10000;
            memcpy(lcfg + 0x00, &cu, 4);
            memcpy(lcfg + 0x04, &pf, 4);
            memcpy(lcfg + 0x0c, &to, 4);
            float ddr = 1.0f;
            memcpy(lcfg + 0x08, &ddr, 4);
            const char * nm = "mm";
            memcpy(lcfg + 0x10, &nm, sizeof nm);

            int rl = launch_graph(gexec, lcfg, stream);
            printf("  launch(type=%u) -> %d", stype, rl);
            if (rl != 0 && err_name) printf(" [%s]", err_name(rl));
            printf("\n");
            fflush(stdout);
            if (rl != 0 && launch_async) {
                int ra = launch_async(gexec, lcfg, stream);
                printf("  launch_async(type=%u) -> %d", stype, ra);
                if (ra != 0 && err_name) printf(" [%s]", err_name(ra));
                printf("\n");
                fflush(stdout);
                rl = ra;
            }
            if (rl == 0) {
                if (sync) printf("  sync -> %d\n", sync(stream));
                const _Float16 * got = (const _Float16 *) p_out;
                double maxdiff = 0.0;
                for (size_t i = 0; i < ref.size(); i++) {
                    double d = std::fabs((double) ref[i] - (double) got[i]);
                    if (d > maxdiff) maxdiff = d;
                }
                printf("  max|NPU-CPU| = %.6g %s\n", maxdiff, maxdiff < 2e-2 ? "MATCH" : "MISMATCH");
                printf("RESULT %s\n", maxdiff < 2e-2 ? "ok" : "numeric_mismatch");
                return 0;
            }
        }
    }

    printf("RESULT launch_failed\n");
    x_destroy(h);
    device_rel(dev);
    return 9;
}
