/*
 * probe_npu_matmul - run one xnnl matmul on the NPU and compare it with a CPU
 * reference.
 *
 * Facts recovered from the binaries and verified on device
 * (docs/NPU-MATMUL-API.md):
 *   - xnnlMatmulCreateGraph is a dispatcher: ParamCheck -> PolicyFunc ->
 *     createGraphSpecializedN (N <= 64) or createGraphSplitMN.
 *   - NHWC, rank 4, dtype HALF; input [1,1,1,K], weight [1,1,K,N],
 *     output [1,1,1,N]; an epilogue descriptor is mandatory.
 *   - tensors live in device memory obtained with midaAddressReserve(type 2)
 *     plus midaMemoryAllocWithAddr; that is the only path that yields a
 *     non-zero device VA (heap allocations report VA 0).
 *   - launching only works asynchronously with midaStreamConfig.type = 1.
 *
 * Build:
 *   aarch64-linux-android21-clang++ -O2 -std=c++17 -o probe_npu_matmul \
 *     tools/probe/probe_npu_matmul.cpp -ldl
 * Run: probe_npu_matmul [K M N]      (O3_HOST_PTR=1 uses host memory instead)
 */
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

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

template <typename T> static T sym(void * h, const char * n) {
    T p = (T) dlsym(h, n);
    if (!p) printf("missing symbol %s\n", n);
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

struct buf {
    void *   addr = nullptr;   /* midaAddress_t reservation (owns the device VA) */
    void *   mem  = nullptr;   /* heap memory: CPU pointer + dma-buf fd          */
    void *   mapped = nullptr; /* the same fd bound at the reserved VA           */
    void *   cpu  = nullptr;
    int      fd   = -1;
    uint64_t va   = 0;
};

int main(int argc, char ** argv) {
    int K = 64, M = 64, N = 8;
    if (argc >= 4) { K = atoi(argv[1]); M = atoi(argv[2]); N = atoi(argv[3]); }
    const bool host_mode = getenv("O3_HOST_PTR") != nullptr;
    printf("== probe_npu_matmul K=%d M=%d N=%d (%s, %s memory) ==\n", K, M, N,
           N <= 64 ? "specializedN" : "splitMN", host_mode ? "host" : "device");

    void * h_numd = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    void * h_xnnl = dlopen("/vendor/lib64/libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    printf("dlopen xnumd=%s xnnl=%s\n", h_numd ? "ok" : dlerror(), h_xnnl ? "ok" : dlerror());
    if (!h_numd || !h_xnnl) return 2;

    auto device_get   = sym<int (*)(void **, uint32_t, void *)>(h_numd, "midaDeviceGet");
    auto addr_reserve = sym<int (*)(void **, void *, uint64_t, uint32_t, uint32_t)>(h_numd, "midaAddressReserve");
    auto addr_va      = sym<uint64_t (*)(void *)>(h_numd, "midaAddressGetDeviceVa");
    auto mem_alloc    = sym<int (*)(void **, void *, uint64_t, uint32_t, uint32_t, const char *)>(h_numd, "midaMemoryAlloc");
    auto mem_get_fd   = sym<int (*)(void *)>(h_numd, "midaMemoryGetFd");
    auto mem_map_addr = sym<int (*)(void **, int, uint64_t, uint64_t, void *)>(h_numd, "midaMemoryMapWithAddr");
    auto mem_get_ptr  = sym<void * (*)(void *)>(h_numd, "midaMemoryGetPtr");
    auto mem_cache_st = sym<int (*)(void *)>(h_numd, "midaMemoryCacheStart");
    auto mem_cache_en = sym<int (*)(void *)>(h_numd, "midaMemoryCacheEnd");
    auto sub_create   = sym<int (*)(void **, void *, void *)>(h_numd, "midaSubgraphCreate");
    auto graph_create = sym<int (*)(void **, void *)>(h_numd, "midaGraphCreate");
    auto graph_add    = sym<int (*)(void **, void *, const void *, uint32_t, void *)>(h_numd, "midaGraphAddSubgraphNode");
    auto graph_inst   = sym<int (*)(void **, void *)>(h_numd, "midaGraphInstantiate");
    auto stream_create= sym<int (*)(void **, void *, void *)>(h_numd, "midaStreamCreate");
    auto launch_async = sym<int (*)(void *, void *, void *)>(h_numd, "midaStreamLaunchGraphAsync");
    auto launch_cb    = sym<int (*)(void *, void (*)(void *, int), void *, void *, void *)>(h_numd, "midaStreamLaunchGraphAsyncWithCallback");
    auto sync         = sym<int (*)(void *)>(h_numd, "midaStreamSynchronize");
    auto err_name     = sym<const char * (*)(int)>(h_numd, "midaGetErrorName");
    auto x_create     = sym<int (*)(xnnlHandle_t *, void *)>(h_xnnl, "xnnlCreate");
    auto d_create     = sym<int (*)(xnnlTensorDescriptor_t *)>(h_xnnl, "xnnlCreateTensorDescriptor");
    auto d_set        = sym<int (*)(xnnlTensorDescriptor_t, const int *, uint32_t, int, int, int, std::string)>(h_xnnl, "xnnlSetTensorDescriptor");
    auto e_create     = sym<int (*)(xnnlEpilogueDescriptor_t *)>(h_xnnl, "xnnlCreateEpilogueDescriptor");
    auto e_set        = sym<int (*)(xnnlEpilogueDescriptor_t, int, int, int, int, int, int, int, int, int, int)>(h_xnnl, "xnnlSetEpilogueDescriptor");
    auto matmul       = sym<int (*)(xnnlHandle_t, void *, xnnlTensorDescriptor_t, const void *,
                                    xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, const void *,
                                    const void *, const void *, xnnlTensorDescriptor_t, void *,
                                    xnnlEpilogueDescriptor_t, bool, uint32_t)>(h_xnnl, "xnnlMatmulCreateGraph");
    if (!device_get || !x_create || !d_create || !d_set || !matmul || !sub_create ||
        !graph_create || !graph_add || !graph_inst || !launch_async) {
        printf("RESULT missing_symbols\n");
        return 3;
    }

    struct midaDeviceConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    void * dev = nullptr;
    int e = device_get(&dev, 0, &cfg);
    printf("midaDeviceGet -> %d", e);
    if (e && err_name) printf(" [%s]", err_name(e));
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
    if (!di || !dw || !dout) { printf("RESULT descriptor_failed\n"); return 6; }

    xnnlEpilogueDescriptor_t epl = nullptr;
    if (e_create && e_set) { e_create(&epl); if (epl) e_set(epl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); }

    std::vector<float> a((size_t) K * M), b((size_t) K * N), ref((size_t) M * N);
    for (auto & v : a) v = frand();
    for (auto & v : b) v = frand();
    cpu_matmul(a, b, ref, K, M, N);
    std::vector<_Float16> fa((size_t) K * M), fb((size_t) K * N);
    for (size_t i = 0; i < a.size(); i++) fa[i] = (_Float16) a[i];
    for (size_t i = 0; i < b.size(); i++) fb[i] = (_Float16) b[i];

    /* --- buffers --- */
    buf bin, bw, bout;
    const uint64_t sz_in = ((uint64_t) K * M * 2 + 4095) & ~4095ull;
    const uint64_t sz_w  = ((uint64_t) K * N * 2 + 4095) & ~4095ull;
    const uint64_t sz_o  = ((uint64_t) M * N * 2 + 4095) & ~4095ull;

    auto setup = [&](buf & b, uint64_t size, const char * name) -> bool {
        if (host_mode) return true;
        /* heap allocation gives the CPU mapping and the dma-buf fd */
        int r1 = mem_alloc(&b.mem, dev, size, 1, 0, name);
        if (r1 != 0 || !b.mem) { printf("midaMemoryAlloc(%s) -> %d\n", name, r1); return false; }
        b.cpu = mem_get_ptr ? mem_get_ptr(b.mem) : nullptr;
        b.fd  = mem_get_fd ? mem_get_fd(b.mem) : -1;
        /* reserve a device address and bind the same fd there */
        int r2 = addr_reserve(&b.addr, dev, size, 2, 0);
        if (r2 != 0 || !b.addr) { printf("midaAddressReserve(%s) -> %d\n", name, r2); return false; }
        int r3 = mem_map_addr(&b.mapped, b.fd, size, 0, b.addr);
        b.va = addr_va ? addr_va(b.addr) : 0;
        printf("  %s: alloc=%d cpu=%p fd=%d reserve=%d va=0x%llx map=%d mapped=%p\n",
               name, r1, b.cpu, b.fd, r2, (unsigned long long) b.va, r3, b.mapped);
        return r3 == 0 && b.cpu != nullptr && b.va != 0;
    };
    if (!host_mode) {
        if (!setup(bin, sz_in, "in") || !setup(bw, sz_w, "w") || !setup(bout, sz_o, "out")) {
            printf("RESULT device_memory_failed\n");
            return 7;
        }
        memcpy(bin.cpu,  fa.data(), (size_t) K * M * 2);
        memcpy(bw.cpu,   fb.data(), (size_t) K * N * 2);
        memset(bout.cpu, 0, (size_t) M * N * 2);
        if (mem_cache_st) { mem_cache_st(bin.mem); mem_cache_st(bw.mem); mem_cache_st(bout.mem); }
        if (mem_cache_en) { mem_cache_en(bin.mem); mem_cache_en(bw.mem); mem_cache_en(bout.mem); }
    } else {
        bin.cpu = fa.data(); bw.cpu = fb.data(); bout.cpu = new _Float16[(size_t) M * N]();
        bin.va = (uint64_t) fa.data(); bw.va = (uint64_t) fb.data(); bout.va = (uint64_t) bout.cpu;
    }

    void * sub = nullptr;
    uint8_t subcfg[0xa0];
    memset(subcfg, 0, sizeof subcfg);
    e = sub_create(&sub, dev, subcfg);
    if (e || !sub) { printf("RESULT subgraph_failed\n"); return 8; }

    e = matmul(h, sub, di, (const void *) bin.va, dw, (const void *) bw.va,
               nullptr, nullptr, nullptr, nullptr, dout, (void *) bout.va, epl, false, 0);
    printf("xnnlMatmulCreateGraph(in=0x%llx w=0x%llx out=0x%llx) -> %d\n",
           (unsigned long long) bin.va, (unsigned long long) bw.va,
           (unsigned long long) bout.va, e);
    if (e != 0) { printf("RESULT builder_status_%d\n", e); return 9; }

    void * graph = nullptr;
    if (graph_create(&graph, dev) != 0 || !graph) { printf("RESULT graph_create_failed\n"); return 10; }
    uint8_t snp[0x28];
    memset(snp, 0, sizeof snp);
    memcpy(snp + 0x00, &sub, 8);
    void * node = nullptr;
    int ra = graph_add(&node, graph, nullptr, 0, snp);
    void * gexec = nullptr;
    int ri = (ra == 0) ? graph_inst(&gexec, graph) : -1;
    printf("graph_add=%d graph_instantiate=%d\n", ra, ri);
    if (ra || ri || !gexec) { printf("RESULT graph_instantiate_failed\n"); return 11; }

    uint32_t stype = 1;
    uint8_t scfg[0x0c];
    memset(scfg, 0, sizeof scfg);
    memcpy(scfg + 0x04, &stype, 4);
    void * stream = nullptr;
    if (stream_create(&stream, dev, scfg) != 0 || !stream) { printf("RESULT stream_failed\n"); return 12; }

    uint8_t lcfg[0xb8];
    memset(lcfg, 0, sizeof lcfg);
    uint32_t cu = getenv("O3_CU") ? (uint32_t) strtoul(getenv("O3_CU"), nullptr, 0) : 1, pf = 0, to = getenv("O3_FW_TO") ? (uint32_t) strtoul(getenv("O3_FW_TO"), nullptr, 0) : 10000;
    float ddr = 1.0f;
    memcpy(lcfg + 0x00, &cu, 4);
    memcpy(lcfg + 0x04, &pf, 4);
    memcpy(lcfg + 0x08, &ddr, 4);
    memcpy(lcfg + 0x0c, &to, 4);

    static volatile int cb_done = 0;
    static volatile int cb_status = -1;
    auto on_done = [](void *, int st) { cb_status = st; cb_done = 1; };

    int rl = (getenv("O3_CALLBACK") && launch_cb)
             ? launch_cb(gexec, (void (*)(void *, int)) on_done, nullptr, lcfg, stream)
             : launch_async(gexec, lcfg, stream);
    printf("launch_async -> %d", rl);
    if (rl && err_name) printf(" [%s]", err_name(rl));
    printf("\n");
    if (rl != 0) { printf("RESULT launch_failed\n"); return 13; }

    int rs;
    if (getenv("O3_CALLBACK")) {
        for (int i = 0; i < 200 && !cb_done; i++) usleep(10000);
        printf("callback fired=%d status=%d\n", (int) cb_done, cb_status);
        rs = cb_done ? cb_status : 10;
    } else {
        rs = sync ? sync(stream) : 0;
    }
    printf("sync -> %d", rs);
    if (rs && err_name) printf(" [%s]", err_name(rs));
    printf("\n");

    double maxdiff = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = std::fabs((double) ref[i] - (double) ((_Float16 *) bout.cpu)[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("max|NPU-CPU| = %.6g %s\n", maxdiff, maxdiff < 2e-2 ? "MATCH" : "MISMATCH");
    printf("RESULT %s\n", (rs == 0 && maxdiff < 2e-2) ? "ok" : (rs != 0 ? "sync_failed" : "numeric_mismatch"));
    return 0;
}
