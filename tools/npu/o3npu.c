/*
 * o3npu - driver layer for the Xiaomi 18 Fold (lhasa) NPU.
 *
 * See o3npu.h for the stack.  This file is C on purpose: mida is a plain C
 * ABI, and the only C++ dependency (xnnlSetTensorDescriptor takes a
 * std::string) is isolated in xnnl_shim.cpp.
 */
#define _GNU_SOURCE 1

#include "o3npu.h"
#include "vendor/mida_types.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* xnnl_shim.cpp */
extern int  o3xnnl_open(void ** out, void * mida_device);
extern void o3xnnl_close(void * handle);
extern int  o3xnnl_unary(void * handle, void * subgraph, const char * name,
                         int n, const void * x, void * y);
extern int  o3xnnl_rmsnorm(void * handle, void * subgraph, int K,
                           const void * x, const void * w, void * y);
extern int  o3xnnl_matmul_graph(void * handle, void * subgraph,
                                const int in_dims[4], const void * in_ptr,
                                const int w_dims[4], const void * w_ptr,
                                const int out_dims[4], void * out_ptr,
                                int transposed_w);

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static int env_int(const char * name, int def);
float o3npu_fp16_to_fp32(uint16_t h);
static void fill_devcfg(midaDeviceConfig * c);

/* ------------------------------------------------------------------ */
/* fp16 -> fp32 (no ggml dependency here)                              */
/* ------------------------------------------------------------------ */

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* ------------------------------------------------------------------ */
/* mida bindings                                                       */
/* ------------------------------------------------------------------ */

struct mida_api {
    void * lib;

    int  (*device_get)(midaDevice_t *, uint32_t, midaDeviceConfig *);
    int  (*device_release)(midaDevice_t);
    int  (*device_reset)(midaDevice_t);
    int  (*device_get_limit)(uint32_t *, int, midaDevice_t);
    int  (*device_set_limit)(int, uint32_t);

    int  (*mem_alloc)(midaMemory_t *, midaDevice_t, uint64_t, uint32_t, uint32_t, const char *);
    int  (*mem_free)(midaMemory_t);
    void * (*mem_ptr)(midaMemory_t);
    int  (*mem_fd)(midaMemory_t);
    uint64_t (*mem_size)(midaMemory_t);
    uint64_t (*mem_va)(midaMemory_t);
    int  (*mem_cache_start)(midaMemory_t);
    int  (*mem_cache_end)(midaMemory_t);

    int  (*addr_reserve)(midaAddress_t *, midaDevice_t, uint64_t, uint32_t, uint32_t);
    int  (*addr_free)(midaAddress_t);
    uint64_t (*addr_va)(midaAddress_t);
    int  (*mem_map_addr)(midaMemory_t *, int, uint64_t, uint64_t, midaAddress_t);

    int  (*sub_create)(midaSubgraph_t *, midaDevice_t, midaSubgraphConfig *);
    int  (*sub_destroy)(midaSubgraph_t);
    int  (*sub_instantiate)(midaSubgraphExec_t *, midaSubgraph_t, midaSubgraphExecParams *);
    int  (*sub_exec_destroy)(midaSubgraphExec_t);
    int  (*sub_exec_stack_info)(midaSubgraphExec_t, midaSubgraphStackInfo *);
    int  (*sub_print_info)(midaSubgraph_t);
    int  (*sub_exec_print_info)(midaSubgraphExec_t);
    int  (*sub_exec_dump)(midaSubgraphExec_t, const char *);

    int  (*stream_create)(midaStream_t *, midaDevice_t, midaStreamConfig *);
    int  (*stream_destroy)(midaStream_t);
    int  (*stream_launch_subgraph_async)(midaSubgraphExec_t, midaLaunchConfig *, midaStream_t);

    int  (*graph_create)(midaGraph_t *, midaDevice_t);
    int  (*graph_destroy)(midaGraph_t);
    int  (*graph_add_node)(midaGraphNode_t *, midaGraph_t, const midaGraphNode_t *, uint32_t, midaSubgraphNodeParams *);
    int  (*graph_instantiate)(midaGraphExec_t *, midaGraph_t);
    int  (*graph_exec_get_sub)(midaSubgraphExec_t *, midaGraphExec_t, midaGraphNode_t);
    int  (*graph_exec_destroy)(midaGraphExec_t);
    int  (*stream_launch_graph_async)(midaGraphExec_t, midaLaunchConfig *, midaStream_t);
    int  (*stream_sync)(midaStream_t);

    const char * (*error_name)(int);
};

static struct mida_api * g_api = NULL;

static void * need_sym(void * lib, const char * name) {
    void * p = dlsym(lib, name);
    if (!p) {
        fprintf(stderr, "o3npu: missing symbol %s\n", name);
    }
    return p;
}

static struct mida_api * mida_get(void) {
    if (g_api) {
        return g_api;
    }
    struct mida_api * a = (struct mida_api *) calloc(1, sizeof *a);
    if (!a) {
        return NULL;
    }
    a->lib = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    if (!a->lib) {
        a->lib = dlopen("libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!a->lib) {
        fprintf(stderr, "o3npu: dlopen libxnumd failed: %s\n", dlerror());
        free(a);
        return NULL;
    }

    a->device_get    = (int (*)(midaDevice_t *, uint32_t, midaDeviceConfig *)) need_sym(a->lib, "midaDeviceGet");
    a->device_release= (int (*)(midaDevice_t)) need_sym(a->lib, "midaDeviceRelease");
    a->device_reset  = (int (*)(midaDevice_t)) need_sym(a->lib, "midaDeviceReset");
    a->device_get_limit = (int (*)(uint32_t *, int, midaDevice_t)) dlsym(a->lib, "midaDeviceGetLimit");
    a->device_set_limit = (int (*)(int, uint32_t)) dlsym(a->lib, "midaDeviceSetLimit");

    a->mem_alloc     = (int (*)(midaMemory_t *, midaDevice_t, uint64_t, uint32_t, uint32_t, const char *)) need_sym(a->lib, "midaMemoryAlloc");
    a->mem_free      = (int (*)(midaMemory_t)) need_sym(a->lib, "midaMemoryFree");
    a->mem_ptr       = (void * (*)(midaMemory_t)) need_sym(a->lib, "midaMemoryGetPtr");
    a->mem_fd        = (int (*)(midaMemory_t)) need_sym(a->lib, "midaMemoryGetFd");
    a->mem_size      = (uint64_t (*)(midaMemory_t)) dlsym(a->lib, "midaMemorySize");
    a->mem_va        = (uint64_t (*)(midaMemory_t)) dlsym(a->lib, "midaMemoryGetDeviceVa");
    a->mem_cache_start = (int (*)(midaMemory_t)) dlsym(a->lib, "midaMemoryCacheStart");
    a->mem_cache_end = (int (*)(midaMemory_t)) dlsym(a->lib, "midaMemoryCacheEnd");

    a->addr_reserve  = (int (*)(midaAddress_t *, midaDevice_t, uint64_t, uint32_t, uint32_t)) need_sym(a->lib, "midaAddressReserve");
    a->addr_free     = (int (*)(midaAddress_t)) dlsym(a->lib, "midaAddressFree");
    a->addr_va       = (uint64_t (*)(midaAddress_t)) need_sym(a->lib, "midaAddressGetDeviceVa");
    a->mem_map_addr  = (int (*)(midaMemory_t *, int, uint64_t, uint64_t, midaAddress_t)) need_sym(a->lib, "midaMemoryMapWithAddr");

    a->sub_create    = (int (*)(midaSubgraph_t *, midaDevice_t, midaSubgraphConfig *)) need_sym(a->lib, "midaSubgraphCreate");
    a->sub_destroy   = (int (*)(midaSubgraph_t)) need_sym(a->lib, "midaSubgraphDestroy");
    a->sub_instantiate = (int (*)(midaSubgraphExec_t *, midaSubgraph_t, midaSubgraphExecParams *)) need_sym(a->lib, "midaSubgraphInstantiate");
    a->sub_exec_destroy = (int (*)(midaSubgraphExec_t)) need_sym(a->lib, "midaSubgraphExecDestroy");
    a->sub_exec_stack_info = (int (*)(midaSubgraphExec_t, midaSubgraphStackInfo *)) dlsym(a->lib, "midaSubgraphExecGetStackInfo");
    a->sub_print_info = (int (*)(midaSubgraph_t)) dlsym(a->lib, "midaSubgraphPrintInfo");
    a->sub_exec_print_info = (int (*)(midaSubgraphExec_t)) dlsym(a->lib, "midaSubgraphExecPrintInfo");
    a->sub_exec_dump = (int (*)(midaSubgraphExec_t, const char *)) dlsym(a->lib, "midaSubgraphExecDumpDataToFile");

    a->stream_create = (int (*)(midaStream_t *, midaDevice_t, midaStreamConfig *)) need_sym(a->lib, "midaStreamCreate");
    a->stream_destroy= (int (*)(midaStream_t)) need_sym(a->lib, "midaStreamDestroy");
    a->stream_launch_subgraph_async = (int (*)(midaSubgraphExec_t, midaLaunchConfig *, midaStream_t)) need_sym(a->lib, "midaStreamLaunchSubgraphAsync");

    a->graph_create = (int (*)(midaGraph_t *, midaDevice_t)) dlsym(a->lib, "midaGraphCreate");
    a->graph_destroy = (int (*)(midaGraph_t)) dlsym(a->lib, "midaGraphDestroy");
    a->graph_add_node = (int (*)(midaGraphNode_t *, midaGraph_t, const midaGraphNode_t *, uint32_t, midaSubgraphNodeParams *)) dlsym(a->lib, "midaGraphAddSubgraphNode");
    a->graph_instantiate = (int (*)(midaGraphExec_t *, midaGraph_t)) dlsym(a->lib, "midaGraphInstantiate");
    a->graph_exec_get_sub = (int (*)(midaSubgraphExec_t *, midaGraphExec_t, midaGraphNode_t)) dlsym(a->lib, "midaGraphExecGetSubgraphExec");
    a->graph_exec_destroy = (int (*)(midaGraphExec_t)) dlsym(a->lib, "midaGraphExecDestroy");
    a->stream_launch_graph_async = (int (*)(midaGraphExec_t, midaLaunchConfig *, midaStream_t)) dlsym(a->lib, "midaStreamLaunchGraphAsync");
    a->stream_sync   = (int (*)(midaStream_t)) need_sym(a->lib, "midaStreamSynchronize");

    a->error_name    = (const char * (*)(int)) dlsym(a->lib, "midaGetErrorName");

    if (!a->device_get || !a->mem_alloc || !a->mem_free ||
        !a->addr_reserve || !a->mem_map_addr ||
        !a->sub_create || !a->sub_instantiate ||
        !a->stream_create || !a->stream_launch_subgraph_async) {
        fprintf(stderr, "o3npu: device symbol resolution failed\n");
        dlclose(a->lib);
        free(a);
        return NULL;
    }
    g_api = a;
    return a;
}

static const char * err_str(const struct mida_api * a, int e) {
    static char buf[128];
    const char * n = (a && a->error_name) ? a->error_name(e) : NULL;
    snprintf(buf, sizeof buf, "%d%s%s", e, n ? " (" : "", n ? n : "");
    if (n) {
        size_t l = strlen(buf);
        snprintf(buf + l, sizeof buf - l, ")");
    }
    return buf;
}

/* ------------------------------------------------------------------ */
/* device buffers                                                      */
/* ------------------------------------------------------------------ */

struct dbuf {
    midaMemory_t   mem;
    midaAddress_t  addr;
    void *         cpu;
    int            fd;
    uint64_t       va;
    uint64_t       size;
};

static void dbuf_free(struct mida_api * a, struct dbuf * d) {
    if (d->mem) {
        a->mem_free(d->mem);
    }
    if (d->addr && a->addr_free) {
        a->addr_free(d->addr);
    }
    memset(d, 0, sizeof *d);
    d->fd = -1;
}

static int dbuf_reserve(struct mida_api * a, midaDevice_t dev, struct dbuf * d,
                        uint64_t want, const char * name) {
    if (want == 0) {
        want = 4096;
    }
    want = (want + 0xfff) & ~0xfffull;
    if (d->size >= want) {
        return 0;
    }
    dbuf_free(a, d);

    int mflag = env_int("O3NPU_MEM_FLAG", 5);
    int rc = a->mem_alloc(&d->mem, dev, want, (uint32_t) mflag, 0, name);
    if (rc != 0 || !d->mem) {
        fprintf(stderr, "o3npu: midaMemoryAlloc(%s,%llu) -> %s\n", name,
                (unsigned long long) want, err_str(a, rc));
        return -1;
    }
    d->cpu = a->mem_ptr(d->mem);
    d->fd  = a->mem_fd(d->mem);
    d->va  = a->mem_va ? a->mem_va(d->mem) : 0;

    if (d->va == 0) {
        /* some allocation flags do not expose a VA; bind the dma-buf to a
         * reserved address instead (the path used by the earlier probes) */
        rc = a->addr_reserve(&d->addr, dev, want, 2, 0);
        if (rc != 0 || !d->addr) {
            fprintf(stderr, "o3npu: midaAddressReserve(%s) -> %s\n", name, err_str(a, rc));
            return -1;
        }
        d->va = a->addr_va(d->addr);

        midaMemory_t mapped = NULL;
        rc = a->mem_map_addr(&mapped, d->fd, want, 0, d->addr);
        if (rc != 0) {
            fprintf(stderr, "o3npu: midaMemoryMapWithAddr(%s) -> %s\n", name, err_str(a, rc));
            return -1;
        }
    }
    d->size = want;
    if (getenv("O3NPU_TRACE")) {
        printf("o3npu: buf %-7s size=%llu flag=%d cpu=%p fd=%d va=0x%llx\n",
               name, (unsigned long long) want, mflag, d->cpu, d->fd, (unsigned long long) d->va);
    }
    if (!d->cpu || d->va == 0) {
        fprintf(stderr, "o3npu: buffer %s cpu=%p va=0x%llx\n", name, d->cpu,
                (unsigned long long) d->va);
        return -1;
    }
    return 0;
}

static void dbuf_sync_cpu(struct mida_api * a, struct dbuf * d) {
    if (a->mem_cache_start) a->mem_cache_start(d->mem);
}

static void dbuf_sync_dev(struct mida_api * a, struct dbuf * d) {
    if (a->mem_cache_end) a->mem_cache_end(d->mem);
}

/* ------------------------------------------------------------------ */
/* legacy o3npu_t API                                                  */
/* ------------------------------------------------------------------ */

struct o3npu {
    struct mida_api * api;
    midaDevice_t dev;
    void * xnnl;
};

o3npu_t * o3npu_open(void) {
    struct mida_api * a = mida_get();
    if (!a) {
        return NULL;
    }
    o3npu_t * n = (o3npu_t *) calloc(1, sizeof *n);
    if (!n) {
        return NULL;
    }
    n->api = a;
    midaDeviceConfig cfg;
    fill_devcfg(&cfg);
    int rc = a->device_get(&n->dev, 0, &cfg);
    if (rc != 0 || !n->dev) {
        fprintf(stderr, "o3npu: midaDeviceGet -> %s\n", err_str(a, rc));
        free(n);
        return NULL;
    }
    o3xnnl_open(&n->xnnl, n->dev);
    return n;
}

void o3npu_close(o3npu_t * n) {
    if (!n) {
        return;
    }
    if (n->xnnl) {
        o3xnnl_close(n->xnnl);
    }
    if (n->dev && n->api) {
        n->api->device_release(n->dev);
    }
    free(n);
}

void * o3npu_alloc(o3npu_t * n, size_t size, int cpu_visible, size_t * out_size) {
    if (!n) {
        return NULL;
    }
    midaMemory_t mem = NULL;
    int rc = n->api->mem_alloc(&mem, n->dev, size, cpu_visible ? 1 : 0, 0, "o3npu");
    if (rc != 0) {
        return NULL;
    }
    if (out_size && n->api->mem_size) {
        *out_size = (size_t) n->api->mem_size(mem);
    }
    return mem;
}

void * o3npu_cpu_ptr(o3npu_t * n, void * mem) {
    return (n && n->api) ? n->api->mem_ptr((midaMemory_t) mem) : NULL;
}

uint64_t o3npu_dev_va(o3npu_t * n, void * mem) {
    return (n && n->api && n->api->mem_va) ? n->api->mem_va((midaMemory_t) mem) : 0;
}

int o3npu_mem_fd(o3npu_t * n, void * mem) {
    return (n && n->api) ? n->api->mem_fd((midaMemory_t) mem) : -1;
}

void o3npu_free(o3npu_t * n, void * mem) {
    if (n && n->api && mem) {
        n->api->mem_free((midaMemory_t) mem);
    }
}

int o3npu_selftest(void) {
    printf("== o3npu selftest ==\n");
    o3npu_t * n = o3npu_open();
    printf("o3npu_open -> %p\n", (void *) n);
    if (!n) {
        return 1;
    }
    size_t sz = 0;
    void * host = o3npu_alloc(n, 4096, 1, &sz);
    printf("  alloc(cpu_visible) -> %p size=%zu\n", host, sz);
    if (host) {
        void * p = o3npu_cpu_ptr(n, host);
        printf("  cpu_ptr=%p dev_va=0x%llx fd=%d\n", p,
               (unsigned long long) o3npu_dev_va(n, host), o3npu_mem_fd(n, host));
        o3npu_free(n, host);
    }
    printf("RESULT %s\n", host ? "ok" : "alloc_failed");
    o3npu_close(n);
    return host ? 0 : 2;
}

/* ------------------------------------------------------------------ */
/* matmul                                                              */
/* ------------------------------------------------------------------ */

struct o3npu_matmul {
    struct mida_api * api;
    midaDevice_t dev;
    void *       xnnl;
    midaStream_t stream;
    struct dbuf  in;
    struct dbuf  w;
    struct dbuf  out;
    struct dbuf  stack;
    int          trace;
    int          transposed_w;   /* 1: weight stays in ggml layout */
    char         err[256];
};


static void apply_limit_env(struct mida_api * a, midaDevice_t dev) {
    (void) dev;
    const char * set = getenv("O3NPU_LIMIT_SET");
    if (!set || !*set || !a->device_set_limit) {
        return;
    }
    int idx = 0;
    unsigned long val = 0;
    if (sscanf(set, "%d:%lu", &idx, &val) == 2) {
        int r = a->device_set_limit(idx, (uint32_t) val);
        printf("o3npu: set_limit[%d]=%lu -> %d\n", idx, val, r);
    }
}

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    if (!v || !*v) {
        return def;
    }
    return (int) strtol(v, NULL, 0);
}

static void fill_devcfg(midaDeviceConfig * c) {
    memset(c, 0, sizeof *c);
    c->type        = (uint32_t) env_int("O3NPU_DEV_TYPE", 0);
    c->mode        = (uint32_t) env_int("O3NPU_DEV_MODE", 0);
    c->flag        = (uint32_t) env_int("O3NPU_DEV_FLAG", 0);
    c->count       = (uint32_t) env_int("O3NPU_DEV_COUNT", 0);
    c->dsmem_large = (uint32_t) env_int("O3NPU_DEV_DSMEM", 0);
    c->power_policy= (uint32_t) env_int("O3NPU_DEV_POWER", 0);
}

o3npu_matmul_t * o3npu_matmul_open(void) {
    struct mida_api * a = mida_get();
    if (!a) {
        return NULL;
    }
    o3npu_matmul_t * m = (o3npu_matmul_t *) calloc(1, sizeof *m);
    if (!m) {
        return NULL;
    }
    m->api = a;
    m->in.fd = m->w.fd = m->out.fd = -1;
    m->trace = getenv("O3NPU_TRACE") != NULL;
    m->transposed_w = env_int("O3NPU_TRANSPOSED_W", 1);

    midaDeviceConfig cfg;
    fill_devcfg(&cfg);
    int rc = a->device_get(&m->dev, 0, &cfg);
    if (rc == 0 && m->dev && env_int("O3NPU_RESET", 0) && a->device_reset) {
        int rr = a->device_reset(m->dev);
        if (m->trace) printf("o3npu: midaDeviceReset -> %d\n", rr);
    }
    if (rc != 0 || !m->dev) {
        snprintf(m->err, sizeof m->err, "midaDeviceGet -> %s", err_str(a, rc));
        free(m);
        return NULL;
    }

    rc = o3xnnl_open(&m->xnnl, m->dev);
    if (rc != 0 || !m->xnnl) {
        snprintf(m->err, sizeof m->err, "xnnlCreate -> %d", rc);
        a->device_release(m->dev);
        free(m);
        return NULL;
    }

    midaStreamConfig scfg;
    memset(&scfg, 0, sizeof scfg);
    scfg.type = 1;  /* the only accepted value (see docs/NPU-MATMUL-API.md 10.1) */
    rc = a->stream_create(&m->stream, m->dev, &scfg);
    if (rc != 0 || !m->stream) {
        snprintf(m->err, sizeof m->err, "midaStreamCreate -> %s", err_str(a, rc));
        o3xnnl_close(m->xnnl);
        a->device_release(m->dev);
        free(m);
        return NULL;
    }
    if (m->trace) {
        printf("o3npu_matmul: device=%p stream=%p transposed_w=%d\n",
               m->dev, m->stream, m->transposed_w);
    }
    return m;
}

void o3npu_matmul_close(o3npu_matmul_t * m) {
    if (!m) {
        return;
    }
    dbuf_free(m->api, &m->in);
    dbuf_free(m->api, &m->w);
    dbuf_free(m->api, &m->out);
    if (m->stream) m->api->stream_destroy(m->stream);
    if (m->xnnl) o3xnnl_close(m->xnnl);
    if (m->dev) m->api->device_release(m->dev);
    free(m);
}

const char * o3npu_matmul_error(const o3npu_matmul_t * m) {
    return m ? m->err : "no handle";
}

void o3npu_matmul_set_trace(int on) {
    (void) on;
}

static void fill_stripe_names(midaLaunchConfig * lc) {
    const char * env = getenv("O3NPU_STRIPE");
    if (!env || !*env) {
        return;
    }
    char tmp[512];
    snprintf(tmp, sizeof tmp, "%s", env);
    char * save = NULL;
    char * tok = strtok_r(tmp, ",", &save);
    int i = 0;
    while (tok && i < 32) {
        snprintf(lc->stripe_names[i], 5, "%s", tok);
        tok = strtok_r(NULL, ",", &save);
        i++;
    }
}

int o3npu_matmul_run(o3npu_matmul_t * m, int K, int M, int T,
                     const uint16_t * w_f16,
                     const uint16_t * x_f16,
                     float * out_f32) {
    if (!m || K <= 0 || M <= 0 || T <= 0) {
        return -1;
    }
    struct mida_api * a = m->api;
    int rc;

    m->err[0] = 0;

    if (dbuf_reserve(a, m->dev, &m->in, (uint64_t) K * T * 2, "o3in") != 0 ||
        dbuf_reserve(a, m->dev, &m->w, (uint64_t) K * M * 2, "o3w") != 0 ||
        dbuf_reserve(a, m->dev, &m->out, (uint64_t) M * T * 2, "o3out") != 0) {
        snprintf(m->err, sizeof m->err, "buffer allocation failed");
        return -2;
    }

    /* upload activations: already in the layout xnnl wants ([K] per token) */
    dbuf_sync_cpu(a, &m->in);
    memcpy(m->in.cpu, x_f16, (size_t) K * T * 2);
    dbuf_sync_dev(a, &m->in);

    /* upload weights */
    dbuf_sync_cpu(a, &m->w);
    if (m->transposed_w) {
        /* ggml layout is already [M][K] with K fastest */
        memcpy(m->w.cpu, w_f16, (size_t) K * M * 2);
    } else {
        /* transpose to xnnl layout [K][M] with M fastest */
        const uint16_t * src = w_f16;
        uint16_t * dst = (uint16_t *) m->w.cpu;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < M; j++) {
                dst[(size_t) k * M + j] = src[(size_t) j * K + k];
            }
        }
    }
    dbuf_sync_dev(a, &m->w);

    dbuf_sync_cpu(a, &m->out);
    memset(m->out.cpu, 0, (size_t) M * T * 2);
    dbuf_sync_dev(a, &m->out);

    midaSubgraphConfig subcfg;
    memset(&subcfg, 0, sizeof subcfg);
    midaSubgraph_t sub = NULL;
    rc = a->sub_create(&sub, m->dev, &subcfg);
    if (rc != 0 || !sub) {
        snprintf(m->err, sizeof m->err, "midaSubgraphCreate -> %s", err_str(a, rc));
        return -3;
    }

    const int in_dims[4]  = { 1, 1, 1, K };
    int       w_dims[4]   = { 1, 1, K, M };
    const int out_dims[4] = { T, 1, 1, M };
    if (m->transposed_w && env_int("O3NPU_WDIMS_MK", 1)) {
        /* transposed weight: the descriptor may need {1,1,M,K} instead */
        w_dims[2] = M;
        w_dims[3] = K;
    }

    rc = o3xnnl_matmul_graph(m->xnnl, sub,
                             in_dims, (const void *) m->in.va,
                             w_dims,  (const void *) m->w.va,
                             out_dims, (void *) m->out.va,
                             m->transposed_w);
    if (rc != 0) {
        snprintf(m->err, sizeof m->err, "xnnlMatmulCreateGraph -> %d", rc);
        a->sub_destroy(sub);
        return -4;
    }

    const char * path_mode = getenv("O3NPU_PATH");
    if (path_mode && strcmp(path_mode, "graph") == 0) {
        if (!a->graph_create || !a->graph_add_node || !a->graph_instantiate ||
            !a->stream_launch_graph_async) {
            snprintf(m->err, sizeof m->err, "graph API missing");
            a->sub_destroy(sub);
            return -8;
        }

        midaSubgraphExecParams gep;
        memset(&gep, 0, sizeof gep);
        gep.l2bSize = (uint32_t) env_int("O3NPU_L2B", 0);
        gep.vaSize  = (uint32_t) env_int("O3NPU_VA", 0);

        midaSubgraphNodeParams np;
        memset(&np, 0, sizeof np);
        np.subgraph = sub;
        np.execPara = gep;

        midaGraph_t graph = NULL;
        midaGraphNode_t node = NULL;
        midaGraphExec_t gexec = NULL;
        int g1 = a->graph_create(&graph, m->dev);
        int g2 = (g1 == 0) ? a->graph_add_node(&node, graph, NULL, 0, &np) : g1;
        int g3 = (g2 == 0) ? a->graph_instantiate(&gexec, graph) : g2;
        if (m->trace) {
            printf("o3npu: graph create=%d add=%d instantiate=%d graph=%p node=%p exec=%p\n",
                   g1, g2, g3, graph, node, gexec);
        }
        if (g3 != 0 || !gexec) {
            snprintf(m->err, sizeof m->err, "midaGraph path -> %d/%d/%d", g1, g2, g3);
            a->sub_destroy(sub);
            return -9;
        }

        midaSubgraphExec_t sub_exec = NULL;
        if (env_int("O3NPU_GRAPH_SUB", 1)) {
            int g4 = a->graph_exec_get_sub(&sub_exec, gexec, node);
            if (m->trace) {
                printf("o3npu: graph_exec_get_sub -> %d (%p)\n", g4, sub_exec);
            }
        }

        midaLaunchConfig lc;
        memset(&lc, 0, sizeof lc);
        lc.cu_num = (uint32_t) env_int("O3NPU_CU", 1);
        lc.performance_level = (uint32_t) env_int("O3NPU_PERF", 0);
        lc.ddr_ratio = (float) env_int("O3NPU_DDR", 100) / 100.0f;
        lc.fw_timeout = (uint32_t) env_int("O3NPU_FW_TIMEOUT", 10000);
        lc.name = "o3npu";
        fill_stripe_names(&lc);

        midaLaunchConfig * lcp = env_int("O3NPU_LC_NULL", 0) ? NULL : &lc;
        rc = sub_exec ? a->stream_launch_subgraph_async(sub_exec, lcp, m->stream)
                      : a->stream_launch_graph_async(gexec, lcp, m->stream);
        if (rc != 0) {
            snprintf(m->err, sizeof m->err, "graph launch -> %s", err_str(a, rc));
            if (a->graph_exec_destroy) a->graph_exec_destroy(gexec);
            a->sub_destroy(sub);
            return -6;
        }

        rc = a->stream_sync(m->stream);
        if (rc != 0) {
            snprintf(m->err, sizeof m->err, "midaStreamSynchronize -> %s", err_str(a, rc));
            if (a->graph_exec_destroy) a->graph_exec_destroy(gexec);
            a->sub_destroy(sub);
            return -7;
        }

        dbuf_sync_cpu(a, &m->out);
        {
            const uint16_t * o = (const uint16_t *) m->out.cpu;
            for (size_t i = 0, n = (size_t) M * T; i < n; i++) {
                out_f32[i] = fp16_to_fp32(o[i]);
            }
        }
        dbuf_sync_dev(a, &m->out);

        if (a->graph_exec_destroy) a->graph_exec_destroy(gexec);
        a->sub_destroy(sub);
        return 0;
    }

    midaSubgraphExecParams ep;
    memset(&ep, 0, sizeof ep);
    ep.l2bSize = (uint32_t) env_int("O3NPU_L2B", 0);
    ep.vaSize  = (uint32_t) env_int("O3NPU_VA", 0);

    {
        uint64_t stack_sz = (uint64_t) env_int("O3NPU_STACK", 0);
        if (stack_sz) {
            int sr = dbuf_reserve(a, m->dev, &m->stack, stack_sz, "o3stack");
            if (m->trace) printf("o3npu: stack reserve %llu -> %d\n", (unsigned long long) stack_sz, sr);
            if (sr == 0) {
                ep.stackInfo.stack_size = stack_sz;
                ep.stackInfo.stack_mem  = m->stack.mem;
                if (m->trace) printf("o3npu: provided stack %llu bytes\n", (unsigned long long) stack_sz);
            }
        }
    }

    midaSubgraphExec_t exec = NULL;
    rc = a->sub_instantiate(&exec, sub, &ep);
    if (rc != 0 || !exec) {
        snprintf(m->err, sizeof m->err, "midaSubgraphInstantiate -> %s", err_str(a, rc));
        a->sub_destroy(sub);
        return -5;
    }

    /* ask the runtime how much stack the subgraph needs, then re-instantiate */
    if (a->sub_exec_stack_info) {
        midaSubgraphStackInfo si;
        memset(&si, 0, sizeof si);
        int sir = a->sub_exec_stack_info(exec, &si);
        if (m->trace) printf("o3npu: stack_info rc=%d size=%llu mem=%p\n", sir, (unsigned long long) si.stack_size, si.stack_mem);
        if (sir == 0 && si.stack_size > 0 && !si.stack_mem) {
            struct dbuf stack;
            memset(&stack, 0, sizeof stack);
            stack.fd = -1;
            if (dbuf_reserve(a, m->dev, &stack, si.stack_size, "o3stack") == 0) {
                ep.stackInfo.stack_size = si.stack_size;
                ep.stackInfo.stack_mem  = stack.mem;
                midaSubgraphExec_t exec2 = NULL;
                if (a->sub_instantiate(&exec2, sub, &ep) == 0 && exec2) {
                    a->sub_exec_destroy(exec);
                    exec = exec2;
                } else {
                    dbuf_free(a, &stack);
                }
            }
        }
    }

    if (m->trace && a->sub_print_info) {
        printf("o3npu: midaSubgraphPrintInfo -> %d\n", a->sub_print_info(sub));
    }
    if (m->trace && a->sub_exec_print_info) {
        printf("o3npu: midaSubgraphExecPrintInfo -> %d\n", a->sub_exec_print_info(exec));
    }
    if (a->sub_exec_dump) {
        const char * dump = getenv("O3NPU_DUMP");
        if (dump && *dump) {
            printf("o3npu: dump %s -> %d\n", dump, a->sub_exec_dump(exec, dump));
        }
    }

    midaLaunchConfig lc;
    memset(&lc, 0, sizeof lc);
    lc.cu_num = (uint32_t) env_int("O3NPU_CU", 1);
    lc.performance_level = (uint32_t) env_int("O3NPU_PERF", 0);
    lc.ddr_ratio = (float) env_int("O3NPU_DDR", 100) / 100.0f;
    lc.fw_timeout = (uint32_t) env_int("O3NPU_FW_TIMEOUT", 10000);
    lc.name = "o3npu";
    fill_stripe_names(&lc);

    midaLaunchConfig * lcp = env_int("O3NPU_LC_NULL", 0) ? NULL : &lc;
    rc = a->stream_launch_subgraph_async(exec, lcp, m->stream);
    if (rc != 0) {
        snprintf(m->err, sizeof m->err, "midaStreamLaunchSubgraphAsync -> %s", err_str(a, rc));
        a->sub_exec_destroy(exec);
        a->sub_destroy(sub);
        return -6;
    }

    rc = a->stream_sync(m->stream);
    if (rc != 0) {
        snprintf(m->err, sizeof m->err, "midaStreamSynchronize -> %s", err_str(a, rc));
        a->sub_exec_destroy(exec);
        a->sub_destroy(sub);
        return -7;
    }

    dbuf_sync_cpu(a, &m->out);
    {
        const uint16_t * o = (const uint16_t *) m->out.cpu;
        for (size_t i = 0, n = (size_t) M * T; i < n; i++) {
            out_f32[i] = fp16_to_fp32(o[i]);
        }
    }
    dbuf_sync_dev(a, &m->out);

    a->sub_exec_destroy(exec);
    a->sub_destroy(sub);
    return 0;
}


/* ------------------------------------------------------------------ */
/* minimal RMSNorm probe (does not touch the ggml backend)             */
/* ------------------------------------------------------------------ */

int o3npu_rmsnorm_test(int K, int trace) {
    struct mida_api * a = mida_get();
    if (!a) return -1;

    midaDeviceConfig cfg;
    fill_devcfg(&cfg);
    midaDevice_t dev = NULL;
    int rc = a->device_get(&dev, 0, &cfg);
    if (rc != 0 || !dev) { printf("rmsnorm device -> %d\n", rc); return -2; }

    midaStreamConfig sc;
    memset(&sc, 0, sizeof sc);
    sc.type = 1;
    midaStream_t st = NULL;
    rc = a->stream_create(&st, dev, &sc);
    if (rc != 0) { printf("rmsnorm stream -> %d\n", rc); return -3; }

    void * xh = NULL;
    if (o3xnnl_open(&xh, dev) != 0) return -4;

    struct dbuf in, w, out;
    memset(&in, 0, sizeof in);  in.fd = -1;
    memset(&w, 0, sizeof w);    w.fd = -1;
    memset(&out, 0, sizeof out); out.fd = -1;
    if (dbuf_reserve(a, dev, &in,  (uint64_t) K * 2, "rin")  != 0) return -5;
    if (dbuf_reserve(a, dev, &w,   (uint64_t) K * 2, "rw")   != 0) return -5;
    if (dbuf_reserve(a, dev, &out, (uint64_t) K * 2, "rout") != 0) return -5;

    uint16_t * pi = (uint16_t *) in.cpu;
    uint16_t * pw = (uint16_t *) w.cpu;
    for (int i = 0; i < K; i++) {
        pi[i] = 0x3C00;   /* 1.0 */
        pw[i] = 0x3C00;
    }
    a->mem_cache_end(in.mem);
    a->mem_cache_end(w.mem);
    a->mem_cache_end(out.mem);

    midaSubgraphConfig subcfg;
    memset(&subcfg, 0, sizeof subcfg);
    midaSubgraph_t sub = NULL;
    rc = a->sub_create(&sub, dev, &subcfg);
    if (rc != 0) { printf("rmsnorm subgraph -> %d\n", rc); return -6; }

    rc = o3xnnl_rmsnorm(xh, sub, K, (const void *) in.va, (const void *) w.va, (void *) out.va);
    printf("rmsnorm create -> %d\n", rc);
    if (rc != 0) return -7;

    midaSubgraphExecParams ep;
    memset(&ep, 0, sizeof ep);
    midaSubgraphExec_t ex = NULL;
    rc = a->sub_instantiate(&ex, sub, &ep);
    printf("rmsnorm instantiate -> %d\n", rc);
    if (rc != 0) return -8;

    midaSubgraphStackInfo si;
    memset(&si, 0, sizeof si);
    int sr = a->sub_exec_stack_info ? a->sub_exec_stack_info(ex, &si) : -1;
    printf("rmsnorm stack_info -> %d size=%llu\n", sr, (unsigned long long) si.stack_size);

    midaLaunchConfig lc;
    memset(&lc, 0, sizeof lc);
    lc.cu_num = 1;
    lc.ddr_ratio = 1.0f;
    lc.fw_timeout = 10000;
    lc.name = "o3rms";
    rc = a->stream_launch_subgraph_async(ex, &lc, st);
    printf("rmsnorm launch -> %d\n", rc);
    if (rc == 0) {
        rc = a->stream_sync(st);
        printf("rmsnorm sync -> %d (%s)\n", rc, err_str(a, rc));
    }
    if (rc == 0) {
        a->mem_cache_end(out.mem);
        const uint16_t * po = (const uint16_t *) out.cpu;
        int bad = 0;
        for (int i = 0; i < K; i++) {
            float v = o3npu_fp16_to_fp32(po[i]);
            if (fabsf(v - 1.0f) > 2e-3f) {
                if (bad < 4 && trace) printf("rmsnorm: out[%d]=%f expect 1.0\n", i, v);
                bad++;
            }
        }
        printf("rmsnorm check: %d/%d mismatch\n", bad, K);
        if (bad) rc = 1;
    }
    (void) trace;
    return rc;
}


int o3npu_limits(void) {
    struct mida_api * a = mida_get();
    if (!a || !a->device_get_limit) { printf("no device_get_limit\n"); return -1; }
    midaDeviceConfig cfg;
    fill_devcfg(&cfg);
    midaDevice_t dev = NULL;
    int rc = a->device_get(&dev, 0, &cfg);
    if (rc != 0 || !dev) { printf("device -> %d\n", rc); return -2; }
    const char * set = getenv("O3NPU_LIMIT_SET");
    if (set && *set && a->device_set_limit) {
        int idx = 0;
        unsigned long val = 0;
        if (sscanf(set, "%d:%lu", &idx, &val) == 2) {
            int r = a->device_set_limit(idx, (uint32_t) val);
            printf("set_limit[%d]=%lu -> %d\n", idx, val, r);
        }
    }
    for (int i = 0; i < 16; i++) {
        uint32_t v = 0;
        int r = a->device_get_limit(&v, i, dev);
        printf("limit[%d] rc=%d value=%u (0x%x)\n", i, r, v, v);
    }
    return 0;
}


int o3npu_unary_test(const char * name, int n, int trace) {
    struct mida_api * a = mida_get();
    if (!a) return -1;

    midaDeviceConfig cfg;
    fill_devcfg(&cfg);
    midaDevice_t dev = NULL;
    int rc = a->device_get(&dev, 0, &cfg);
    if (rc != 0 || !dev) { printf("unary device -> %d\n", rc); return -2; }
    apply_limit_env(a, dev);

    midaStreamConfig sc;
    memset(&sc, 0, sizeof sc);
    sc.type = 1;
    midaStream_t st = NULL;
    rc = a->stream_create(&st, dev, &sc);
    if (rc != 0) { printf("unary stream -> %d\n", rc); return -3; }

    void * xh = NULL;
    if (o3xnnl_open(&xh, dev) != 0) return -4;

    struct dbuf in, out;
    memset(&in, 0, sizeof in);   in.fd = -1;
    memset(&out, 0, sizeof out); out.fd = -1;
    if (dbuf_reserve(a, dev, &in,  (uint64_t) n * 2, "uin")  != 0) return -5;
    if (dbuf_reserve(a, dev, &out, (uint64_t) n * 2, "uout") != 0) return -5;

    uint16_t * pi = (uint16_t *) in.cpu;
    for (int i = 0; i < n; i++) {
        pi[i] = 0x3C00;  /* 1.0 */
    }
    a->mem_cache_end(in.mem);
    {
        uint16_t * po = (uint16_t *) out.cpu;
        for (int i = 0; i < n; i++) po[i] = 0x3800;  /* 0.5 */
    }
    a->mem_cache_end(out.mem);

    midaSubgraphConfig subcfg;
    memset(&subcfg, 0, sizeof subcfg);
    midaSubgraph_t sub = NULL;
    rc = a->sub_create(&sub, dev, &subcfg);
    if (rc != 0) { printf("unary subgraph -> %d\n", rc); return -6; }

    rc = o3xnnl_unary(xh, sub, name, n, (const void *) in.va, (void *) out.va);
    printf("unary %s create -> %d\n", name, rc);
    if (rc != 0) return -7;

    midaSubgraphExecParams ep;
    memset(&ep, 0, sizeof ep);
    ep.l2bSize = (uint32_t) env_int("O3NPU_L2B", 0);
    midaSubgraphExec_t ex = NULL;
    rc = a->sub_instantiate(&ex, sub, &ep);
    printf("unary %s instantiate -> %d (l2b=%u)\n", name, rc, ep.l2bSize);
    if (rc != 0) return -8;

    midaLaunchConfig lc;
    memset(&lc, 0, sizeof lc);
    lc.cu_num = 1;
    lc.ddr_ratio = 1.0f;
    lc.fw_timeout = 10000;
    lc.name = "o3un";
    rc = a->stream_launch_subgraph_async(ex, &lc, st);
    printf("unary %s launch -> %d\n", name, rc);
    if (rc != 0) return -9;

    rc = a->stream_sync(st);
    printf("unary %s sync -> %d (%s)\n", name, rc, err_str(a, rc));
    if (rc != 0) return -10;

    /* read back and compare against the expected value */
    a->mem_cache_end(out.mem);
    const uint16_t * po = (const uint16_t *) out.cpu;
    if (getenv("O3NPU_UNARY_INPLACE")) {
        a->mem_cache_end(in.mem);
        po = (const uint16_t *) in.cpu;
    }
    float expect = (strcmp(name, "neg") == 0) ? -1.0f : 1.0f;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float v = o3npu_fp16_to_fp32(po[i]);
        if (fabsf(v - expect) > 1e-3f) {
            if (bad < 4 && trace) {
                printf("unary %s: out[%d]=%f expect %f\n", name, i, v, expect);
            }
            bad++;
        }
    }
    if (trace && bad) {
        for (int c = 0; c < n; c += 64) {
            int cb = 0;
            int ce = c + 64 < n ? c + 64 : n;
            for (int i = c; i < ce; i++) {
                float v = o3npu_fp16_to_fp32(po[i]);
                if (fabsf(v - expect) > 1e-3f) cb++;
            }
            printf("unary %s chunk %d-%d: %d bad\n", name, c, ce - 1, cb);
        }
    }
    printf("unary %s check: %d/%d mismatch (in[0]=%f out[0]=%f)\n", name, bad, n,
           o3npu_fp16_to_fp32(((const uint16_t *) in.cpu)[0]),
           o3npu_fp16_to_fp32(po[0]));
    return bad == 0 ? 0 : 1;
}

/* exposed for the host unit test only */
float o3npu_fp16_to_fp32(uint16_t h) {
    return fp16_to_fp32(h);
}
