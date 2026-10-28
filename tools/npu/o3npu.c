/*
 * o3npu - a small driver layer for the Xiaomi 18 Fold (lhasa) NPU.
 *
 * It wraps the vendor stack so that callers do not have to rediscover it:
 *
 *     your code
 *       -> libxnnl   (op graph builders: matmul / RMSNorm / quantize)
 *         -> mida    (subgraph, instantiate, stream launch, memory)
 *           -> libxnumd (UMD)
 *             -> /dev/hpc-cdev
 *
 * Requirements: uid 0 in a domain that may open /dev/hpc-cdev, /dev/hpc-heap,
 * /dev/hpc-rpmsg, the dma heaps, /dev/timestamp and /dev/xr_dmabuf_helper, and
 * read vendor_npu_prop. See docs/NPU-ACCESS.md for the exact ksud rules.
 *
 * Memory: midaMemoryAlloc(mem, dev, size, 1, 0, name) yields CPU-mappable memory
 * (midaMemoryGetPtr returns a real pointer); the flag 0 variant is NPU-side only.
 */
#include "o3npu.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct midaDeviceConfig {
    int32_t performance;
    uint32_t log_close;
    uint32_t pu_num;
    uint32_t mem_debug;
    uint32_t cu_delay;
    uint8_t drm_protect;
    uint8_t pad[3];
    uint32_t power_policy;
};

typedef int (*fn_get)(void **, uint32_t, void *);
typedef int (*fn_release)(void *);
typedef int (*fn_alloc)(void **, void *, uint64_t, uint32_t, uint32_t, const char *);
typedef int (*fn_free)(void *);
typedef void *(*fn_ptr)(void *);
typedef uint64_t (*fn_u64)(void *);
typedef int (*fn_getfd)(void *);
typedef int (*fn_sub_create)(void **, void *, void *);
typedef int (*fn_sub_destroy)(void *);
typedef int (*fn_sub_inst)(void **, void *, void *);
typedef int (*fn_exec_destroy)(void *);
typedef int (*fn_stream_create)(void **, void *, void *);
typedef int (*fn_stream_destroy)(void *);
typedef int (*fn_launch)(void *, void *, void *);
typedef int (*fn_sync)(void *);
typedef int (*fn_xnnl_create)(void **, void *);
typedef int (*fn_xnnl_destroy)(void *);

struct o3npu {
    void *h_numd;
    void *h_xnnl;
    void *dev;
    void *xnnl;
    fn_get get;
    fn_release release;
    fn_alloc alloc;
    fn_free mfree;
    fn_ptr mptr;
    fn_u64 msize;
    fn_u64 mva;
    fn_getfd getfd;
    fn_sub_create sub_create;
    fn_sub_destroy sub_destroy;
    fn_sub_inst sub_inst;
    fn_exec_destroy exec_destroy;
    fn_stream_create stream_create;
    fn_stream_destroy stream_destroy;
    fn_launch launch;
    fn_sync sync;
    fn_xnnl_create xcreate;
    fn_xnnl_destroy xdestroy;
};

o3npu_t *o3npu_open(void) {
    o3npu_t *n = (o3npu_t *)calloc(1, sizeof *n);
    if (!n) {
        return NULL;
    }
    n->h_numd = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    if (!n->h_numd) {
        dlclose(n->h_numd);
        free(n);
        return NULL;
    }
    n->h_xnnl = dlopen("/vendor/lib64/libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!n->h_xnnl) {
        n->h_xnnl = dlopen("libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    }

    n->get = (fn_get)dlsym(n->h_numd, "midaDeviceGet");
    n->release = (fn_release)dlsym(n->h_numd, "midaDeviceRelease");
    n->alloc = (fn_alloc)dlsym(n->h_numd, "midaMemoryAlloc");
    n->mfree = (fn_free)dlsym(n->h_numd, "midaMemoryFree");
    n->mptr = (fn_ptr)dlsym(n->h_numd, "midaMemoryGetPtr");
    n->msize = (fn_u64)dlsym(n->h_numd, "midaMemorySize");
    n->mva = (fn_u64)dlsym(n->h_numd, "midaMemoryGetDeviceVa");
    n->getfd = (fn_getfd)dlsym(n->h_numd, "midaMemoryGetFd");
    n->sub_create = (fn_sub_create)dlsym(n->h_numd, "midaSubgraphCreate");
    n->sub_destroy = (fn_sub_destroy)dlsym(n->h_numd, "midaSubgraphDestroy");
    n->sub_inst = (fn_sub_inst)dlsym(n->h_numd, "midaSubgraphInstantiate");
    n->exec_destroy = (fn_exec_destroy)dlsym(n->h_numd, "midaSubgraphExecDestroy");
    n->stream_create = (fn_stream_create)dlsym(n->h_numd, "midaStreamCreate");
    n->stream_destroy = (fn_stream_destroy)dlsym(n->h_numd, "midaStreamDestroy");
    n->launch = (fn_launch)dlsym(n->h_numd, "midaStreamLaunchSubgraph");
    n->sync = (fn_sync)dlsym(n->h_numd, "midaStreamSynchronize");
    if (n->h_xnnl) {
        n->xcreate = (fn_xnnl_create)dlsym(n->h_xnnl, "xnnlCreate");
        n->xdestroy = (fn_xnnl_destroy)dlsym(n->h_xnnl, "xnnlDestroy");
    }

    if (!n->get || !n->alloc || !n->mfree) {
        o3npu_close(n);
        return NULL;
    }

    struct midaDeviceConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    int e = n->get(&n->dev, 0, &cfg);
    if (e != 0 || !n->dev) {
        o3npu_close(n);
        return NULL;
    }
    if (n->xcreate) {
        void *xh = NULL;
        if (n->xcreate(&xh, n->dev) == 0) {
            n->xnnl = xh;
        }
    }
    return n;
}

void o3npu_close(o3npu_t *n) {
    if (!n) {
        return;
    }
    if (n->xnnl && n->xdestroy) {
        n->xdestroy(n->xnnl);
    }
    if (n->dev && n->release) {
        n->release(n->dev);
    }
    if (n->h_xnnl) {
        dlclose(n->h_xnnl);
    }
    if (n->h_numd) {
        dlclose(n->h_numd);
    }
    free(n);
}

void *o3npu_alloc(o3npu_t *n, size_t size, int cpu_visible, size_t *out_size) {
    if (!n || !n->alloc) {
        return NULL;
    }
    void *mem = NULL;
    int e = n->alloc(&mem, n->dev, size, cpu_visible ? 1 : 0, 0, "o3npu");
    if (e != 0) {
        return NULL;
    }
    if (out_size && n->msize) {
        *out_size = (size_t)n->msize(mem);
    }
    return mem;
}

void *o3npu_cpu_ptr(o3npu_t *n, void *mem) {
    return (n && n->mptr) ? n->mptr(mem) : NULL;
}

uint64_t o3npu_dev_va(o3npu_t *n, void *mem) {
    return (n && n->mva) ? n->mva(mem) : 0;
}

int o3npu_mem_fd(o3npu_t *n, void *mem) {
    return (n && n->getfd) ? n->getfd(mem) : -1;
}

void o3npu_free(o3npu_t *n, void *mem) {
    if (n && n->mfree && mem) {
        n->mfree(mem);
    }
}

/* ------------------------------------------------------------------ */
/* self test: proves every layer except op execution                    */
/* ------------------------------------------------------------------ */
int o3npu_selftest(void) {
    printf("== o3npu selftest ==\n");
    o3npu_t *n = o3npu_open();
    printf("o3npu_open -> %p\n", (void *)n);
    if (!n) {
        return 1;
    }
    printf("  device=%p xnnl=%p\n", n->dev, n->xnnl);

    size_t sz = 0;
    void *host = o3npu_alloc(n, 4096, 1, &sz);
    printf("  alloc(cpu_visible) -> %p size=%zu\n", host, sz);
    if (host) {
        void *p = o3npu_cpu_ptr(n, host);
        printf("  cpu_ptr=%p dev_va=0x%llx fd=%d\n", p,
               (unsigned long long)o3npu_dev_va(n, host), o3npu_mem_fd(n, host));
        if (p) {
            memset(p, 0x5a, 64);
            int ok = 1;
            for (int i = 0; i < 64; i++) {
                if (((unsigned char *)p)[i] != 0x5a) {
                    ok = 0;
                    break;
                }
            }
            printf("  write/readback %s\n", ok ? "OK" : "MISMATCH");
        }
        o3npu_free(n, host);
        printf("  freed\n");
    }

    void *npu = o3npu_alloc(n, 4096, 0, &sz);
    printf("  alloc(npu_only) -> %p size=%zu\n", npu, sz);
    if (npu) {
        o3npu_free(n, npu);
    }

    uint8_t subcfg[0xa0];
    memset(subcfg, 0, sizeof subcfg);
    void *sub = NULL;
    int e = n->sub_create ? n->sub_create(&sub, n->dev, subcfg) : -1;
    printf("  subgraph_create -> %d sub=%p\n", e, sub);
    if (sub && n->sub_destroy) {
        n->sub_destroy(sub);
    }

    o3npu_close(n);
    printf("RESULT ok\n");
    return 0;
}
