/*
 * Exercise the whole NPU execution path with an empty subgraph:
 *
 *   midaDeviceGet -> midaSubgraphCreate -> midaSubgraphInstantiate
 *   -> midaStreamCreate -> midaStreamLaunchSubgraph -> midaStreamSynchronize
 *
 * If this succeeds the driver skeleton is complete and ops (xnnlMatmulCreateGraph,
 * xnnlRMSNormalizationCreateGraph, ...) can be added one at a time.
 *
 * Struct sizes from the recovered headers:
 *   midaSubgraphConfig   ~0xa0 with flags at 0x90/0x91 (all zero is accepted)
 *   midaSubgraphExecParams 0x20
 *   midaLaunchConfig     0xb8
 *   midaStreamConfig     0x0c
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
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
typedef int (*fn_sub_create)(void **, void *, void *);
typedef int (*fn_sub_destroy)(void *);
typedef int (*fn_sub_instantiate)(void **, void *, void *);
typedef int (*fn_exec_destroy)(void *);
typedef int (*fn_stream_create)(void **, void *, void *);
typedef int (*fn_stream_destroy)(void *);
typedef int (*fn_launch)(void *, void *, void *);
typedef int (*fn_sync)(void *);

int main(void) {
    void *h = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 2;
    }
    fn_get get = (fn_get)dlsym(h, "midaDeviceGet");
    fn_release release = (fn_release)dlsym(h, "midaDeviceRelease");
    fn_sub_create sub_create = (fn_sub_create)dlsym(h, "midaSubgraphCreate");
    fn_sub_destroy sub_destroy = (fn_sub_destroy)dlsym(h, "midaSubgraphDestroy");
    fn_sub_instantiate sub_inst = (fn_sub_instantiate)dlsym(h, "midaSubgraphInstantiate");
    fn_exec_destroy exec_destroy = (fn_exec_destroy)dlsym(h, "midaSubgraphExecDestroy");
    fn_stream_create stream_create = (fn_stream_create)dlsym(h, "midaStreamCreate");
    fn_stream_destroy stream_destroy = (fn_stream_destroy)dlsym(h, "midaStreamDestroy");
    fn_launch launch = (fn_launch)dlsym(h, "midaStreamLaunchSubgraph");
    fn_sync sync = (fn_sync)dlsym(h, "midaStreamSynchronize");

    printf("syms: sub=%p inst=%p stream=%p launch=%p sync=%p\n",
           (void *)sub_create, (void *)sub_inst, (void *)stream_create,
           (void *)launch, (void *)sync);
    if (!get || !sub_create || !sub_inst || !stream_create || !launch) {
        printf("RESULT missing_symbols\n");
        return 3;
    }

    struct midaDeviceConfig dcfg;
    memset(&dcfg, 0, sizeof dcfg);
    void *dev = NULL;
    int e = get(&dev, 0, &dcfg);
    printf("midaDeviceGet -> %d dev=%p\n", e, dev);
    if (e || !dev) {
        return 4;
    }

    uint8_t subcfg[0xa0];
    memset(subcfg, 0, sizeof subcfg);
    void *sub = NULL;
    e = sub_create(&sub, dev, subcfg);
    printf("midaSubgraphCreate -> %d sub=%p\n", e, sub);
    if (e || !sub) {
        release(dev);
        return 5;
    }

    uint8_t params[0x20];
    memset(params, 0, sizeof params);
    void *exec = NULL;
    e = sub_inst(&exec, sub, params);
    printf("midaSubgraphInstantiate -> %d exec=%p\n", e, exec);
    if (e || !exec) {
        if (sub_destroy) sub_destroy(sub);
        release(dev);
        return 6;
    }

    uint8_t scfg[0x0c];
    memset(scfg, 0, sizeof scfg);
    void *stream = NULL;
    e = stream_create(&stream, dev, scfg);
    printf("midaStreamCreate -> %d stream=%p\n", e, stream);
    if (e || !stream) {
        if (exec_destroy) exec_destroy(exec);
        if (sub_destroy) sub_destroy(sub);
        release(dev);
        return 7;
    }

    uint8_t lcfg[0xb8];
    memset(lcfg, 0, sizeof lcfg);
    e = launch(exec, lcfg, stream);
    printf("midaStreamLaunchSubgraph -> %d\n", e);
    if (sync) {
        int s = sync(stream);
        printf("midaStreamSynchronize -> %d\n", s);
    }

    if (stream_destroy) stream_destroy(stream);
    if (exec_destroy) exec_destroy(exec);
    if (sub_destroy) sub_destroy(sub);
    release(dev);
    printf("RESULT %s\n", e == 0 ? "ok" : "launch_failed");
    return 0;
}
