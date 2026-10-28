/*
 * Check whether the vendor kernel library libxnnl can be created on top of a
 * mida device from the shell domain.
 *
 * Stack this validates:
 *     libxnnl (matmul / RMSNorm / quantize graph builders)
 *       -> mida  (graph, subgraph, memory, exec)
 *         -> libxnumd (UMD)
 *           -> /dev/hpc-cdev
 *
 * xnnlStatus_t values (from the recovered header):
 *   0 SUCCESS, 1 NOT_INITIALIZED, 2 ALLOC_FAILED, 3 BAD_PARAM, 4 INTERNAL_ERROR,
 *   5 INVALID_HANDLE, 6 NOT_SUPPORTED, 7 ...
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

typedef int (*fn_device_get)(void **, uint32_t, struct midaDeviceConfig *);
typedef int (*fn_device_release)(void *);
typedef int (*fn_xnnl_create)(void **, void *);
typedef int (*fn_xnnl_destroy)(void *);

int main(void) {
    void *hnumd = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    printf("dlopen(libxnumd) -> %s\n", hnumd ? "ok" : dlerror());
    if (!hnumd) {
        return 2;
    }

    void *hx = dlopen("/vendor/lib64/libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!hx) {
        printf("dlopen(/vendor/lib64/libxnnl.so) -> %s\n", dlerror());
        hx = dlopen("libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
        printf("dlopen(libxnnl.so) -> %s\n", hx ? "ok" : dlerror());
    } else {
        printf("dlopen(/vendor/lib64/libxnnl.so) -> ok\n");
    }
    if (!hx) {
        return 3;
    }

    fn_device_get get = (fn_device_get)dlsym(hnumd, "midaDeviceGet");
    fn_device_release release = (fn_device_release)dlsym(hnumd, "midaDeviceRelease");
    fn_xnnl_create xcreate = (fn_xnnl_create)dlsym(hx, "xnnlCreate");
    fn_xnnl_destroy xdestroy = (fn_xnnl_destroy)dlsym(hx, "xnnlDestroy");
    printf("dlsym: midaDeviceGet=%p xnnlCreate=%p xnnlDestroy=%p\n",
           (void *)get, (void *)xcreate, (void *)xdestroy);
    if (!get || !xcreate) {
        printf("RESULT missing_symbols\n");
        return 4;
    }

    struct midaDeviceConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    void *dev = NULL;
    int e = get(&dev, 0, &cfg);
    printf("midaDeviceGet -> err=%d dev=%p\n", e, dev);
    if (e || !dev) {
        return 5;
    }

    void *xh = NULL;
    int r = xcreate(&xh, dev);
    printf("xnnlCreate(dev) -> status=%d handle=%p\n", r, xh);
    if (r == 0 && xh && xdestroy) {
        int r2 = xdestroy(xh);
        printf("xnnlDestroy -> status=%d\n", r2);
    }
    release(dev);
    printf("RESULT %s\n", r == 0 ? "ok" : "create_failed");
    return 0;
}
