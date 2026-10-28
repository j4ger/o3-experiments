/*
 * Work out how to obtain a CPU-accessible pointer and device VA for NPU memory.
 *
 * midaMemoryAlloc() gives an NPU-side allocation whose midaMemoryGetPtr() and
 * midaMemoryGetDeviceVa() are both 0. The UMD also exposes:
 *
 *   int  midaMemoryGetFd(midaMemory_t)
 *   int  midaMemoryMap(midaMemory_t *out, midaDevice_t dev, int fd, void *addr,
 *                      uint64_t size, uint32_t, uint32_t)
 *   int  midaMemoryMapWithAddr(midaMemory_t *out, int fd, uint64_t, uint64_t,
 *                              midaAddress_t)
 *
 * This probe tries GetFd then Map, and reports what each returns.
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
typedef int (*fn_alloc)(void **, void *, uint64_t, uint32_t, uint32_t, const char *);
typedef int (*fn_free)(void *);
typedef void *(*fn_ptr)(void *);
typedef uint64_t (*fn_u64)(void *);
typedef int (*fn_getfd)(void *);
typedef int (*fn_map)(void **, void *, int, void *, uint64_t, uint32_t, uint32_t);
typedef int (*fn_mapaddr)(void **, int, uint64_t, uint64_t, void *);
typedef int (*fn_cache)(void *);

int main(void) {
    void *h = dlopen("/vendor/lib64/libxnumd.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 2;
    }
    fn_device_get get = (fn_device_get)dlsym(h, "midaDeviceGet");
    fn_device_release release = (fn_device_release)dlsym(h, "midaDeviceRelease");
    fn_alloc alloc = (fn_alloc)dlsym(h, "midaMemoryAlloc");
    fn_free mfree = (fn_free)dlsym(h, "midaMemoryFree");
    fn_ptr mptr = (fn_ptr)dlsym(h, "midaMemoryGetPtr");
    fn_u64 msize = (fn_u64)dlsym(h, "midaMemorySize");
    fn_u64 mva = (fn_u64)dlsym(h, "midaMemoryGetDeviceVa");
    fn_getfd getfd = (fn_getfd)dlsym(h, "midaMemoryGetFd");
    fn_map mmap_ = (fn_map)dlsym(h, "midaMemoryMap");
    fn_mapaddr mmapaddr = (fn_mapaddr)dlsym(h, "midaMemoryMapWithAddr");
    fn_cache cstart = (fn_cache)dlsym(h, "midaMemoryCacheStart");
    fn_cache cend = (fn_cache)dlsym(h, "midaMemoryCacheEnd");

    printf("syms: getfd=%p map=%p mapaddr=%p cache=%p/%p\n",
           (void *)getfd, (void *)mmap_, (void *)mmapaddr, (void *)cstart, (void *)cend);

    struct midaDeviceConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    void *dev = NULL;
    int e = get(&dev, 0, &cfg);
    printf("midaDeviceGet -> %d dev=%p\n", e, dev);
    if (e || !dev) {
        return 3;
    }

    /* try a few alloc flag combinations */
    uint32_t flags[][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 0}};
    for (unsigned i = 0; i < sizeof(flags) / sizeof(flags[0]); i++) {
        void *mem = NULL;
        int ea = alloc(&mem, dev, 4096, flags[i][0], flags[i][1], "probe2");
        if (ea || !mem) {
            printf("alloc(%u,%u) -> err=%d\n", flags[i][0], flags[i][1], ea);
            continue;
        }
        void *ptr = mptr ? mptr(mem) : NULL;
        uint64_t va = mva ? mva(mem) : 0;
        int fd = getfd ? getfd(mem) : -2;
        printf("alloc(%u,%u) -> mem=%p ptr=%p va=0x%llx fd=%d\n",
               flags[i][0], flags[i][1], mem, ptr,
               (unsigned long long)va, fd);

        if (!ptr && mmap_ && fd >= 0) {
            void *m2 = NULL;
            int em = mmap_(&m2, dev, fd, NULL, 4096, 0, 0);
            printf("  Map(fd=%d) -> err=%d m2=%p", fd, em, m2);
            if (em == 0 && m2) {
                printf(" ptr=%p va=0x%llx size=%llu",
                       mptr ? mptr(m2) : NULL,
                       (unsigned long long)(mva ? mva(m2) : 0),
                       (unsigned long long)(msize ? msize(m2) : 0));
            }
            printf("\n");
        }
        mfree(mem);
    }

    release(dev);
    printf("done\n");
    return 0;
}
