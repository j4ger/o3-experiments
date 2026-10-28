/*
 * Probe the VeriSilicon/Xiaomi NPU through the mida UMD (libxnumd.so).
 *
 * libOpenVX.so on this device is the Vivante GPU OpenVX (it crashes inside
 * libGAL gcoVX_*), so it is not the NPU door. The NPU is reached through
 * mida: libxnumd.so opens /dev/hpc-cdev and drives the VIP9000 through ioctls.
 *
 * midaDeviceConfig (size 0x1c, from libxnlm DWARF):
 *   +0x00 PerformanceMode performance
 *   +0x04 uint32 log_close
 *   +0x08 uint32 pu_num
 *   +0x0c uint32 mem_debug
 *   +0x10 uint32 cu_delay
 *   +0x14 bool   drm_protect
 *   +0x18 uint32 power_policy
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
typedef int (*fn_mem_usage)(void *);
typedef int (*fn_mem_alloc)(void **, void *, uint64_t, uint32_t, uint32_t, const char *);
typedef int (*fn_mem_free)(void *);
typedef void *(*fn_mem_ptr)(void *);
typedef uint64_t (*fn_mem_size)(void *);
typedef uint64_t (*fn_mem_va)(void *);
typedef int (*fn_version)(void);

int main(void) {
    const char *paths[] = {
        "/vendor/lib64/libxnumd.so",
        "libxnumd.so",
        NULL,
    };
    void *h = NULL;
    for (int i = 0; paths[i]; i++) {
        h = dlopen(paths[i], RTLD_NOW | RTLD_GLOBAL);
        printf("dlopen(%s) -> %s\n", paths[i], h ? "ok" : dlerror());
        if (h) {
            break;
        }
    }
    if (!h) {
        printf("RESULT dlopen_failed\n");
        return 2;
    }

    fn_device_get get = (fn_device_get)dlsym(h, "midaDeviceGet");
    fn_device_release release = (fn_device_release)dlsym(h, "midaDeviceRelease");
    fn_mem_usage usage = (fn_mem_usage)dlsym(h, "midaGetNPUMemoryUsage");
    fn_version version = (fn_version)dlsym(h, "midaGetVersion");
    printf("dlsym: get=%p release=%p usage=%p version=%p\n",
           (void *)get, (void *)release, (void *)usage, (void *)version);

    if (version) {
        printf("midaGetVersion() -> %d\n", version());
    }
    if (!get) {
        printf("RESULT no_midaDeviceGet\n");
        return 3;
    }

    struct midaDeviceConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    printf("sizeof(config)=%zu (expect 28)\n", sizeof cfg);

    void *dev = NULL;
    int err = get(&dev, 0, &cfg);
    printf("midaDeviceGet(type=0) -> err=%d dev=%p\n", err, dev);
    if (err != 0 || !dev) {
        printf("RESULT device_get_failed\n");
        return 4;
    }

    if (usage) {
        uint8_t buf[128];
        memset(buf, 0, sizeof buf);
        int e2 = usage(buf);
        printf("midaGetNPUMemoryUsage -> err=%d\n", e2);
        printf("usage bytes:");
        for (int i = 0; i < 32; i++) {
            printf(" %02x", buf[i]);
        }
        printf("\n");
    }

    fn_mem_alloc alloc = (fn_mem_alloc)dlsym(h, "midaMemoryAlloc");
    fn_mem_free mfree = (fn_mem_free)dlsym(h, "midaMemoryFree");
    fn_mem_ptr mptr = (fn_mem_ptr)dlsym(h, "midaMemoryGetPtr");
    fn_mem_size msize = (fn_mem_size)dlsym(h, "midaMemorySize");
    fn_mem_va mva = (fn_mem_va)dlsym(h, "midaMemoryGetDeviceVa");
    printf("dlsym: alloc=%p free=%p ptr=%p size=%p va=%p\n",
           (void *)alloc, (void *)mfree, (void *)mptr, (void *)msize, (void *)mva);

    if (alloc && mfree && mptr) {
        void *mem = NULL;
        int ea = alloc(&mem, dev, 4096, 0, 0, "probe");
        printf("midaMemoryAlloc(4096) -> err=%d mem=%p\n", ea, mem);
        if (ea == 0 && mem) {
            void *ptr = mptr(mem);
            printf("  ptr=%p size=%llu devva=", ptr,
                   (unsigned long long)(msize ? msize(mem) : 0));
            if (mva) {
                printf("0x%llx", (unsigned long long)mva(mem));
            }
            printf("\n");
            if (ptr) {
                memset(ptr, 0xAB, 64);
                int ok = 1;
                for (int i = 0; i < 64; i++) {
                    if (((unsigned char *)ptr)[i] != 0xAB) {
                        ok = 0;
                        break;
                    }
                }
                printf("  write/readback %s\n", ok ? "OK" : "MISMATCH");
            }
            if (usage) {
                uint8_t buf2[256];
                memset(buf2, 0, sizeof buf2);
                printf("midaGetNPUMemoryUsage(after alloc) -> err=%d\n", usage(buf2));
                for (int i = 0; i < 256; i += 4) {
                    uint32_t v;
                    memcpy(&v, buf2 + i, 4);
                    if (v) {
                        printf("  usage+0x%02x = %u (0x%x)\n", i, v, v);
                    }
                }
            }
            printf("midaMemoryFree -> err=%d\n", mfree(mem));
        }
    }

    if (release) {
        int e3 = release(dev);
        printf("midaDeviceRelease -> err=%d\n", e3);
    }
    printf("RESULT ok\n");
    return 0;
}
