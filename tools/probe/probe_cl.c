/* probe_cl.c - enumerate OpenCL devices on the device via dlopen (no link-time dep).
 * Build: aarch64-linux-android21-clang probe_cl.c -o probe_cl -ldl
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_TYPE_CPU 2
#define CL_DEVICE_TYPE_GPU 4
#define CL_DEVICE_TYPE_ACCELERATOR 8
#define CL_DEVICE_TYPE_CUSTOM 16
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF

typedef int (*fn_ids)(unsigned, void**, unsigned*);
typedef int (*fn_dev)(void*, unsigned long, unsigned, void**, unsigned*);
typedef int (*fn_info)(void*, unsigned, size_t, void*, size_t*);

static const char *tname(unsigned long t) {
    switch (t) {
        case CL_DEVICE_TYPE_CPU: return "CPU";
        case CL_DEVICE_TYPE_GPU: return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "ACCELERATOR";
        case CL_DEVICE_TYPE_CUSTOM: return "CUSTOM";
        default: return "other";
    }
}

int main(void) {
    const char *libs[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", NULL};
    void *h = NULL;
    for (int i = 0; libs[i]; i++) {
        h = dlopen(libs[i], RTLD_NOW);
        printf("dlopen(%s) -> %s\n", libs[i], h ? "ok" : dlerror());
        if (h) break;
    }
    if (!h) return 1;

    fn_ids getPlatformIDs = (fn_ids)dlsym(h, "clGetPlatformIDs");
    fn_dev getDeviceIDs = (fn_dev)dlsym(h, "clGetDeviceIDs");
    fn_info getDeviceInfo = (fn_info)dlsym(h, "clGetDeviceInfo");
    printf("dlsym: getPlatformIDs=%p getDeviceIDs=%p getDeviceInfo=%p\n",
           (void*)getPlatformIDs, (void*)getDeviceIDs, (void*)getDeviceInfo);
    if (!getPlatformIDs || !getDeviceIDs || !getDeviceInfo) return 2;

    void *plats[8] = {0};
    unsigned np = 0;
    int e = getPlatformIDs(8, plats, &np);
    printf("clGetPlatformIDs err=%d platforms=%u\n", e, np);

    for (unsigned p = 0; p < np; p++) {
        char buf[512];
        size_t n = 0;
        buf[0] = 0;
        getDeviceInfo(plats[p], 0x0900 /*CL_PLATFORM_NAME*/, sizeof(buf), buf, &n);
        printf("platform[%u] name=%s\n", p, buf);
        void *devs[16] = {0};
        unsigned nd = 0;
        int e2 = getDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 16, devs, &nd);
        printf("  clGetDeviceIDs err=%d devices=%u\n", e2, nd);
        for (unsigned d = 0; d < nd; d++) {
            unsigned long type = 0;
            unsigned cu = 0;
            unsigned long gm = 0;
            char nm[256] = {0}, vn[256] = {0}, ver[256] = {0}, ext[1024] = {0};
            getDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            getDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
            getDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gm), &gm, NULL);
            getDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(nm), nm, NULL);
            getDeviceInfo(devs[d], CL_DEVICE_VENDOR, sizeof(vn), vn, NULL);
            getDeviceInfo(devs[d], CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
            getDeviceInfo(devs[d], CL_DEVICE_EXTENSIONS, sizeof(ext), ext, NULL);
            printf("    device[%u] type=%s(%lu) cu=%u gmem=%.1fMB name=%s vendor=%s ver=%s\n",
                   d, tname(type), type, cu, (double)gm / (1024.0 * 1024.0), nm, vn, ver);
            printf("      ext=%s\n", ext);
        }
    }
    return 0;
}
