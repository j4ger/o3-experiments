/* probe_vx.c - test whether an unprivileged shell process can reach the
 * Vivante/VeriSilicon NPU through OpenVX (vxCreateContext) and enumerate OpenCL.
 * Build: aarch64-linux-android21-clang probe_vx.c -o probe_vx -ldl
 */
#include <stdio.h>
#include <dlfcn.h>

typedef void* (*fn_vxCreateContext)(void);
typedef int   (*fn_vxGetStatus)(void*);
typedef void* (*fn_vxQueryContext)(void);
typedef int   (*fn_clGetPlatformIDs)(unsigned, void**, unsigned*);
typedef int   (*fn_clGetDeviceIDs)(void*, unsigned long, unsigned, void**, unsigned*);
typedef int   (*fn_clGetDeviceInfo)(void*, unsigned, size_t, void*, size_t*);

static void try_openvx(void) {
    printf("\n== OpenVX (NPU) ==\n");
    void *h = dlopen("libOpenVX.so", RTLD_NOW);
    if (!h) h = dlopen("/vendor/lib64/libOpenVX.so", RTLD_NOW);
    printf("dlopen(libOpenVX.so) -> %s\n", h ? "ok" : dlerror());
    if (!h) return;

    fn_vxCreateContext create = (fn_vxCreateContext)dlsym(h, "vxCreateContext");
    fn_vxGetStatus status = (fn_vxGetStatus)dlsym(h, "vxGetStatus");
    printf("dlsym: vxCreateContext=%p vxGetStatus=%p\n", (void*)create, (void*)status);
    if (!create || !status) return;

    void *ctx = create();
    printf("vxCreateContext() -> %p, status=%d\n", ctx, ctx ? status(ctx) : -1);
    if (ctx && status(ctx) == 0) {
        /* VX_CONTEXT_VENDOR_ID == 0x0009 in OpenVX 1.3 */
        typedef int (*fn_query)(void*, int, size_t, void*);
        fn_query q = (fn_query)dlsym(h, "vxQueryContext");
        if (q) {
            char buf[256] = {0};
            unsigned id = 0;
            q(ctx, 0x0009, sizeof(buf), buf);
            q(ctx, 0x0008, sizeof(id), &id);   /* VX_CONTEXT_VERSION */
            printf("context vendor=%s version=0x%x\n", buf, id);
        }
    }
}

static void try_opencl(void) {
    printf("\n== OpenCL ==\n");
    void *h = dlopen("libOpenCL.so", RTLD_NOW);
    printf("dlopen(libOpenCL.so) -> %s\n", h ? "ok" : dlerror());
    if (!h) return;
    fn_clGetPlatformIDs gp = (fn_clGetPlatformIDs)dlsym(h, "clGetPlatformIDs");
    fn_clGetDeviceIDs gd = (fn_clGetDeviceIDs)dlsym(h, "clGetDeviceIDs");
    fn_clGetDeviceInfo gi = (fn_clGetDeviceInfo)dlsym(h, "clGetDeviceInfo");
    if (!gp || !gd || !gi) { printf("missing cl symbols\n"); return; }
    void *plats[8] = {0}; unsigned np = 0;
    printf("clGetPlatformIDs err=%d n=%u\n", gp(8, plats, &np), np);
    for (unsigned p = 0; p < np; p++) {
        void *devs[16] = {0}; unsigned nd = 0;
        int e = gd(plats[p], 0xFFFFFFFF, 16, devs, &nd);
        printf("  platform %u: getDeviceIDs err=%d n=%u\n", p, e, nd);
        for (unsigned d = 0; d < nd; d++) {
            char nm[256] = {0}, vn[256] = {0}, ver[256] = {0};
            gi(devs[d], 0x102B, sizeof(nm), nm, NULL);
            gi(devs[d], 0x102C, sizeof(vn), vn, NULL);
            gi(devs[d], 0x102F, sizeof(ver), ver, NULL);
            printf("    dev %u: %s | %s | %s\n", d, nm, vn, ver);
        }
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    try_openvx();
    try_opencl();
    return 0;
}
