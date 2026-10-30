/*
 * probe_bell - run a vendor .bell plan through libxnnapi's XRingXnnApi.
 *
 * This is the vendor's own path (libxnnapi -> libxnn_runtime -> libxninfer ->
 * libxnumd) and is only used to answer one question: can this device execute
 * an NPU plan at all?  It is not part of the ggml backend.
 *
 * Build: tools/probe/build.sh (adds this file when invoked as probe_bell)
 *
 * Usage: probe_bell [/data/local/tmp/sample.bell]
 */
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/data/local/tmp/sample.bell";

    void * h = dlopen("/vendor/lib64/libxnnapi.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        printf("dlopen libxnnapi: %s\n", dlerror());
        return 2;
    }

    auto ctor = (void (*)(void *)) dlsym(h, "_ZN10xvpuxnnapi11XRingXnnApiC1Ev");
    auto dtor = (void (*)(void *)) dlsym(h, "_ZN10xvpuxnnapi11XRingXnnApiD1Ev");
    auto init = (int (*)(void *, const std::string &, int)) dlsym(
        h, "_ZN10xvpuxnnapi11XRingXnnApi10initVppXnnERKNSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEE19Xnn_PerformanceMode");
    auto run = (int (*)(void *)) dlsym(h, "_ZN10xvpuxnnapi11XRingXnnApi3RunEv");
    auto initbuf = (int (*)(void *)) dlsym(h, "_ZN10xvpuxnnapi11XRingXnnApi16initTensorBufferEv");

    printf("syms ctor=%p init=%p run=%p initbuf=%p\n",
           (void *) ctor, (void *) init, (void *) run, (void *) initbuf);
    if (!ctor || !init) {
        printf("RESULT missing_symbols\n");
        return 3;
    }

    void * obj = nullptr;
    if (posix_memalign(&obj, 64, 65536) != 0 || !obj) {
        return 4;
    }
    memset(obj, 0, 65536);
    ctor(obj);

    std::string p(path);
    int rc = init(obj, p, 0);
    printf("initVppXnn(%s) -> %d\n", path, rc);

    if (rc == 0) {
        if (initbuf) {
            printf("initTensorBuffer -> %d\n", initbuf(obj));
        }
        if (run) {
            printf("Run -> %d\n", run(obj));
        }
    }

    if (dtor) {
        dtor(obj);
    }
    free(obj);
    printf("RESULT %s\n", rc == 0 ? "ok" : "init_failed");
    return rc == 0 ? 0 : 5;
}
