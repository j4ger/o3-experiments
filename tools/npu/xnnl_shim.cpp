/*
 * xnnl_shim.cpp - the C++-only part of libxnnl.
 *
 * xnnlSetTensorDescriptor takes a std::string by value, so it cannot be called
 * from C.  Everything else here is a thin dlsym wrapper; the goal is only to
 * keep tools/npu/o3npu.c free of C++.
 */
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

extern "C" {

typedef void * xnnlHandle_t;
typedef void * xnnlTensorDescriptor_t;
typedef void * xnnlEpilogueDescriptor_t;

enum {
    XNNL_DATA_HALF = 1,
    XNNL_TENSOR_NHWC = 2,
    XNNL_MEM_DDR = 0,
};

struct o3xnnl {
    void * lib;
    xnnlHandle_t handle;
    int (*create)(xnnlHandle_t *, void *);
    int (*destroy)(xnnlHandle_t);
    int (*create_desc)(xnnlTensorDescriptor_t *);
    int (*destroy_desc)(xnnlTensorDescriptor_t);
    int (*set_desc)(xnnlTensorDescriptor_t, const int *, unsigned, int, int, int, std::string);
    int (*create_epl)(xnnlEpilogueDescriptor_t *);
    int (*destroy_epl)(xnnlEpilogueDescriptor_t);
    int (*set_epl)(xnnlEpilogueDescriptor_t, int, int, int, int, int, int, int, int, int, int);
    int (*matmul)(xnnlHandle_t, void *, xnnlTensorDescriptor_t, const void *,
                  xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, const void *,
                  const void *, const void *, xnnlTensorDescriptor_t, void *,
                  xnnlEpilogueDescriptor_t, bool, uint32_t);
};

static int sym(void * lib, const char * name, void ** out) {
    *out = dlsym(lib, name);
    if (!*out) {
        fprintf(stderr, "xnnl_shim: missing symbol %s\n", name);
        return -1;
    }
    return 0;
}

static int make_desc(struct o3xnnl * s, xnnlTensorDescriptor_t * d, const int dims[4],
                     const char * name) {
    if (s->create_desc(d) != 0) {
        return -2;
    }
    std::string n(name);
    int rc = s->set_desc(*d, dims, 4, XNNL_TENSOR_NHWC, XNNL_DATA_HALF, XNNL_MEM_DDR, n);
    if (rc != 0) {
        s->destroy_desc(*d);
        *d = nullptr;
        return -3;
    }
    return 0;
}

int o3xnnl_open(void ** out, void * mida_device) {
    if (!out || !mida_device) {
        return -1;
    }
    struct o3xnnl * s = new o3xnnl();
    std::memset(s, 0, sizeof *s);

    s->lib = dlopen("/vendor/lib64/libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!s->lib) {
        s->lib = dlopen("libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!s->lib) {
        fprintf(stderr, "xnnl_shim: dlopen failed: %s\n", dlerror());
        delete s;
        return -2;
    }

    if (sym(s->lib, "xnnlCreate", (void **) &s->create) ||
        sym(s->lib, "xnnlDestroy", (void **) &s->destroy) ||
        sym(s->lib, "xnnlCreateTensorDescriptor", (void **) &s->create_desc) ||
        sym(s->lib, "xnnlDestroyTensorDescriptor", (void **) &s->destroy_desc) ||
        sym(s->lib, "xnnlSetTensorDescriptor", (void **) &s->set_desc) ||
        sym(s->lib, "xnnlCreateEpilogueDescriptor", (void **) &s->create_epl) ||
        sym(s->lib, "xnnlSetEpilogueDescriptor", (void **) &s->set_epl) ||
        sym(s->lib, "xnnlMatmulCreateGraph", (void **) &s->matmul)) {
        dlclose(s->lib);
        delete s;
        return -3;
    }

    xnnlHandle_t h = nullptr;
    int rc = s->create(&h, mida_device);
    if (rc != 0 || !h) {
        fprintf(stderr, "xnnl_shim: xnnlCreate -> %d\n", rc);
        dlclose(s->lib);
        delete s;
        return -4;
    }
    s->handle = h;
    *out = s;
    return 0;
}

void o3xnnl_close(void * handle) {
    struct o3xnnl * s = (struct o3xnnl *) handle;
    if (!s) {
        return;
    }
    if (s->handle && s->destroy) {
        s->destroy(s->handle);
    }
    if (s->lib) {
        dlclose(s->lib);
    }
    delete s;
}

/*
 * Build one matmul subgraph.
 *
 * in_dims  = {1,1,1,K}
 * w_dims   = {1,1,K,M}      (xnnl layout: M innermost)
 * out_dims = {T,1,1,M}
 *
 * transposed_w asks xnnl to accept the ggml weight layout instead (K
 * innermost); pass 0 when the caller already transposed the weight.
 */
int o3xnnl_matmul_graph(void * handle, void * subgraph,
                        const int in_dims[4], const void * in_ptr,
                        const int w_dims[4], const void * w_ptr,
                        const int out_dims[4], void * out_ptr,
                        int transposed_w) {
    struct o3xnnl * s = (struct o3xnnl *) handle;
    if (!s || !subgraph) {
        return -1;
    }

    xnnlTensorDescriptor_t di = nullptr, dw = nullptr, dout = nullptr;
    xnnlEpilogueDescriptor_t epl = nullptr;
    int rc;

    if ((rc = make_desc(s, &di, in_dims, "in")) != 0) return rc;
    if ((rc = make_desc(s, &dw, w_dims, "w")) != 0) return rc;
    if ((rc = make_desc(s, &dout, out_dims, "out")) != 0) return rc;

    if (s->create_epl(&epl) != 0 || !epl) {
        return -5;
    }
    /* all epilogue features disabled */
    if ((rc = s->set_epl(epl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) != 0) {
        return -6;
    }

    rc = s->matmul(s->handle, (void *) subgraph,
                   di, in_ptr, dw, w_ptr,
                   nullptr, nullptr, nullptr, nullptr,
                   dout, out_ptr, epl, transposed_w != 0, 0);

    s->destroy_desc(di);
    s->destroy_desc(dw);
    s->destroy_desc(dout);
    if (s->destroy_epl) {
        s->destroy_epl(epl);
    }
    return rc;
}

} /* extern "C" */
