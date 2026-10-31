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
    int (*abs)(xnnlHandle_t, void *, int, xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, void *);
    int (*neg)(xnnlHandle_t, void *, int, xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, void *);
    int (*sign)(xnnlHandle_t, void *, int, xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, void *);
    int (*rmsnorm)(xnnlHandle_t, void *, int, xnnlTensorDescriptor_t, const void *,
                   xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, void *, int, float, int);
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

    sym(s->lib, "xnnlRMSNormalizationCreateGraph", (void **) &s->rmsnorm);
    sym(s->lib, "xnnlAbsCreateGraph", (void **) &s->abs);
    sym(s->lib, "xnnlNegCreateGraph", (void **) &s->neg);
    sym(s->lib, "xnnlSignCreateGraph", (void **) &s->sign);
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

    {
        const char * hb = getenv("O3NPU_HANDLE_L2B");
        if (hb && *hb) {
            *(uint32_t *) ((char *) h + 0x10) = (uint32_t) strtoul(hb, nullptr, 0);
            fprintf(stderr, "xnnl_shim: handle+0x10 <- %s\n", hb);
        }
    }

    {
        const char * lvl = getenv("O3NPU_XNNL_LOG");
        void (*set_level)(int) = (void (*)(int)) dlsym(s->lib, "_ZN10XNNLLogger13set_log_levelE12XNNLLogLevel");
        if (lvl && *lvl && set_level) {
            set_level(atoi(lvl));
            fprintf(stderr, "xnnl_shim: log level -> %d\n", atoi(lvl));
        }
    }

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

    if (!getenv("O3NPU_EPL_NULL")) {
        if (s->create_epl(&epl) != 0 || !epl) {
            return -5;
        }
        /* all epilogue features disabled */
        if ((rc = s->set_epl(epl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) != 0) {
            return -6;
        }
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

extern "C" int o3xnnl_rmsnorm(void * handle, void * subgraph, int K,
                   const void * x, const void * w, void * y) {
    struct o3xnnl * s = (struct o3xnnl *) handle;
    if (!s || !s->rmsnorm) {
        return -1;
    }
    int dims[4] = { 1, 1, 1, K };
    xnnlTensorDescriptor_t xd = nullptr, wd = nullptr, yd = nullptr;
    if (make_desc(s, &xd, dims, "rin") != 0) return -2;
    if (make_desc(s, &wd, dims, "rw")  != 0) return -2;
    if (make_desc(s, &yd, dims, "rout")!= 0) return -2;
    int rc = s->rmsnorm(s->handle, subgraph, 0, xd, x, wd, w, yd, y, 0, 1e-6f, 0);
    s->destroy_desc(xd); s->destroy_desc(wd); s->destroy_desc(yd);
    return rc;
}

extern "C" int o3xnnl_unary(void * handle, void * subgraph, const char * name,
                            int n, const void * x, void * y) {
    struct o3xnnl * s = (struct o3xnnl *) handle;
    if (!s || !subgraph || !name) {
        return -1;
    }
    int (*fn)(xnnlHandle_t, void *, int, xnnlTensorDescriptor_t, const void *,
              xnnlTensorDescriptor_t, void *) = nullptr;
    if (strcmp(name, "abs") == 0)  fn = s->abs;
    if (strcmp(name, "neg") == 0)  fn = s->neg;
    if (strcmp(name, "sign") == 0) fn = s->sign;
    if (!fn) {
        return -2;
    }
    int dims[4] = { 1, 1, 1, n };
    xnnlTensorDescriptor_t xd = nullptr, yd = nullptr;
    if (make_desc(s, &xd, dims, "uin") != 0) return -3;
    if (make_desc(s, &yd, dims, "uout") != 0) return -3;
    int rc = fn(s->handle, subgraph, 0, xd, x, yd, y);
    s->destroy_desc(xd);
    s->destroy_desc(yd);
    return rc;
}
