/*
 * xnnl_shim.cpp - the C++-only part of libxnnl.
 *
 * xnnlSetTensorDescriptor takes a std::string by value, so it cannot be called
 * from C.  Everything else here is a thin dlsym wrapper; the goal is only to
 * keep tools/npu/o3npu.c free of C++.
 */
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    int (*set_quant)(xnnlTensorDescriptor_t, bool, bool, const void *, const void *, int, int);
    int (*set_box)(void *, const unsigned long *, const unsigned long *);
    int (*add_gemm)(void *, void *, int, void *, void *, void *, void *, void *, void *,
                    int, int, int, int, int, int, void *, void *, int);
    int (*get_gemm_name)(std::string &, void *, int, unsigned int, unsigned int);
    int (*get_reorder_size)(void *, void *, void *, void *, uint32_t *, uint32_t *,
                            uint32_t, uint8_t, uint32_t);
    int (*matmul)(xnnlHandle_t, void *, xnnlTensorDescriptor_t, const void *,
                  xnnlTensorDescriptor_t, const void *, xnnlTensorDescriptor_t, const void *,
                  const void *, const void *, xnnlTensorDescriptor_t, void *,
                  xnnlEpilogueDescriptor_t, bool, uint32_t);
};

/* Patch xnnlMatmulCreateGraph policyMode_ compare so 0 takes the
 * createGraphSpecializedN branch instead of createGraphSplitMN.
 * offset 0x8a4 from function entry; cmp w8,#1 -> cmp w8,#0 */
static void o3xnnl_force_specn(void * matmul_fn) {
    if (!getenv("O3NPU_FORCE_SPECN")) return;
    unsigned char * p = (unsigned char *) matmul_fn + 0x8a4;
    uint32_t insn = 0;
    memcpy(&insn, p, 4);
    fprintf(stderr, "xnnl_shim: policy insn @ %p = 0x%08x\n", p, insn);
    if (insn != 0x7100051Fu) {
        fprintf(stderr, "xnnl_shim: unexpected insn, not patching\n");
        return;
    }
    long page = sysconf(_SC_PAGESIZE);
    uintptr_t pg = (uintptr_t) p & ~((uintptr_t) page - 1);
    if (mprotect((void *) pg, (size_t) page, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        perror("xnnl_shim: mprotect");
        return;
    }
    uint32_t patched = 0x7100011Fu;   /* cmp w8, #0 */
    memcpy(p, &patched, 4);
    __builtin___clear_cache((char *) p, (char *) p + 4);
    fprintf(stderr, "xnnl_shim: patched policyMode_ compare -> specializedN\n");
}

/* Rename the L1B matmul kernel string to its non-L1B sibling. */
static void o3xnnl_force_nol1b(void * matmul_fn) {
    if (!getenv("O3NPU_FORCE_NOL1B")) return;
    uintptr_t bias = (uintptr_t) matmul_fn - 0x5d386c;
    unsigned char * str = (unsigned char *) (bias + 0x330a6);
    fprintf(stderr, "xnnl_shim: kernel str @ %p = %.40s\n", str, str);
    long page = sysconf(_SC_PAGESIZE);
    uintptr_t pg = (uintptr_t) str & ~((uintptr_t) page - 1);
    if (mprotect((void *) pg, (size_t) page, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        perror("xnnl_shim: mprotect str");
        return;
    }
    str[25] = 0;   /* "MatmulOpPerChnSplitKernelL1B" -> "...SplitKernel" */
    __builtin___clear_cache((char *) str, (char *) str + 32);
    fprintf(stderr, "xnnl_shim: kernel str now = %.40s\n", str);
}

static int env_i(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

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
    int layout = XNNL_TENSOR_NHWC;
    int memloc = XNNL_MEM_DDR;
    const char * me = getenv("O3NPU_DESC_MEM");
    if (me && *me) memloc = atoi(me);
    const char * le = getenv("O3NPU_DESC_LAYOUT");
    if (le && *le) layout = atoi(le);
    int rc = s->set_desc(*d, dims, 4, layout, XNNL_DATA_HALF, memloc, n);
    if (rc != 0) {
        s->destroy_desc(*d);
        *d = nullptr;
        return -3;
    }
    if (getenv("O3NPU_DUMP_DESC")) {
        unsigned int * q = (unsigned int *) *d;
        fprintf(stderr, "xnnl_shim: desc %s %p dims:", name, *d);
        int * dp = (int *) *(void **) *d;
        for (int i = 0; i < 6; i++) fprintf(stderr, " %d", dp[i]);
        fprintf(stderr, " |");
        for (int i = 0; i < 0x28; i++) fprintf(stderr, " %08x", q[i]);
        fprintf(stderr, "\n");
    }
    return 0;
}

int o3xnnl_open(void ** out, void * mida_device) {
    if (!out || !mida_device) {
        return -1;
    }
    struct o3xnnl * s = new o3xnnl();
    std::memset(s, 0, sizeof *s);

    const char * xl_env = getenv("O3NPU_XNNL_LIB");
    if (xl_env) fprintf(stderr, "xnnl_shim: loading %s\n", xl_env);
    s->lib = xl_env ? dlopen(xl_env, RTLD_NOW | RTLD_GLOBAL) : nullptr;
    if (!s->lib && xl_env) fprintf(stderr, "xnnl_shim: load failed: %s\n", dlerror());
    if (!s->lib) s->lib = dlopen("/vendor/lib64/libxnnl.so", RTLD_NOW | RTLD_GLOBAL);
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
        sym(s->lib, "xnnlSetTensorDescriptorQuantParam", (void **) &s->set_quant) ||
        sym(s->lib, "xnnlSetTensorDescriptorBoxInfo", (void **) &s->set_box) ||
        sym(s->lib, "xnnlAddGemmSplitMNKernelNode", (void **) &s->add_gemm) ||
        sym(s->lib, "xnnlGetGemmSplitMNKernelName", (void **) &s->get_gemm_name) ||
        sym(s->lib, "xnnlGetGemmSplitMNKernelReorderSize", (void **) &s->get_reorder_size) ||
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
        const char * dv = getenv("O3NPU_DEV_VER");
        int (*setver)(xnnlHandle_t, uint32_t) =
            (int (*)(xnnlHandle_t, uint32_t)) dlsym(s->lib, "xnnlSetDeviceVersion");
        if (dv && *dv && setver) {
            int r = setver(h, (uint32_t) strtoul(dv, nullptr, 0));
            fprintf(stderr, "xnnl_shim: set_device_version(%s) -> %d\n", dv, r);
        }
    }

    /* Note: patching handle+0x10 (old O3NPU_HANDLE_L2B) makes the next
     * xnnlAddGemmSplitMNKernelNode segfault; that offset is not l2bMaxSize. */

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
                        int transposed_w, void * wscale, int nscale) {
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

    o3xnnl_force_specn((void *) s->matmul);
    o3xnnl_force_nol1b((void *) s->matmul);

    if (wscale && s->set_quant) {
        int qr = s->set_quant(dw, true, false, wscale, nullptr, 0, 0);
        fprintf(stderr, "xnnl_shim: set_quant(w, perChn=1, n=%d) -> %d\n", nscale, qr);
    }

    if (!getenv("O3NPU_EPL_NULL")) {
        if (s->create_epl(&epl) != 0 || !epl) {
            return -5;
        }
        /* all epilogue features disabled */
        int epl_act = getenv("O3NPU_EPL_ACT") ? atoi(getenv("O3NPU_EPL_ACT")) : 0;
        int epl_ew = getenv("O3NPU_EPL_EW") ? atoi(getenv("O3NPU_EPL_EW")) : 0;
        if ((rc = s->set_epl(epl, 0, 0, epl_act, epl_ew, 0, 0, 0, 0, 0, 0)) != 0) {
            return -6;
        }
    }

    rc = s->matmul(s->handle, (void *) subgraph,
                   di, in_ptr, dw, w_ptr,
                   nullptr, nullptr, nullptr, nullptr,
                   dout, out_ptr, epl, transposed_w != 0,
                   (uint32_t) strtoul(getenv("O3NPU_MM_LAST") ? getenv("O3NPU_MM_LAST") : "0", nullptr, 0));

    s->destroy_desc(di);
    s->destroy_desc(dw);
    s->destroy_desc(dout);
    if (s->destroy_epl) {
        s->destroy_epl(epl);
    }
    return rc;
}


struct o3_tensor_box {
    int tensor_shape[6];   /* +0x00 */
    int box_start[6];      /* +0x18 */
    int box_dim[6];        /* +0x30 */
    int rank;              /* +0x48 */
    int reserved0;         /* +0x4c */
    unsigned long long ddr_va;   /* +0x50 */
    int layout;            /* +0x58 */
    int dtype;             /* +0x5c */
    int location;          /* +0x60 */
    unsigned char flags[4];/* +0x64 */
    int reserved2;         /* +0x68 */
    int reserved3;         /* +0x6c */
    void *scale;           /* +0x70 */
    int dtype_scale;       /* +0x78 */
    int reserved4;         /* +0x7c */
    void *zp;              /* +0x80 */
    int dtype_zp;          /* +0x88 */
    int reserved5;         /* +0x8c */
    int groupwise_cnt;     /* +0x90 */
    unsigned char tail[0x100];
};

static void fill_box(o3_tensor_box *b, const int dims[4], void *va) {
    memset(b, 0, sizeof(*b));
    unsigned long long n = 1;
    for (int i = 0; i < 4; i++) {
        b->tensor_shape[i] = dims[i];
        b->box_dim[i] = dims[i];
        b->box_start[i] = 0;
        n *= (unsigned long long) dims[i];
    }
    b->rank = 4;
    b->layout = 2;         /* XNNL_TENSOR_NHWC */
    b->dtype = 1;          /* XNNL_DATA_HALF */
    b->location = 0;       /* XNNL_MEM_DDR */
    b->ddr_va = (unsigned long long) (unsigned long) va;
    (void) n;
}

extern "C" int o3xnnl_gemm_splitmn(void * handle, void * subgraph,
                        const int in_dims[4], const void * in_ptr,
                        const int w_dims[4], const void * w_ptr,
                        const int out_dims[4], void * out_ptr,
                        int transposed_w, void * wscale, int nscale) {
    struct o3xnnl * s = (struct o3xnnl *) handle;
    if (!s || !subgraph || !s->add_gemm) return -1;
    o3_tensor_box bi, bw, bo;
    fill_box(&bi, in_dims, (void *) in_ptr);
    fill_box(&bw, w_dims, (void *) w_ptr);
    fill_box(&bo, out_dims, out_ptr);
    (void) wscale; (void) nscale;
    int p3  = env_i("O3NPU_GEMM_PARAM3", 2);
    int p10 = env_i("O3NPU_GEMM_PARAM10", 5);
    int p18 = env_i("O3NPU_GEMM_PARAM18", 2);
    /* xnnlAddGemmSplitMNKernelNode(graph, handle, ...), not (handle, graph)! */
    int rc = s->add_gemm(subgraph, s->handle, p3, &bi, &bw, 0, 0, 0, &bo,
                         p10, 1, 0, transposed_w ? 1 : 0, 0, 0, nullptr, nullptr, p18);
    fprintf(stderr, "xnnl_shim: addGemmSplitMN(box, p3=%d p10=%d p18=%d) -> %d\n",
            p3, p10, p18, rc);
    return rc;
}

extern "C" int o3xnnl_gemm_name(void * handle, const int out_dims[4], void * out_ptr,
                                int act, int all_head_num, int flags) {
    struct o3xnnl * s = (struct o3xnnl *) handle;
    if (!s || !s->get_gemm_name) return -1;
    o3_tensor_box bo;
    fill_box(&bo, out_dims, out_ptr);
    std::string name;
    int rc = s->get_gemm_name(name, &bo, act, (unsigned) all_head_num, (unsigned) flags);
    fprintf(stderr, "xnnl_shim: gemm_name(act=%d allhead=%d flags=%d) -> %d name='%s'\n",
            act, all_head_num, flags, rc, name.c_str());
    return rc;
}

extern "C" int o3xnnl_gemm_reorder_size(void * handle,
                                        const int in_dims[4], const void * in_ptr,
                                        const int w_dims[4], const void * w_ptr,
                                        const int out_dims[4], void * out_ptr,
                                        int act, int transposed, int config) {
    struct o3xnnl * s = (struct o3xnnl *) handle;
    if (!s || !s->get_reorder_size) return -1;
    o3_tensor_box bi, bw, bo;
    fill_box(&bi, in_dims, (void *) in_ptr);
    fill_box(&bw, w_dims, (void *) w_ptr);
    fill_box(&bo, out_dims, out_ptr);
    uint32_t sdp = 0, gw = 0;
    int rc = s->get_reorder_size(&bi, &bw, nullptr, &bo, &sdp, &gw,
                                 (uint32_t) act, (uint8_t) (transposed ? 1 : 0),
                                 (uint32_t) config);
    fprintf(stderr, "xnnl_shim: gemm_reorder_size(act=%d trans=%d cfg=%d) -> %d "
                    "sdpSize=%u gwScaleSize=%u\n",
            act, transposed, config, rc, sdp, gw);
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
    if (getenv("O3NPU_UNARY_DIMS")) {
        dims[0] = 1; dims[1] = n; dims[2] = 1; dims[3] = 1;
    }
    xnnlTensorDescriptor_t xd = nullptr, yd = nullptr;
    if (make_desc(s, &xd, dims, "uin") != 0) return -3;
    if (make_desc(s, &yd, dims, "uout") != 0) return -3;
    int l2b = 1024 * 1024;
    const char * e = getenv("O3NPU_UNARY_L2BPARAM");
    if (!e) e = getenv("O3NPU_L2B");
    if (e && *e) l2b = atoi(e);
    int rc;
    if (getenv("O3NPU_UNARY_INPLACE")) {
        rc = fn(s->handle, subgraph, l2b, xd, x, xd, (void *) x);
    } else if (getenv("O3NPU_UNARY_SWAP")) {
        rc = fn(s->handle, subgraph, l2b, yd, y, xd, (void *) x);
    } else {
        rc = fn(s->handle, subgraph, l2b, xd, x, yd, y);
    }
    s->destroy_desc(xd);
    s->destroy_desc(yd);
    return rc;
}
