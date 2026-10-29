/*
 * ggml backend for the Xiaomi 18 Fold (lhasa) NPU.
 *
 * Architecture
 * ------------
 * The device reports the CPU backend's host buffer type, so every tensor stays in
 * ordinary host memory and the scheduler only ever assigns us ops that
 * supports_op() claims. Everything else (views, copies, anything not listed)
 * stays on the CPU backend automatically. That keeps this file small and makes
 * correctness a matter of getting the matmul right.
 *
 * Current state: MUL_MAT for float types is computed by a reference loop on host
 * memory. The NPU path (xnnlMatmulCreateGraph -> midaSubgraphInstantiate ->
 * midaStreamLaunchSubgraph) is the next step and will replace o3npu_matmul.
 *
 * Build: see docs/GGML-BACKEND.md. Load with GGML_BACKEND_PATH=<path to this .so>.
 */
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define O3NPU_NAME "O3NPU"

static bool o3npu_debug(void) {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("O3NPU_DEBUG");
        v = (e && *e != '0') ? 1 : 0;
    }
    return v == 1;
}

/* ------------------------------------------------------------------ */
/* host buffer type (our own malloc-backed memory)                     */
/* ------------------------------------------------------------------ */

static const char * o3npu_buft_get_name(ggml_backend_buffer_type_t) {
    return O3NPU_NAME " host";
}

static void * o3npu_malloc(size_t size) {
    void * p = nullptr;
    if (posix_memalign(&p, 64, size ? size : 1) != 0) {
        return nullptr;
    }
    return p;
}

static void o3npu_buffer_free(ggml_backend_buffer_t buffer) {
    free(buffer->context);
    buffer->context = nullptr;
}

static void * o3npu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void o3npu_buffer_memset(ggml_backend_buffer_t buffer, ggml_tensor * t, uint8_t v, size_t off, size_t size) {
    (void) buffer;
    /* t->data already points at this tensor inside the buffer */
    memset((char *) t->data + off, v, size);
}

static void o3npu_buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * t, const void * data, size_t off, size_t size) {
    (void) buffer;
    memcpy((char *) t->data + off, data, size);
}

static void o3npu_buffer_get(ggml_backend_buffer_t buffer, const ggml_tensor * t, void * data, size_t off, size_t size) {
    (void) buffer;
    memcpy(data, (const char *) t->data + off, size);
}

static void o3npu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t v) {
    memset(buffer->context, v, buffer->size);
}

static ggml_backend_buffer_t o3npu_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = o3npu_malloc(size);
    if (!data) {
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, (ggml_backend_buffer_i) {
        /* .free_buffer   = */ o3npu_buffer_free,
        /* .get_base      = */ o3npu_buffer_get_base,
        /* .init_tensor   = */ nullptr,
        /* .memset_tensor = */ o3npu_buffer_memset,
        /* .set_tensor    = */ o3npu_buffer_set,
        /* .get_tensor    = */ o3npu_buffer_get,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ o3npu_buffer_clear,
        /* .reset         = */ nullptr,
    }, data, size);
}

static size_t o3npu_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

static bool o3npu_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static const ggml_backend_buffer_type_i o3npu_buft_iface = {
    /* .get_name         = */ o3npu_buft_get_name,
    /* .alloc_buffer     = */ o3npu_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ o3npu_buft_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ o3npu_buft_is_host,
};

static ggml_backend_buffer_type_t o3npu_buffer_type(void) {
    static ggml_backend_buffer_type_t buft = nullptr;
    if (!buft) {
        ggml_backend_buffer_type_t b = new ggml_backend_buffer_type();
        b->iface   = o3npu_buft_iface;
        b->device  = nullptr;
        b->context = nullptr;
        buft = b;
    }
    return buft;
}

/* ------------------------------------------------------------------ */
/* backend (stream)                                                    */
/* ------------------------------------------------------------------ */

static ggml_guid_t o3npu_guid(void) {
    static ggml_guid g = { 0x4f, 0x33, 0x4e, 0x50, 0x55, 0x00, 0x00, 0x01,
                           0x9a, 0x1c, 0x2b, 0x3d, 0x4e, 0x5f, 0x60, 0x71 };
    return &g;
}

static const char * o3npu_backend_get_name(ggml_backend_t) {
    return O3NPU_NAME;
}

static void o3npu_backend_free(ggml_backend_t backend) {
    delete backend;
}

static float o3npu_get_f32(const ggml_tensor * t, size_t off) {
    const char * p = (const char *) t->data + off;
    switch (t->type) {
        case GGML_TYPE_F32:  return *(const float *) p;
        case GGML_TYPE_F16:  return GGML_FP16_TO_FP32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return GGML_BF16_TO_FP32(*(const ggml_bf16_t *) p);
        default:             return 0.0f;
    }
}

static void o3npu_set_f32(ggml_tensor * t, size_t off, float v) {
    char * p = (char *) t->data + off;
    switch (t->type) {
        case GGML_TYPE_F32:  *(float *) p = v; break;
        case GGML_TYPE_F16:  *(ggml_fp16_t *) p = GGML_FP32_TO_FP16(v); break;
        case GGML_TYPE_BF16: *(ggml_bf16_t *) p = GGML_FP32_TO_BF16(v); break;
        default: break;
    }
}

static bool o3npu_type_ok(enum ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
}

/* can this exact node be computed by the reference matmul? */
static bool o3npu_can_matmul(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!a || !b) {
        return false;
    }
    if (!o3npu_type_ok(a->type) || !o3npu_type_ok(b->type)) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    /* no batched / broadcast / expert cases yet */
    if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1) {
        return false;
    }
    return true;
}

/*
 * dst[m, n] = sum_k src0[k, m] * src1[k, n]
 *
 * Uses nb[] strides instead of assuming contiguous layout: weights can be views
 * (for example a transposed tensor), and ne0-contiguous arithmetic silently
 * produces wrong numbers for those.
 */
static enum ggml_status o3npu_matmul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const size_t a0 = src0->nb[0], a1 = src0->nb[1];
    const size_t b0 = src1->nb[0], b1 = src1->nb[1];
    const size_t d0 = dst->nb[0],  d1 = dst->nb[1];

    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += o3npu_get_f32(src0, (size_t) k * a0 + (size_t) m * a1) *
                       o3npu_get_f32(src1, (size_t) k * b0 + (size_t) n * b1);
            }
            o3npu_set_f32(dst, (size_t) m * d0 + (size_t) n * d1, acc);
        }
    }
    return GGML_STATUS_SUCCESS;
}

/* element-wise copy with broadcasting and float type conversion */
static enum ggml_status o3npu_copy(ggml_tensor * dst, const ggml_tensor * src) {
    if (!src || !dst || !o3npu_type_ok(src->type) || !o3npu_type_ok(dst->type)) {
        return GGML_STATUS_FAILED;
    }
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
                    const size_t di = (size_t) i0 * dst->nb[0] + (size_t) i1 * dst->nb[1] +
                                      (size_t) i2 * dst->nb[2] + (size_t) i3 * dst->nb[3];
                    const int64_t s0 = src->ne[0] == 1 ? 0 : i0;
                    const int64_t s1 = src->ne[1] == 1 ? 0 : i1;
                    const int64_t s2 = src->ne[2] == 1 ? 0 : i2;
                    const int64_t s3 = src->ne[3] == 1 ? 0 : i3;
                    const size_t si = (size_t) s0 * src->nb[0] + (size_t) s1 * src->nb[1] +
                                      (size_t) s2 * src->nb[2] + (size_t) s3 * src->nb[3];
                    o3npu_set_f32(dst, di, o3npu_get_f32(src, si));
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status o3npu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    (void) backend;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (!o3npu_can_matmul(node)) {
                    fprintf(stderr, "O3NPU: MUL_MAT shape/type not supported\n");
                    return GGML_STATUS_FAILED;
                }
                if (o3npu_matmul(node->src[0], node->src[1], node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;

            /* views: the data pointer already points at the right place */
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
            case GGML_OP_NONE:
                break;

            case GGML_OP_CONT:
            case GGML_OP_CPY:
                if (o3npu_copy(node, node->src[0]) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "O3NPU: copy %d -> %d not supported\n",
                            (int) node->src[0]->type, (int) node->type);
                    return GGML_STATUS_FAILED;
                }
                break;

            default:
                fprintf(stderr, "O3NPU: unexpected op %d in graph_compute\n", (int) node->op);
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i o3npu_backend_iface = {
    /* .get_name           = */ o3npu_backend_get_name,
    /* .free               = */ o3npu_backend_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ o3npu_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

/* ------------------------------------------------------------------ */
/* device                                                              */
/* ------------------------------------------------------------------ */

static const char * o3npu_dev_get_name(ggml_backend_dev_t) {
    return O3NPU_NAME;
}

static const char * o3npu_dev_get_description(ggml_backend_dev_t) {
    return "VeriSilicon VIP9000 (Xiaomi 18 Fold) via o3npu";
}

static void o3npu_dev_get_memory(ggml_backend_dev_t, size_t * free_, size_t * total) {
    *free_ = 0;
    *total = 0;
}

static enum ggml_backend_dev_type o3npu_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void o3npu_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name         = o3npu_dev_get_name(dev);
    props->description  = o3npu_dev_get_description(dev);
    props->type         = o3npu_dev_get_type(dev);
    props->device_id    = nullptr;
    props->memory_free  = 0;
    props->memory_total = 0;
    props->caps = (ggml_backend_dev_caps) { 0 };
}

static ggml_backend_t o3npu_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void) params;
    ggml_backend_t backend = new ggml_backend();
    backend->guid    = o3npu_guid();
    backend->iface   = o3npu_backend_iface;
    backend->device  = dev;
    backend->context = nullptr;
    if (o3npu_debug()) {
        fprintf(stderr, "O3NPU: backend created\n");
    }
    return backend;
}

/*
 * Host memory: reuse the CPU backend's buffer type so tensors never have to be
 * copied between devices, and so ops we do not claim simply stay on the CPU.
 */
static ggml_backend_buffer_type_t o3npu_dev_get_buffer_type(ggml_backend_dev_t) {
    return o3npu_buffer_type();
}

static bool o3npu_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return o3npu_can_matmul(op);
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CONT:
        case GGML_OP_CPY:
            return false;
        default:
            return false;
    }
}

static bool o3npu_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft == o3npu_dev_get_buffer_type(nullptr);
}

static const ggml_backend_device_i o3npu_device_iface = {
    /* .get_name            = */ o3npu_dev_get_name,
    /* .get_description     = */ o3npu_dev_get_description,
    /* .get_memory          = */ o3npu_dev_get_memory,
    /* .get_type            = */ o3npu_dev_get_type,
    /* .get_props           = */ o3npu_dev_get_props,
    /* .init_backend        = */ o3npu_dev_init_backend,
    /* .get_buffer_type     = */ o3npu_dev_get_buffer_type,
    /* .get_host_buffer_type= */ nullptr,
    /* .buffer_from_host_ptr= */ nullptr,
    /* .supports_op         = */ o3npu_dev_supports_op,
    /* .supports_buft       = */ o3npu_dev_supports_buft,
    /* .offload_op          = */ nullptr,
    /* .event_new           = */ nullptr,
    /* .event_free          = */ nullptr,
    /* .event_synchronize   = */ nullptr,
};

/* ------------------------------------------------------------------ */
/* registry                                                            */
/* ------------------------------------------------------------------ */

static const char * o3npu_reg_get_name(ggml_backend_reg_t) {
    return O3NPU_NAME;
}

static size_t o3npu_reg_get_device_count(ggml_backend_reg_t) {
    return 1;
}

static ggml_backend_dev_t o3npu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return nullptr;
    }
    static ggml_backend_dev_t dev = nullptr;
    if (!dev) {
        ggml_backend_dev_t d = new ggml_backend_device();
        d->iface   = o3npu_device_iface;
        d->reg     = reg;
        d->context = nullptr;
        dev = d;
    }
    return dev;
}

static void * o3npu_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    (void) name;
    return nullptr;
}

static const ggml_backend_reg_i o3npu_reg_iface = {
    /* .get_name         = */ o3npu_reg_get_name,
    /* .get_device_count = */ o3npu_reg_get_device_count,
    /* .get_device       = */ o3npu_reg_get_device,
    /* .get_proc_address = */ o3npu_reg_get_proc_address,
};

/* ------------------------------------------------------------------ */
/* exports                                                             */
/* ------------------------------------------------------------------ */

static ggml_backend_reg_t o3npu_reg_get(void) {
    if (o3npu_debug()) {
        fprintf(stderr, "O3NPU: ggml_backend_init\n");
    }
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ o3npu_reg_iface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

static int o3npu_score(void) {
    return 100;
}

GGML_BACKEND_DL_IMPL(o3npu_reg_get)
GGML_BACKEND_DL_SCORE_IMPL(o3npu_score)

/* ------------------------------------------------------------------ */
/* debug export (offline unit test)                                    */
/* ------------------------------------------------------------------ */

extern "C" int o3npu_debug_matmul(const struct ggml_tensor * a, const struct ggml_tensor * b, struct ggml_tensor * d) {
    return (int) o3npu_matmul(a, b, d);
}

/* debug export for the offline test: element-wise copy with broadcasting */
extern "C" int o3npu_debug_copy(struct ggml_tensor * dst, const struct ggml_tensor * src) {
    return (int) o3npu_copy(dst, src);
}
