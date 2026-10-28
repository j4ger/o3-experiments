/*
 * Minimal ggml backend for the Xiaomi 18 Fold (lhasa) NPU.
 *
 * Milestone 1 (this file): plumbing. It registers as a ggml device that claims
 * GGML_OP_MUL_MAT for F32/F16 operands and computes it with a reference loop on
 * host memory. Everything else is left to the CPU backend by the scheduler, so
 * llama.cpp runs correctly with this backend loaded.
 *
 * Milestone 2 replaces the reference loop with the NPU path:
 *   upload weights to NPU memory (o3npu_alloc(..., cpu_visible = 1))
 *   -> xnnlMatmulCreateGraph -> midaSubgraphInstantiate
 *   -> midaStreamLaunchSubgraph -> read back
 *
 * Load with -DGGML_BACKEND_DL=ON at llama.cpp build time and put this .so in
 * GGML_BACKEND_PATH.
 */
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

#define GGML_BACKEND_DL 1

#define O3NPU_NAME "O3NPU"

/* ------------------------------------------------------------------ */
/* host buffer type                                                    */
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
    memset((char *) buffer->context + t->view_offs + off, v, size);
}

static void o3npu_buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * t, const void * data, size_t off, size_t size) {
    memcpy((char *) buffer->context + t->view_offs + off, data, size);
}

static void o3npu_buffer_get(ggml_backend_buffer_t buffer, const ggml_tensor * t, void * data, size_t off, size_t size) {
    memcpy(data, (const char *) buffer->context + t->view_offs + off, size);
}

static void o3npu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t v) {
    memset(buffer->context, v, buffer->size);
}

static ggml_backend_buffer_t o3npu_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = o3npu_malloc(size);
    if (!data) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, (ggml_backend_buffer_i) {
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
    return buffer;
}

static size_t o3npu_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

static bool o3npu_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static const ggml_backend_buffer_type_i o3npu_buft_iface = {
    /* .get_name       = */ o3npu_buft_get_name,
    /* .alloc_buffer   = */ o3npu_buft_alloc_buffer,
    /* .alloc_buffer_n = */ nullptr,
    /* .get_alignment  = */ o3npu_buft_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host        = */ o3npu_buft_is_host,
};

static ggml_backend_buffer_type_t o3npu_buffer_type(void) {
    static ggml_backend_buffer_type_t buft = nullptr;
    if (!buft) {
        ggml_backend_buffer_type_t b = new ggml_backend_buffer_type();
        b->iface = o3npu_buft_iface;
        b->device = nullptr;
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

static float o3npu_read_f32(const ggml_tensor * t, size_t i) {
    switch (t->type) {
        case GGML_TYPE_F32: return ((const float *) t->data)[i];
        case GGML_TYPE_F16: return GGML_FP16_TO_FP32(((const ggml_fp16_t *) t->data)[i]);
        default:            return 0.0f;
    }
}

/*
 * ggml_mul_mat: src0 [K, M], src1 [K, N] -> dst [M, N]
 *   dst[m, n] = sum_k src0[k, m] * src1[k, n]
 */
static enum ggml_status o3npu_matmul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    if (dst->type != GGML_TYPE_F32) {
        return GGML_STATUS_FAILED;
    }
    float * out = (float *) dst->data;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += o3npu_read_f32(src0, (size_t) k + (size_t) m * K) *
                       o3npu_read_f32(src1, (size_t) k + (size_t) n * K);
            }
            out[m + n * M] = acc;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status o3npu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    (void) backend;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                ggml_tensor * src0 = node->src[0];
                ggml_tensor * src1 = node->src[1];
                /* continuation of a split matmul: accumulate into the same dst */
                enum ggml_status st = o3npu_matmul(src0, src1, node);
                if (st != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "O3NPU: matmul failed (types %d x %d -> %d)\n",
                            src0->type, src1->type, node->type);
                    return st;
                }
                break;
            }
            /* view ops: the data pointer is already correct, nothing to compute */
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
            case GGML_OP_NONE:
                break;

            /* materialise a contiguous copy of the source */
            case GGML_OP_CONT:
            case GGML_OP_CPY: {
                ggml_tensor * src = node->src[0];
                if (!src || !node->data || !src->data) {
                    return GGML_STATUS_FAILED;
                }
                const size_t nb = ggml_nbytes(node);
                if (nb > ggml_nbytes(src)) {
                    return GGML_STATUS_FAILED;
                }
                memcpy(node->data, src->data, nb);
                break;
            }

            default:
                fprintf(stderr, "O3NPU: unsupported op %d\n", (int) node->op);
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
    /* GPU-ish so llama.cpp treats it as an accelerator for offload */
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void o3npu_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = o3npu_dev_get_name(dev);
    props->description = o3npu_dev_get_description(dev);
    props->type = o3npu_dev_get_type(dev);
    props->device_id = nullptr;
    props->memory_free = 0;
    props->memory_total = 0;
    props->caps = (ggml_backend_dev_caps) { 0 };
}

static ggml_backend_t o3npu_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void) params;
    ggml_backend_t backend = new ggml_backend();
    backend->guid = o3npu_guid();
    backend->iface = o3npu_backend_iface;
    backend->device = dev;
    backend->context = nullptr;
    fprintf(stderr, "O3NPU: backend created\n");
    return backend;
}

static ggml_backend_buffer_type_t o3npu_dev_get_buffer_type(ggml_backend_dev_t) {
    return o3npu_buffer_type();
}

static bool o3npu_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
            return true;
        case GGML_OP_MUL_MAT:
            break;
        default:
            return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!a || !b) {
        return false;
    }
    /* milestone 1: plain float matmul only; quantized types stay on the CPU */
    const bool at = a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16;
    const bool bt = b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16;
    const bool dt = op->type == GGML_TYPE_F32;
    return at && bt && dt;
}

static bool o3npu_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == o3npu_buffer_type();
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
        d->iface = o3npu_device_iface;
        d->reg = reg;
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
    fprintf(stderr, "O3NPU: ggml_backend_init called\n");
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ o3npu_reg_iface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

static int o3npu_score(void) {
    fprintf(stderr, "O3NPU: score asked\n");
    return 100;
}

GGML_BACKEND_DL_IMPL(o3npu_reg_get)
GGML_BACKEND_DL_SCORE_IMPL(o3npu_score)
