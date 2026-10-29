/* Generated from libxnlm.so DWARF by tools/xnnl/gen_headers.js.
 * Source of truth: the shipped library; do not edit by hand.
 */
#ifndef XNNL_API_H
#define XNNL_API_H

#include <stdint.h>


/* ---- types ---- */
typedef enum xnnlStatus_t {
    XNNL_STATUS_SUCCESS = 0,
    XNNL_STATUS_BAD_PARAM = 1,
    XNNL_STATUS_NOT_SUPPORTED = 2,
    XNNL_STATUS_INTERNAL_ERROR = 3,
} xnnlStatus_t;
/* size ? */
struct xnnlTensorDescriptor {
};
/* size 0xa0 */
struct xnnlQuantParam {
    bool is_asym;  /* +0x00 */
    xnnlQuantizeMode_t mode;  /* +0x04 */
    int groupwise_cnt;  /* +0x08 */
    int channel_cnt;  /* +0x0c */
    xnnlTensorData scale;  /* +0x10 */
    xnnlTensorData zp;  /* +0x40 */
    xnnlTensorData groupwise_scale;  /* +0x70 */
};
/* size 0x30 */
struct xnnlTensorData {
    xnnlDataType_t dtype;  /* +0x00 */
    void * data;  /* +0x08 */
    uint64_t dataSize;  /* +0x10 */
    std::__ndk1::vector<xnnlWeightPatch, std::__ndk1::allocator<xnnlWeightPatch> > splitInfo;  /* +0x18 */
};
/* size 0x01b8 */
struct xnnlDecoderWeight {
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > reordered_weights;  /* +0x00 */
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > prefill_sdp_params;  /* +0x18 */
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > decode_sdp_params;  /* +0x30 */
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > prefill_groupwise_scale;  /* +0x48 */
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > decode_groupwise_scale;  /* +0x60 */
    std::__ndk1::vector<std::__ndk1::pair<xnnlQuantParam, xnnlQuantParam>, std::__ndk1::allocator<std::__ndk1::pair<xnnlQuantParam, xnnlQuantParam> > > quant_param;  /* +0x78 */
    xnnlTensorData pre_embedding_table;  /* +0x90 */
    xnnlTensorData pre_rmsnorm_weight;  /* +0xc0 */
    xnnlTensorData post_rmsnorm_weight;  /* +0xf0 */
    xnnlTensorData attention_sink_bias;  /* +0x0120 */
    xnnlTensorData q_rmsnorm_weight;  /* +0x0150 */
    xnnlTensorData k_rmsnorm_weight;  /* +0x0180 */
    float lora_alpha;  /* +0x01b0 */
    bool enable_compress;  /* +0x01b4 */
    bool enable_embedding;  /* +0x01b5 */
};
/* size 0x90 */
struct xnnlEagletLmheadWeight {
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > reordered_weights;  /* +0x00 */
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > sdp_params;  /* +0x18 */
    std::__ndk1::vector<xnnlTensorData, std::__ndk1::allocator<xnnlTensorData> > groupwise_scale;  /* +0x30 */
    std::__ndk1::vector<std::__ndk1::pair<xnnlQuantParam, xnnlQuantParam>, std::__ndk1::allocator<std::__ndk1::pair<xnnlQuantParam, xnnlQuantParam> > > quant_param;  /* +0x48 */
    xnnlTensorData rmsnorm_weight;  /* +0x60 */
};
/* size 0x10 */
struct xnnlWeightPatch {
    bool isCompress;  /* +0x00 */
    size_t patchSize;  /* +0x08 */
};
/* size ? */
struct xnnlEpilogueDescriptor {
};
typedef enum xnnlTensorLayout_t {
    XNNL_TENSOR_LINEAR = 0,
    XNNL_TENSOR_NCHW = 1,
    XNNL_TENSOR_NHWC = 2,
    XNNL_TENSOR_NHSWC = 3,
    XNNL_TENSOR_SWIZZLE = 4,
} xnnlTensorLayout_t;
typedef enum xnnlDataType_t {
    XNNL_DATA_FLOAT = 0,
    XNNL_DATA_HALF = 1,
    XNNL_DATA_INT4 = 2,
    XNNL_DATA_UINT4 = 3,
    XNNL_DATA_INT8 = 4,
    XNNL_DATA_UINT8 = 5,
    XNNL_DATA_INT16 = 6,
    XNNL_DATA_UINT16 = 7,
    XNNL_DATA_INT32 = 8,
    XNNL_DATA_UINT32 = 9,
    XNNL_DATA_INT64 = 10,
} xnnlDataType_t;
typedef enum xnnlQuantizeMode_t {
    XNNL_QUANTIZE_MODE_NONE = 0,
    XNNL_QUANTIZE_MODE_PERTENSOR = 1,
    XNNL_QUANTIZE_MODE_PERCHANNEL = 2,
    XNNL_QUANTIZE_MODE_PERGROUP = 3,
    XNNL_QUANTIZE_MODE_PERHEAD = 4,
} xnnlQuantizeMode_t;
typedef enum xnnlMemLocation_t {
    XNNL_MEM_DDR = 0,
    XNNL_MEM_L2B = 1,
} xnnlMemLocation_t;
typedef struct xnnlHandle * xnnlHandle_t;
typedef struct xnnlTensorDescriptor * xnnlTensorDescriptor_t;
typedef struct xnnlEpilogueDescriptor * xnnlEpilogueDescriptor_t;
typedef enum xnnlEpilogueDataMode_t {
    XNNL_EPL_DATA_MODE_DISABLE = 0,
    XNNL_EPL_DATA_MODE_PER_TENSOR = 1,
    XNNL_EPL_DATA_MODE_PER_CHANNEL = 2,
} xnnlEpilogueDataMode_t;
typedef enum xnnlActivation1Type_t {
    XNNL_ACT1_TYPE_DISABLE = 0,
    XNNL_ACT1_TYPE_RELU = 1,
    XNNL_ACT1_TYPE_RELU6 = 2,
    XNNL_ACT1_TYPE_LEAKY_RELU = 3,
    XNNL_ACT1_TYPE_PRELU = 4,
    XNNL_ACT1_TYPE_SWISH = 5,
} xnnlActivation1Type_t;
typedef enum xnnlActivation2Type_t {
    XNNL_ACT2_TYPE_DISABLE = 0,
    XNNL_ACT2_TYPE_SIGMOID = 1,
    XNNL_ACT2_TYPE_TANH = 2,
    XNNL_ACT2_TYPE_RELU = 3,
} xnnlActivation2Type_t;
typedef enum xnnlEwOp_t {
    XNNL_EW_OP_DISABLE = 0,
    XNNL_EW_OP_CAST = 1,
    XNNL_EW_OP_MUL = 2,
    XNNL_EW_OP_MIN = 3,
    XNNL_EW_OP_MAX = 4,
    XNNL_EW_OP_ADD = 5,
    XNNL_EW_OP_SUB = 6,
} xnnlEwOp_t;

/* ---- declarations with full signatures (from DWARF) ---- */
xnnlStatus_t xnnlCreate(xnnlHandle_t *, midaDevice_t);
xnnlStatus_t xnnlCreateEpilogueDescriptor(xnnlEpilogueDescriptor_t *);
xnnlStatus_t xnnlCreateTensorDescriptor(xnnlTensorDescriptor_t *);
xnnlStatus_t xnnlDestroy(xnnlHandle_t);
xnnlStatus_t xnnlDestroyEpilogueDescriptor(xnnlEpilogueDescriptor_t);
xnnlStatus_t xnnlDestroyTensorDescriptor(xnnlTensorDescriptor_t);
xnnlStatus_t xnnlMatmulCreateGraph(const xnnlHandle_t, midaSubgraph_t, const xnnlTensorDescriptor_t, const void *, const xnnlTensorDescriptor_t, const void *, const xnnlTensorDescriptor_t, const void *, const void *, const void *, const xnnlTensorDescriptor_t, void *, const xnnlEpilogueDescriptor_t, const bool, const uint32_t);
xnnlStatus_t xnnlQuantizeCreateGraph(const xnnlHandle_t, midaSubgraph_t, const int32_t, const int32_t, const int32_t, const xnnlTensorDescriptor_t, const void *, const xnnlTensorDescriptor_t, const void *, const xnnlTensorDescriptor_t, const void *, const xnnlTensorDescriptor_t, void *, const uint32_t);
xnnlStatus_t xnnlRMSNormalizationCreateGraph(const xnnlHandle_t, midaSubgraph_t, const int, const xnnlTensorDescriptor_t, const void *, const xnnlTensorDescriptor_t, const void *, const xnnlTensorDescriptor_t, void *, const int, float, int);
xnnlStatus_t xnnlSetEpilogueDescriptor(xnnlEpilogueDescriptor_t, const xnnlEpilogueDataMode_t, const xnnlEpilogueDataMode_t, const xnnlActivation1Type_t, const xnnlEwOp_t, const xnnlEpilogueDataMode_t, const xnnlEpilogueDataMode_t, const xnnlEpilogueDataMode_t, const xnnlActivation2Type_t, const xnnlEpilogueDataMode_t, const xnnlEpilogueDataMode_t);
xnnlStatus_t xnnlSetTensorDescriptor(xnnlTensorDescriptor_t, const int *, const uint, const xnnlTensorLayout_t, const xnnlDataType_t, const xnnlMemLocation_t, const std::__ndk1::string);
xnnlStatus_t xnnlSetTensorDescriptorGroupwiseParam(xnnlTensorDescriptor_t, const uint32_t, const uint8_t *);
xnnlStatus_t xnnlSetTensorDescriptorQuantParam(xnnlTensorDescriptor_t, const bool, const bool, const void *, const void *, const xnnlDataType_t, const xnnlDataType_t);

/* ---- declarations whose parameters are not in DWARF;
 *      arity recovered from DW_TAG_GNU_call_site_parameter ---- */

#endif