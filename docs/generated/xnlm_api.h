/* Generated from libxnlm.so DWARF by tools/xnnl/gen_headers.js.
 * Source of truth: the shipped library; do not edit by hand.
 */
#ifndef XNLM_API_H
#define XNLM_API_H

#include <stdint.h>


/* ---- types ---- */
typedef enum XnlmStatus {
    XNLM_STATUS_SUCCESS = 0,
    XNLM_STATUS_BAD_PARAM = 1,
    XNLM_STATUS_NOT_SUPPORTED = 2,
    XNLM_STATUS_TIMEOUT = 3,
    XNLM_STATUS_INTERNAL_ERROR = 4,
    XNLM_STATUS_EXCEED_MAX_SEQ_LEN = 5,
    XNLM_STATUS_EXCEED_MAX_CTX_LEN = 6,
    XNLM_STATUS_DEVICE_EXCEPTION_ERROR = 7,
} XnlmStatus;
typedef enum XnlmDataType {
    XNLM_DTYPE_FLOAT32 = 0,
    XNLM_DTYPE_FLOAT16 = 1,
    XNLM_DTYPE_INT4 = 2,
    XNLM_DTYPE_UINT4 = 3,
    XNLM_DTYPE_INT8 = 4,
    XNLM_DTYPE_UINT8 = 5,
    XNLM_DTYPE_INT16 = 6,
    XNLM_DTYPE_UINT16 = 7,
    XNLM_DTYPE_INT32 = 8,
    XNLM_DTYPE_UINT32 = 9,
    XNLM_DTYPE_INT64 = 10,
} XnlmDataType;
/* size 0x28 */
struct XnlmQuantParam {
    XnlmDataType mDtype;  /* +0x00 */
    bool mIsPerchn;  /* +0x04 */
    bool mIsAsym;  /* +0x05 */
    XnlmDataType mScaleDtype;  /* +0x08 */
    XnlmDataType mZpDtype;  /* +0x0c */
    uint32_t mGroupwiseCnt;  /* +0x10 */
    uint32_t mScaleSize;  /* +0x14 */
    uint32_t mZpSize;  /* +0x18 */
    uint32_t mWeightSize;  /* +0x1c */
    uint32_t mChannelCnt;  /* +0x20 */
    xnnlQuantizeMode_t mQuantMode;  /* +0x24 */
};
typedef struct {
    int32_t shape[4];  /* +0x00 */
    uint64_t dim;  /* +0x10 */
    XnlmDataType dtype;  /* +0x18 */
    XnlmMemory mem;  /* +0x20 */
    char name[64];  /* +0x28 */
} XnlmTensorDesc;
typedef struct {
    int32_t shape[4];  /* +0x00 */
    uint64_t dim;  /* +0x10 */
    XnlmDataType dtype;  /* +0x18 */
    XnlmMemoryImpl mem;  /* +0x20 */
    char name[64];  /* +0x30 */
} XnlmTensorDescImpl;
typedef struct {
    void * mAddr;  /* +0x00 */
    void * mDmaMem;  /* +0x08 */
} XnlmMemoryImpl;

/* ---- exported API ---- */
XnlmStatus XNLM_KVCache_getCacheSize(XnlmKVCache kvCache, int32_t nValid, uint64_t * bufSize);
XnlmStatus XNLM_KVCache_getCurrentLength(XnlmKVCache kvCache, int32_t * nPast);
XnlmStatus XNLM_KVCache_loadCacheData(XnlmKVCache kvCache, int32_t nValid, void * ptr, uint64_t bufSize);
XnlmStatus XNLM_KVCache_saveCacheData(XnlmKVCache kvCache, int32_t nValid, void * ptr, uint64_t bufSize);
XnlmStatus XNLM_KVCache_update(XnlmKVCache kvCache, int32_t nPast);
XnlmStatus XNLM_KVCache_updateWithSelection(XnlmKVCache kvCache, int32_t nPast, bool * selecteFlag, uint64_t seletedNum);
XnlmStatus XNLM_Memory_flushCache(XnlmMemory mem);
XnlmStatus XNLM_Memory_flushCachePartial(XnlmMemory mem, uint64_t offset, uint64_t size);
XnlmStatus XNLM_Memory_getAddr(XnlmMemory mem, void ** addr);
XnlmStatus XNLM_Memory_invalidateCache(XnlmMemory mem);
XnlmStatus XNLM_Memory_invalidateCachePartial(XnlmMemory mem, uint64_t offset, uint64_t size);
XnlmStatus XNLM_Model_getKVCache(XnlmModel model, XnlmKVCache * kvCache);
XnlmStatus XNLM_Model_loadLora(XnlmModel model, XnlmLoraWeight loraWeight);
XnlmStatus XNLM_Model_run(XnlmModel model, XnlmTensorDesc ** inputs, uint64_t inputsNum, XnlmTensorDesc ** outputs, uint64_t outputsNum);
XnlmStatus XNLM_Model_setPerformance(XnlmModel model, XnlmPerformanceConfig perfConfig);
XnlmStatus XNLM_Model_unloadLora(XnlmModel model);
XnlmStatus XNLM_RunConfig_create(XnlmRunConfig ** runConfig);
XnlmStatus XNLM_RunConfig_destroy(XnlmRunConfig * runConfig);
XnlmStatus XNLM_Runtime_allocMemory(XnlmRuntime runtime, uint64_t size, XnlmMemory * mem);
XnlmStatus XNLM_Runtime_create(XnlmRuntime * runtime);
XnlmStatus XNLM_Runtime_createLoraWeight(XnlmRuntime runtime, const char * path, XnlmLoraWeight * loraWeight);
XnlmStatus XNLM_Runtime_createModel(XnlmRuntime runtime, XnlmWeight weight, XnlmRunConfig * runConfig, XnlmModel * model);
XnlmStatus XNLM_Runtime_createWeight(XnlmRuntime runtime, const char * path, bool asyncLoad, bool directIO, XnlmWeight refWeight, XnlmWeight * newWeight);
XnlmStatus XNLM_Runtime_destroy(XnlmRuntime runtime);
XnlmStatus XNLM_Runtime_destroyLoraWeight(XnlmRuntime runtime, XnlmLoraWeight loraWeight);
XnlmStatus XNLM_Runtime_destroyModel(XnlmRuntime runtime, XnlmModel model);
XnlmStatus XNLM_Runtime_destroyWeight(XnlmRuntime runtime, XnlmWeight weight);
XnlmStatus XNLM_Runtime_freeMemory(XnlmRuntime runtime, XnlmMemory mem);
XnlmStatus XNLM_getVersion(const char ** version);

#endif