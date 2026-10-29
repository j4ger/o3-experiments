/* Generated from libxnlm.so DWARF by tools/xnnl/gen_headers.js.
 * Source of truth: the shipped library; do not edit by hand.
 */
#ifndef MIDA_API_H
#define MIDA_API_H

#include <stdint.h>


/* opaque handles as declared in the vendor headers */
typedef struct opaque * midaAddress_t;
/* size 0x1c */
struct midaDeviceConfig {
    PerformanceMode performance;  /* +0x00 */
    uint32_t log_close;  /* +0x04 */
    uint32_t pu_num;  /* +0x08 */
    uint32_t mem_debug;  /* +0x0c */
    uint32_t cu_delay;  /* +0x10 */
    bool drm_protect;  /* +0x14 */
    uint32_t power_policy;  /* +0x18 */
};
typedef struct opaque * midaDevice_t;
typedef struct opaque * midaDirectIO_t;
typedef struct opaque * midaEvent_t;
/* size ? */
struct midaGraphExecUpdateResultInfo {
};
typedef struct opaque * midaGraphExec_t;
typedef struct opaque * midaGraphNode_t;
typedef struct opaque * midaGraph_t;
/* size 0xb8 */
struct midaLaunchConfig {
    uint32_t cu_num;  /* +0x00 */
    uint32_t performance_level;  /* +0x04 */
    float ddr_ratio;  /* +0x08 */
    uint32_t fw_timeout;  /* +0x0c */
    const char * name;  /* +0x10 */
    char[5] stripe_names[32];  /* +0x18 */
};
/* size ? */
struct midaMemoryUsage {
};
typedef struct opaque * midaMemory_t;
typedef struct void__midaError_t__void___ * midaStreamCallback_t;
/* size 0x0c */
struct midaStreamConfig {
    midaStreamPriorityLevel priority;  /* +0x00 */
    midaStreamType type;  /* +0x04 */
    bool drm_protect;  /* +0x08 */
};
typedef struct opaque * midaStream_t;
/* size ? */
struct midaSubgraphConfig {
};
/* size 0x20 */
struct midaSubgraphExecParams {
    uint32_t l2bSize;  /* +0x00 */
    uint32_t vaSize;  /* +0x04 */
    uint32_t regFenceFlag;  /* +0x08 */
    uint32_t regNum;  /* +0x0c */
    midaSubgraphStackInfo stackInfo;  /* +0x10 */
};
typedef struct opaque * midaSubgraphExec_t;
/* size 0x28 */
struct midaSubgraphNodeParams {
    midaSubgraph_t subgraph;  /* +0x00 */
    midaSubgraphExecParams execPara;  /* +0x08 */
};
typedef struct opaque * midaSubgraphNode_t;
/* size 0x10 */
struct midaSubgraphStackInfo {
    uint64_t stack_size;  /* +0x00 */
    midaMemory_t stack_mem;  /* +0x08 */
};
typedef struct opaque * midaSubgraph_t;

/* ---- resolved through struct WrapperFuncTable (dlsym) ---- */
midaError_t midaDeviceGet(midaDevice_t *, uint32_t, midaDeviceConfig *);
midaError_t midaDeviceRelease(midaDevice_t);
midaError_t midaMemoryAlloc(midaMemory_t *, midaDevice_t, uint64_t, uint32_t, uint32_t, const char *);
midaError_t midaMemoryMap(midaMemory_t *, midaDevice_t, int, void *, uint64_t, uint32_t, uint32_t);
midaError_t midaMemoryAddMap(midaMemory_t, uint32_t);
midaError_t midaMemoryUnMap(midaMemory_t);
midaError_t midaMemoryFree(midaMemory_t);
midaError_t midaMemoryCacheStart(midaMemory_t);
midaError_t midaMemoryCacheEnd(midaMemory_t);
midaError_t midaMemoryPartCacheStart(midaMemory_t, uint64_t, uint64_t);
midaError_t midaMemoryPartCacheEnd(midaMemory_t, uint64_t, uint64_t);
void * midaMemoryGetPtr(midaMemory_t);
uint64_t midaMemorySize(midaMemory_t);
uint64_t midaMemoryGetDeviceVa(midaMemory_t);
midaError_t midaGetNPUMemoryUsage(midaMemoryUsage *);
int midaMemoryGetFd(midaMemory_t);
midaError_t midaAddressReserve(midaAddress_t *, midaDevice_t, uint64_t, uint32_t, uint32_t);
midaError_t midaAddressFree(midaAddress_t);
void * midaAddressGetPtr(midaAddress_t);
uint64_t midaAddressGetDeviceVa(midaAddress_t);
uint64_t midaAddressSize(midaAddress_t);
midaError_t midaMemoryAllocWithAddr(midaMemory_t *, uint64_t, uint64_t, midaAddress_t);
midaError_t midaMemoryMapWithAddr(midaMemory_t *, int, uint64_t, uint64_t, midaAddress_t);
midaError_t midaStreamCreate(midaStream_t *, midaDevice_t, midaStreamConfig *);
midaError_t midaStreamDestroy(midaStream_t);
midaError_t midaStreamCancel(midaStream_t);
midaError_t midaStreamLaunchSubgraph(midaSubgraphExec_t, midaLaunchConfig *, midaStream_t);
midaError_t midaStreamLaunchSubgraphAsync(midaSubgraphExec_t, midaLaunchConfig *, midaStream_t);
midaError_t midaStreamLaunchSubgraphs(midaSubgraphExec_t *, uint32_t, midaLaunchConfig *, midaStream_t);
midaError_t midaStreamLaunchSubgraphsAsync(midaSubgraphExec_t *, uint32_t, midaLaunchConfig *, midaStream_t);
midaError_t midaStreamLaunchGraph(midaGraphExec_t, midaLaunchConfig *, midaStream_t);
midaError_t midaStreamLaunchGraphAsync(midaGraphExec_t, midaLaunchConfig *, midaStream_t);
midaError_t midaStreamWaitEvent(midaStream_t, midaEvent_t);
midaError_t midaStreamSynchronize(midaStream_t);
midaError_t midaSubgraphCreate(midaSubgraph_t *, midaDevice_t, midaSubgraphConfig *);
midaError_t midaSubgraphDestroy(midaSubgraph_t);
midaError_t midaSubgraphInstantiate(midaSubgraphExec_t *, midaSubgraph_t, midaSubgraphExecParams *);
midaError_t midaSubgraphDestroyNode(midaSubgraphNode_t);
midaError_t midaSubgraphExecClone(midaSubgraphExec_t *, midaSubgraphExec_t);
midaError_t midaSubgraphExecDestroy(midaSubgraphExec_t);
midaError_t midaSubgraphExecDumpDataToFile(midaSubgraphExec_t, const char *);
midaError_t midaSubgraphExecGetStackInfo(midaSubgraphExec_t, midaSubgraphStackInfo *);
midaError_t midaStreamLaunchSubgraphAsyncWithCallback(midaSubgraphExec_t, midaStreamCallback_t, void *, midaLaunchConfig *, midaStream_t);
midaError_t midaStreamLaunchGraphAsyncWithCallback(midaGraphExec_t, midaStreamCallback_t, void *, midaLaunchConfig *, midaStream_t);
midaError_t midaGraphCreate(midaGraph_t *, midaDevice_t);
midaError_t midaGraphDestroy(midaGraph_t);
midaError_t midaGraphAddSubgraphNode(midaGraphNode_t *, midaGraph_t, const midaGraphNode_t *, uint32_t, midaSubgraphNodeParams *);
midaError_t midaGraphDestroyNode(midaGraphNode_t);
midaError_t midaGraphClone(midaGraph_t *, midaGraph_t);
midaError_t midaGraphInstantiate(midaGraphExec_t *, midaGraph_t);
midaError_t midaGraphExecGetSubgraphExec(midaSubgraphExec_t *, midaGraphExec_t, midaGraphNode_t);
midaError_t midaGraphExecUpdate(midaGraphExec_t, midaGraph_t, midaGraphExecUpdateResultInfo *);
midaError_t midaGraphExecDestroy(midaGraphExec_t);
midaStatus midaDirectIOOpen(midaDirectIO_t *, const char *);
midaStatus midaDirectIORead(midaDirectIO_t, uint64_t, int, uint64_t, uint64_t);
midaStatus midaDirectIOClose(midaDirectIO_t);

#endif