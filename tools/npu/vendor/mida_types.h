/*
 * mida_types.h - compile-ready mirrors of the mida structs recovered from the
 * vendor libraries.
 *
 * Source of truth:
 *   docs/generated/mida_api.h      (struct sizes/offsets, from libxnlm DWARF)
 *   docs/generated/mida_configs.h  (accepted value ranges, from libxnumd code)
 *
 * The generated headers are documentation: they use names before defining them
 * and omit several enums, so they cannot be included directly.  This file keeps
 * the same layout and adds the missing scalar types.
 */
#ifndef O3_MIDA_TYPES_H
#define O3_MIDA_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* opaque handles */
typedef void * midaDevice_t;
typedef void * midaAddress_t;
typedef void * midaMemory_t;
typedef void * midaSubgraph_t;
typedef void * midaSubgraphExec_t;
typedef void * midaSubgraphNode_t;
typedef void * midaGraph_t;
typedef void * midaGraphNode_t;
typedef void * midaGraphExec_t;
typedef void * midaStream_t;
typedef void * midaEvent_t;
typedef void * midaDirectIO_t;

/* midaError_t / midaStatus are plain ints on the ABI boundary */
typedef int midaError_t;
typedef int midaStatus;

/* size 0x1c; every field must satisfy the range checks in mida_configs.h */
typedef struct midaDeviceConfig {
    uint32_t type;          /* +0x00  < 6                          */
    uint32_t reserved04;    /* +0x04                               */
    uint32_t mode;          /* +0x08  < 3                          */
    uint32_t flag;          /* +0x0c                               */
    uint32_t count;         /* +0x10  < 0x15                       */
    uint32_t dsmem_large;   /* +0x14                               */
    uint32_t power_policy;  /* +0x18  < 2                          */
} midaDeviceConfig;

/* size 0x0c */
typedef struct midaStreamConfig {
    uint32_t priority;      /* +0x00  < 4                          */
    uint32_t type;          /* +0x04  < 3   (1 is the accepted one) */
    uint8_t  drm_protect;   /* +0x08                               */
    uint8_t  pad[3];
} midaStreamConfig;

/* size 0xa0; all-zero is accepted (see probe_mida_exec.c) */
typedef struct midaSubgraphConfig {
    uint8_t bytes[0xa0];
} midaSubgraphConfig;

/* size 0x10 */
typedef struct midaSubgraphStackInfo {
    uint64_t      stack_size;  /* +0x00 */
    midaMemory_t  stack_mem;   /* +0x08 */
} midaSubgraphStackInfo;

/* size 0x20 */
typedef struct midaSubgraphExecParams {
    uint32_t l2bSize;       /* +0x00 */
    uint32_t vaSize;        /* +0x04 */
    uint32_t regFenceFlag;  /* +0x08 */
    uint32_t regNum;        /* +0x0c */
    midaSubgraphStackInfo stackInfo; /* +0x10 */
} midaSubgraphExecParams;

/* size 0x28 */
typedef struct midaSubgraphNodeParams {
    midaSubgraph_t        subgraph; /* +0x00 */
    midaSubgraphExecParams execPara;/* +0x08 */
} midaSubgraphNodeParams;

/* size 0xb8 */
typedef struct midaLaunchConfig {
    uint32_t    cu_num;           /* +0x00 */
    uint32_t    performance_level;/* +0x04 */
    float       ddr_ratio;        /* +0x08 */
    uint32_t    fw_timeout;       /* +0x0c */
    const char *name;             /* +0x10 */
    char        stripe_names[32][5]; /* +0x18 */
} midaLaunchConfig;

#ifdef __cplusplus
}
#endif

#endif /* O3_MIDA_TYPES_H */
