/* mida config structures recovered from the vendor library libxnumd.so
 * (pulled from /vendor/lib64 on the device, world-readable, no root needed)
 * by Ghidra-decompiling its consumers.  Field offsets and the accepted value
 * ranges come from the implementation's own checks; names are inferred.
 */
#ifndef MIDA_CONFIGS_H
#define MIDA_CONFIGS_H
#include <stdint.h>

/* Consumer: npu::umd::DeviceImpl::Init(uint index, midaDeviceConfig*) in
 * vendor/xring/ai/xnrt/kernel_runtime/device/device_impl.cpp
 * midaDeviceGet(out, index, cfg) requires cfg != NULL and checks cfg+8 < 3. */
typedef struct {
    uint32_t type;          /* +0x00  must be < 6                       */
    uint32_t reserved04;    /* +0x04  (not read by DeviceImpl::Init)    */
    uint32_t mode;          /* +0x08  must be < 3  -> DeviceImpl+0x160  */
    uint32_t flag;          /* +0x0c  != 0 becomes bool DeviceImpl+0x164*/
    uint32_t count;         /* +0x10  must be < 0x15 (21) -> +0x16c     */
    uint32_t dsmem_large;   /* +0x14  -> byte DeviceImpl+0x170          */
    uint32_t power_policy;  /* +0x18  < 2 (0=PERFORMANCE, else fallback) */
} midaDeviceConfig;         /* size >= 0x1c                             */

/* Consumer: npu::umd::StreamImpl::Create(StreamImpl*, midaStreamConfig*)
 * midaStreamCreate(out, device, cfg) requires cfg != NULL. */
typedef struct {
    uint32_t priority;      /* +0x00  must be < 4                       */
    uint32_t mode;          /* +0x04  must be < 3                       */
    uint8_t  flag;          /* +0x08  -> StreamImpl+0x28                */
    uint8_t  pad[7];
} midaStreamConfig;         /* size >= 9                                */

/* Consumer: npu::umd::SubgraphImpl::SubgraphImpl(SubgraphImpl*, DeviceImpl*,
 * midaSubgraphConfig*).  midaSubgraphCreate accepts cfg == NULL. */
typedef struct {
    uint32_t enable;        /* +0x00  if != 0 the parameter block is read */
    uint32_t pad04;
    uint64_t param08;       /* +0x08  -> SubgraphImpl+0x188              */
    uint64_t param10;       /* +0x10                                     */
    uint64_t param18;       /* +0x18                                     */
    uint64_t param20;       /* +0x20                                     */
    uint64_t param28;       /* +0x28                                     */
    uint64_t reserved[2];   /* +0x30, +0x38                              */
    uint64_t param40;       /* +0x40                                     */
    uint64_t reserved48;    /* +0x48                                     */
    uint64_t param50;       /* +0x50                                     */
    uint64_t param58;       /* +0x58                                     */
    uint8_t  reserved60[0x30]; /* up to +0x90                           */
    uint8_t  flag90;        /* +0x90  -> SubgraphImpl+0x36c              */
    uint8_t  flag91;        /* +0x91  -> SubgraphImpl+0x36f              */
} midaSubgraphConfig;       /* size >= 0x92                              */

/* A zeroed midaDeviceConfig / midaStreamConfig passes every observed check,
 * so a first on-device probe can pass {0} for both. */
#endif
