/*
 * o3npu - driver layer for the Xiaomi 18 Fold (lhasa) NPU.
 * See o3npu.c for the stack and the required SELinux rules.
 */
#ifndef O3NPU_H
#define O3NPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct o3npu o3npu_t;

/* Open the mida device and, when available, an xnnl handle on top of it. */
o3npu_t *o3npu_open(void);
void o3npu_close(o3npu_t *n);

/*
 * Allocate NPU memory. cpu_visible != 0 asks for host-mappable memory
 * (midaMemoryAlloc flag 1), for which o3npu_cpu_ptr returns a real pointer.
 * cpu_visible == 0 yields an NPU-side allocation only.
 */
void *o3npu_alloc(o3npu_t *n, size_t size, int cpu_visible, size_t *out_size);
void o3npu_free(o3npu_t *n, void *mem);

void *o3npu_cpu_ptr(o3npu_t *n, void *mem);
uint64_t o3npu_dev_va(o3npu_t *n, void *mem);
int o3npu_mem_fd(o3npu_t *n, void *mem);

/* Runs every layer of the stack and returns 0 on success. */
int o3npu_selftest(void);

#endif
