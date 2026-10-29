/*
 * o3npu - driver layer for the Xiaomi 18 Fold (lhasa) NPU.
 *
 * Stack:
 *     your code
 *       -> libxnnl   (matmul graph builder; 686 precompiled kernels embedded)
 *         -> mida    (subgraph, instantiate, stream launch, memory)
 *           -> libxnumd (UMD)
 *             -> /dev/hpc-cdev
 *
 * Requirements: uid 0 in a domain that may open /dev/hpc-cdev, /dev/hpc-heap,
 * /dev/hpc-rpmsg, the dma heaps, /dev/timestamp and /dev/xr_dmabuf_helper, and
 * read vendor_npu_prop. See docs/NPU-ACCESS rules in apps/o3root RootChain.java.
 */
#ifndef O3NPU_H
#define O3NPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct o3npu o3npu_t;

/* Open the mida device and, when available, an xnnl handle on top of it. */
o3npu_t *o3npu_open(void);
void o3npu_close(o3npu_t *n);

/*
 * Allocate NPU memory. cpu_visible != 0 asks for host-mappable memory
 * (midaMemoryAlloc flag 1), for which o3npu_cpu_ptr returns a real pointer.
 */
void *o3npu_alloc(o3npu_t *n, size_t size, int cpu_visible, size_t *out_size);
void o3npu_free(o3npu_t *n, void *mem);

void *o3npu_cpu_ptr(o3npu_t *n, void *mem);
uint64_t o3npu_dev_va(o3npu_t *n, void *mem);
int o3npu_mem_fd(o3npu_t *n, void *mem);

/* Runs every layer of the stack and returns 0 on success. */
int o3npu_selftest(void);

/* ------------------------------------------------------------------ */
/* xnnl matmul                                                         */
/* ------------------------------------------------------------------ */

typedef struct o3npu_matmul o3npu_matmul_t;

/* Opens the device, loads the embedded kernel modules and creates a stream. */
o3npu_matmul_t *o3npu_matmul_open(void);
void o3npu_matmul_close(o3npu_matmul_t *m);

/*
 * C[m + t*M] = sum_k W[k + m*K] * X[k + t*K]
 *
 * W and X are fp16 in ggml layout (ne0 fastest); out is fp32, same layout as
 * ggml's dst.  T is the number of tokens (xnnl "maxBatch").
 *
 * Returns 0 on success, a negative status otherwise.
 */
int o3npu_matmul_run(o3npu_matmul_t *m, int K, int M, int T,
                     const uint16_t *w_f16,
                     const uint16_t *x_f16,
                     float *out_f32);

const char *o3npu_matmul_error(const o3npu_matmul_t *m);
void o3npu_matmul_set_trace(int on);

#ifdef __cplusplus
}
#endif

#endif /* O3NPU_H */
