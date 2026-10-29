/*
 * Offline check: our reference matmul against ggml's CPU implementation.
 * Build and run on the host; no device involved.
 */
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int o3npu_debug_matmul(const struct ggml_tensor * a, const struct ggml_tensor * b, struct ggml_tensor * d);
extern "C" int o3npu_debug_copy(struct ggml_tensor * d, const struct ggml_tensor * s);

static float frand(void) {
    return (float) (rand() % 2000 - 1000) / 1000.0f;
}

static int run_case_t(int64_t K, int64_t M, int64_t N, enum ggml_type ta, enum ggml_type tb) {
    struct ggml_init_params ip = { 256u * 1024u * 1024u, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return 1;
    }

    /* raw storage is [M, K]; a transposed view makes it the [K, M] operand */
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, ta, K, M);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, tb, K, N);

    for (int64_t i = 0; i < ggml_nelements(a); i++) {
        const float v = frand();
        if (ta == GGML_TYPE_F16) ((ggml_fp16_t *) a->data)[i] = ggml_fp32_to_fp16(v);
        else ((float *) a->data)[i] = v;
    }
    for (int64_t i = 0; i < ggml_nelements(b); i++) {
        const float v = frand();
        if (tb == GGML_TYPE_F16) ((ggml_fp16_t *) b->data)[i] = ggml_fp32_to_fp16(v);
        else ((float *) b->data)[i] = v;
    }

    /* reference: ggml's CPU matmul */
    struct ggml_tensor * ref = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        printf("  ggml compute failed\n");
        ggml_free(ctx);
        return 1;
    }

    /* ours */
    struct ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    memset(out->data, 0, ggml_nbytes(out));
    if (o3npu_debug_matmul(a, b, out) != 0) {
        printf("  o3npu matmul failed\n");
        ggml_free(ctx);
        return 1;
    }

    double maxdiff = 0.0;
    for (int64_t i = 0; i < M * N; i++) {
        double d = fabs((double) ((float *) ref->data)[i] - (double) ((float *) out->data)[i]);
        if (d > maxdiff) {
            maxdiff = d;
        }
    }
    printf("  K=%4lld M=%4lld N=%4lld a=%d b=%d  max|diff|=%.6g  %s\n",
           (long long) K, (long long) M, (long long) N, (int) ta, (int) tb,
           maxdiff, maxdiff < 1e-2 ? "OK" : "MISMATCH");
    ggml_free(ctx);
    return maxdiff < 1e-2 ? 0 : 2;
}

static int run_copy_case(enum ggml_type dst_type, bool transposed_src) {
    struct ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;

    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 5);
    for (int64_t i = 0; i < ggml_nelements(src); i++) ((float *) src->data)[i] = frand();

    struct ggml_tensor * view = transposed_src ? ggml_transpose(ctx, src) : src;

    struct ggml_tensor * ref = ggml_new_tensor(ctx, dst_type, ggml_n_dims(view), view->ne);
    struct ggml_tensor * cpy = ggml_cpy(ctx, view, ref);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, cpy);
    if (ggml_graph_compute_with_ctx(ctx, g, 1) != GGML_STATUS_SUCCESS) {
        printf("  ggml cpy compute failed\n");
        ggml_free(ctx);
        return 1;
    }

    struct ggml_tensor * out = ggml_new_tensor(ctx, dst_type, ggml_n_dims(view), view->ne);
    memset(out->data, 0, ggml_nbytes(out));
    if (o3npu_debug_copy(out, view) != 0) { ggml_free(ctx); return 1; }

    const int64_t n = ggml_nelements(view);
    double maxdiff = 0.0;
    for (int64_t i = 0; i < n; i++) {
        float a = dst_type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ggml_fp16_t *) ref->data)[i]) : ((float *) ref->data)[i];
        float b = dst_type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ggml_fp16_t *) out->data)[i]) : ((float *) out->data)[i];
        double d = fabs((double) a - (double) b);
        if (d > maxdiff) maxdiff = d;
    }
    printf("  copy -> %s transposed=%d max|diff|=%.6g %s\n",
           dst_type == GGML_TYPE_F16 ? "f16" : "f32", (int) transposed_src, maxdiff,
           maxdiff < 1e-2 ? "OK" : "MISMATCH");
    ggml_free(ctx);
    return maxdiff < 1e-2 ? 0 : 2;
}

int main(void) {
    srand(1234);
    int bad = 0;
    bad |= run_case_t(64, 32, 8, GGML_TYPE_F32, GGML_TYPE_F32);
    bad |= run_case_t(576, 64, 16, GGML_TYPE_F32, GGML_TYPE_F32);
    bad |= run_case_t(576, 64, 16, GGML_TYPE_F16, GGML_TYPE_F32);
    bad |= run_case_t(576, 576, 4, GGML_TYPE_F16, GGML_TYPE_F32);
    bad |= run_case_t(1536, 128, 4, GGML_TYPE_F16, GGML_TYPE_F16);
    bad |= run_copy_case(GGML_TYPE_F32, false);
    bad |= run_copy_case(GGML_TYPE_F16, false);
    bad |= run_copy_case(GGML_TYPE_F32, true);
    printf("%s\n", bad ? "RESULT fail" : "RESULT ok");
    return bad ? 1 : 0;
}
