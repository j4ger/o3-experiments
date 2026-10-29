/*
 * probe_npu_matmul - run one xnnl matmul on the NPU through the o3npu runtime
 * and compare it with a CPU reference.
 *
 * This exercises exactly the code path that the ggml backend uses
 * (tools/npu/o3npu.c: o3npu_matmul_open / o3npu_matmul_run), so a pass here
 * means the backend path is sound.
 *
 * Build: tools/probe/build.sh
 *
 * Usage:
 *   probe_npu_matmul                     # 576 x 576, 1 token
 *   probe_npu_matmul 576 1536 1          # K M T
 *   probe_npu_matmul --sweep             # every SmolLM2-135M matmul shape
 *   probe_npu_matmul --sweep --transposed 0
 *
 * Env knobs are read by o3npu.c: O3NPU_TRACE=1, O3NPU_TRANSPOSED_W=0|1,
 * O3NPU_L2B=..., O3NPU_VA=..., O3NPU_CU=..., O3NPU_STRIPE=cu0,cu1,...
 */
#include "o3npu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static float frand(void) {
    return (float) (rand() % 2000 - 1000) / 1000.0f;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* C[m + t*M] = sum_k W[k + m*K] * X[k + t*K] */
static uint16_t float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t) ((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        return (uint16_t) sign;
    }
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

static void cpu_ref(const std::vector<uint16_t> & w, const std::vector<uint16_t> & x,
                    std::vector<float> & out, int K, int M, int T) {
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += half_to_float(w[(size_t) k + (size_t) m * K]) *
                       half_to_float(x[(size_t) k + (size_t) t * K]);
            }
            out[(size_t) m + (size_t) t * M] = acc;
        }
    }
}

static int run_one(o3npu_matmul_t * mm, int K, int M, int T, int verbose) {
    std::vector<uint16_t> w((size_t) K * M);
    std::vector<uint16_t> x((size_t) K * T);
    std::vector<float>    ref((size_t) M * T, 0.0f);
    std::vector<float>    got((size_t) M * T, 0.0f);

    for (auto & v : w) {
        v = float_to_half(frand());
    }
    for (auto & v : x) {
        v = float_to_half(frand());
    }
    cpu_ref(w, x, ref, K, M, T);

    int rc = o3npu_matmul_run(mm, K, M, T, w.data(), x.data(), got.data());
    if (rc != 0) {
        printf("K=%d M=%d T=%d  FAILED  rc=%d  %s\n", K, M, T, rc, o3npu_matmul_error(mm));
        return 1;
    }

    double maxdiff = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = std::fabs((double) ref[i] - (double) got[i]);
        if (d > maxdiff) {
            maxdiff = d;
        }
    }
    int ok = maxdiff < 2e-2;
    printf("K=%d M=%d T=%d  max|NPU-CPU|=%.6g  %s\n", K, M, T, maxdiff, ok ? "OK" : "MISMATCH");
    if (verbose && !ok) {
        for (size_t i = 0; i < ref.size() && i < 8; i++) {
            printf("   [%zu] ref=%.6f got=%.6f\n", i, ref[i], got[i]);
        }
    }
    return ok ? 0 : 2;
}

int main(int argc, char ** argv) {
    int K = 576, M = 576, T = 1;
    bool sweep = false;
    int  transposed = -1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--sweep") {
            sweep = true;
        } else if (a == "--transposed" && i + 1 < argc) {
            transposed = atoi(argv[++i]);
        } else if (a == "--k" && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (a == "--m" && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (a == "--t" && i + 1 < argc) {
            T = atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            printf("usage: %s [K M T] [--sweep] [--transposed 0|1]\n", argv[0]);
            return 0;
        }
    }

    if (transposed >= 0) {
        char buf[8];
        snprintf(buf, sizeof buf, "%d", transposed ? 1 : 0);
        setenv("O3NPU_TRANSPOSED_W", buf, 1);
    }

    const char * wl = "ggml";
    const char * wle = getenv("O3NPU_TRANSPOSED_W");
    if (wle && wle[0] == '0') {
        wl = "xnnl";
    }
    printf("== probe_npu_matmul (weight layout: %s) ==\n", wl);

    o3npu_matmul_t * mm = o3npu_matmul_open();
    if (!mm) {
        printf("RESULT open_failed: %s\n", "o3npu_matmul_open returned NULL");
        return 3;
    }

    int fails = 0;
    if (sweep) {
        struct { int K, M; } shapes[] = {
            { 576,  576 },   /* q_proj  */
            { 576,  192 },   /* k_proj / v_proj */
            { 576,  576 },   /* o_proj  */
            { 576, 1536 },   /* gate/up */
            { 1536, 576 },   /* down    */
            { 576, 49152 },  /* lm_head */
        };
        for (auto & s : shapes) {
            fails += run_one(mm, s.K, s.M, T, 1);
        }
    } else {
        fails += run_one(mm, K, M, T, 1);
    }

    o3npu_matmul_close(mm);
    printf("RESULT %s (%d failures)\n", fails == 0 ? "ok" : "failed", fails);
    return fails == 0 ? 0 : 4;
}
