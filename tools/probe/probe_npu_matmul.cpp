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
 *   probe_npu_matmul                     # 576 x 576, 1 token, GemmSplitMN act=0
 *   probe_npu_matmul --gemm --act 0      # plain GEMM, reference = plain matmul
 *   probe_npu_matmul --gemm --act 5      # SiLU-fused kernel, reference = silu(matmul)
 *   probe_npu_matmul --gemm --precheck   # RMSNorm (positive control) then GEMM
 *   probe_npu_matmul --gemm --blocks     # shapes that change the block count
 *   probe_npu_matmul --sweep             # every SmolLM2-135M matmul shape
 *   probe_npu_matmul --gemm-name         # resolve kernel names for act 0/5/6/7
 *   probe_npu_matmul --find-gemm         # bounded per-module kernel lookup
 *   probe_npu_matmul --ops               # rmsnorm + abs/neg/sign
 *
 * The --gemm flag turns on the GemmSplitMN path and fills in the defaults that
 * a single-node subgraph needs (queue id 0, barrier after the kernel, config 2).
 * Anything already in the environment is left untouched, so a one-off override
 * still works:  O3NPU_GEMM_W_VA=0x1 ./probe_npu_matmul --gemm
 */
#include "o3npu.h"

extern "C" int o3npu_rmsnorm_test(int K, int trace);
extern "C" int o3npu_limits(void);
extern "C" int o3npu_modules(void);
extern "C" int o3npu_funcs(void);
extern "C" int o3npu_gemm_name(void);
extern "C" int o3npu_gemm_reorder_size(void);
extern "C" int o3npu_find_gemm(void);
extern "C" int o3npu_unary_test(const char * name, int n, int trace);

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000ull + (uint64_t) ts.tv_nsec / 1000ull;
}

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

/* activation applied by the fused epilogue kernels (xnnlActivationType_t) */
static float act_apply(float x, int act) {
    switch (act) {
        case 5:  return x / (1.0f + expf(-x));                       /* SiLU   */
        case 6:  return 0.5f * x * (1.0f + erff(x * 0.70710678f));   /* GELU   */
        default: return x;                                           /* none   */
    }
}

/* C[m + t*M] = act( sum_k W[k + m*K] * X[k + t*K] ) */
static void cpu_ref(const std::vector<uint16_t> & w, const std::vector<uint16_t> & x,
                    std::vector<float> & out, int K, int M, int T, int act) {
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += half_to_float(w[(size_t) k + (size_t) m * K]) *
                       half_to_float(x[(size_t) k + (size_t) t * K]);
            }
            out[(size_t) m + (size_t) t * M] = act_apply(acc, act);
        }
    }
}

static int run_one(o3npu_matmul_t * mm, int K, int M, int T, int act, int verbose) {
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
    cpu_ref(w, x, ref, K, M, T, act);

    uint64_t t0 = now_us();
    int rc = o3npu_matmul_run(mm, K, M, T, w.data(), x.data(), got.data());
    uint64_t t1 = now_us();
    if (rc != 0) {
        printf("K=%d M=%d T=%d act=%d  FAILED  rc=%d  %s\n", K, M, T, act, rc,
               o3npu_matmul_error(mm));
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
    printf("K=%d M=%d T=%d act=%d  max|NPU-CPU|=%.6g  run_us=%llu  %s\n",
           K, M, T, act, maxdiff, (unsigned long long) (t1 - t0), ok ? "OK" : "MISMATCH");
    if (verbose && !ok) {
        for (size_t i = 0; i < ref.size() && i < 8; i++) {
            printf("   [%zu] ref=%.6f got=%.6f\n", i, ref[i], got[i]);
        }
    }
    return ok ? 0 : 2;
}

static void setenv_if_unset(const char * name, const char * value) {
    if (!getenv(name) || !*getenv(name)) {
        setenv(name, value, 1);
    }
}

/* --gemm: select the GemmSplitMN path and set the single-node defaults.
 * act is the epilogue the reference will also apply. */
static void enable_gemm(int act) {
    setenv("O3NPU_GEMM", "1", 1);
    setenv_if_unset("O3NPU_GEMM_PARAM3", "0");   /* queue id: single node -> 0 */
    {
        char b[16];
        snprintf(b, sizeof b, "%d", act);
        setenv_if_unset("O3NPU_GEMM_PARAM10", b);
    }
    setenv_if_unset("O3NPU_GEMM_PARAM18", "2");  /* configType used by FFN */
    setenv_if_unset("O3NPU_BARRIER", "1");       /* match RMSNorm graph shape */
}

int main(int argc, char ** argv) {
    int K = 576, M = 576, T = 1;
    bool sweep = false;
    bool blocks = false;
    bool rms = false;
    bool ops_all = false;
    bool precheck = false;
    bool gemm = false;
    bool gemm_name = false;
    bool gemm_reorder = false;
    bool find_gemm = false;
    int  act = 0;
    int  transposed = -1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--sweep") {
            sweep = true;
        } else if (a == "--blocks") {
            blocks = true;
        } else if (a == "--rms") {
            rms = true;
        } else if (a == "--ops") {
            ops_all = true;
        } else if (a == "--precheck") {
            precheck = true;
        } else if (a == "--gemm") {
            gemm = true;
        } else if (a == "--gemm-name") {
            gemm_name = true;
        } else if (a == "--gemm-reorder-size") {
            gemm_reorder = true;
        } else if (a == "--find-gemm") {
            find_gemm = true;
        } else if (a == "--act" && i + 1 < argc) {
            act = atoi(argv[++i]);
        } else if (a == "--limits") {
            return o3npu_limits();
        } else if (a == "--modules") {
            return o3npu_modules();
        } else if (a == "--funcs") {
            return o3npu_funcs();
        } else if (a == "--op" && i + 1 < argc) {
            const char * op = argv[++i];
            int rc = o3npu_unary_test(op, K, 1);
            printf("RESULT %s (rc=%d)\n", rc == 0 ? "ok" : "failed", rc);
            return rc == 0 ? 0 : 5;
        } else if (a == "--transposed" && i + 1 < argc) {
            transposed = atoi(argv[++i]);
        } else if (a == "--k" && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (a == "--m" && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (a == "--t" && i + 1 < argc) {
            T = atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            printf("usage: %s [K M T] [--gemm] [--act N] [--precheck] [--blocks]\n"
                   "       %*s [--sweep] [--rms] [--ops] [--transposed 0|1]\n"
                   "       %*s [--gemm-name] [--find-gemm]\n",
                   argv[0], (int) strlen(argv[0]), "", (int) strlen(argv[0]), "");
            return 0;
        }
    }

    if (gemm_name) {
        return o3npu_gemm_name();
    }
    if (gemm_reorder) {
        return o3npu_gemm_reorder_size();
    }
    if (find_gemm) {
        int rc = o3npu_find_gemm();
        printf("RESULT %s (rc=%d)\n", rc == 0 ? "ok" : "failed", rc);
        return rc == 0 ? 0 : 7;
    }

    if (ops_all) {
        int fails = 0;
        struct { const char * name; int rc; } res[8];
        int n = 0;
        res[n].name = "rmsnorm"; res[n].rc = o3npu_rmsnorm_test(K, 1); n++;
        const char * unary[] = { "abs", "neg", "sign" };
        for (int i = 0; i < 3; i++) {
            res[n].name = unary[i];
            res[n].rc = o3npu_unary_test(unary[i], K, 1);
            n++;
        }
        printf("\n==== operator summary ====\n");
        for (int i = 0; i < n; i++) {
            printf("%-8s : %s (rc=%d)\n", res[i].name,
                   res[i].rc == 0 ? "PASS" : "FAIL", res[i].rc);
            if (res[i].rc != 0) fails++;
        }
        printf("=========================\n");
        return fails == 0 ? 0 : 6;
    }

    if (rms) {
        printf("== rmsnorm probe K=%d ==\n", K);
        int rc = o3npu_rmsnorm_test(K, 1);
        printf("RESULT %s (rc=%d)\n", rc == 0 ? "ok" : "failed", rc);
        return rc == 0 ? 0 : 4;
    }

    if (transposed >= 0) {
        char buf[8];
        snprintf(buf, sizeof buf, "%d", transposed ? 1 : 0);
        setenv("O3NPU_TRANSPOSED_W", buf, 1);
    }
    if (gemm) {
        enable_gemm(act);
    } else if (act != 0) {
        /* --act without --gemm still means "use this epilogue" */
        char b[16];
        snprintf(b, sizeof b, "%d", act);
        setenv("O3NPU_GEMM_PARAM10", b, 1);
    }

    const char * wl = "ggml";
    const char * wle = getenv("O3NPU_TRANSPOSED_W");
    if (wle && wle[0] == '0') {
        wl = "xnnl";
    }
    printf("== probe_npu_matmul (weight layout: %s, path: %s, act: %d) ==\n",
           wl, gemm ? "gemmSplitMN" : "xnnlMatmul", act);

    if (precheck) {
        printf("== precheck: rmsnorm (positive control) ==\n");
        int prc = o3npu_rmsnorm_test(K, 1);
        printf("precheck rmsnorm rc=%d (%s)\n", prc, prc == 0 ? "PASS" : "FAIL");
        fflush(stdout);
    }

    o3npu_matmul_t * mm = o3npu_matmul_open();
    if (!mm) {
        printf("RESULT open_failed: %s\n", "o3npu_matmul_open returned NULL");
        return 3;
    }

    int fails = 0;
    if (blocks) {
        /* Shapes chosen to change only the block geometry the kernel picks:
         * blockN is 256, so M<=256 is a single N block (gridDim.y=1) and
         * M=576 is three.  If only the single-block shapes write back, the
         * fault is in the multi-block dispatch, not the node contract. */
        struct { int K, M; const char * why; } shapes[] = {
            { 576, 192, "N=192 single block" },
            { 576, 256, "N=256 single block" },
            { 576, 576, "N=576 three blocks" },
            {  64,  64, "tiny 64x64" },
        };
        for (auto & s : shapes) {
            printf("-- block sweep: %s --\n", s.why);
            fails += run_one(mm, s.K, s.M, T, act, 1);
        }
    } else if (sweep) {
        struct { int K, M; } shapes[] = {
            { 576,  576 },   /* q_proj  */
            { 576,  192 },   /* k_proj / v_proj */
            { 576,  576 },   /* o_proj  */
            { 576, 1536 },   /* gate/up */
            { 1536, 576 },   /* down    */
            { 576, 49152 },  /* lm_head */
        };
        for (auto & s : shapes) {
            fails += run_one(mm, s.K, s.M, T, act, 1);
        }
    } else {
        fails += run_one(mm, K, M, T, act, 1);
    }

    o3npu_matmul_close(mm);
    printf("RESULT %s (%d failures)\n", fails == 0 ? "ok" : "failed", fails);
    return fails == 0 ? 0 : 4;
}
