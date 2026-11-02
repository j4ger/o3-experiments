#!/usr/bin/env bash
# Offline acceptance checks: everything that must be true before touching
# the device again.
#
#     tools/npu/offline-check.sh
#
# No adb, no NPU.  Verifies:
#   - the extracted vendor libraries contain the expected symbols
#   - libxnnl.so still contains the GemmSplitMN kernels
#   - the Nix probe can be built without network access
#   - the device-test commands and docs are present

set -uo pipefail
cd "$(dirname "$0")/../.."

fail=0
note() { printf '%s\n' "$*"; }
ok()   { printf '  ok   %s\n' "$*"; }
bad()  { printf '  FAIL %s\n' "$*"; fail=1; }

check_symbol() { # <lib> <symbol>
    local lib="$1" sym="$2"
    if [[ ! -f "$lib" ]]; then
        bad "$lib missing"
        return
    fi
    if nm -D --defined-only "$lib" 2>/dev/null | grep -q "[[:space:]]$sym\$"; then
        ok "$(basename "$lib"): $sym"
    else
        bad "$(basename "$lib"): $sym"
    fi
}

note "== extracted artifacts =="
for f in artifacts/vendor/libxnnl.so artifacts/vendor/libxnumd.so \
         artifacts/device/20261103-015444/manifest.json \
         artifacts/device/20261103-015444/tmp/gemm.dot \
         artifacts/probe-logs/2026-11-03-barrier/gemm_barrier.dot; do
    if [[ -f "$f" ]]; then ok "$f"; else bad "$f"; fi
done

note "== libxnnl symbols =="
for s in xnnlAddGemmSplitMNKernelNode xnnlGetGemmSplitMNKernelName \
         xnnlGemmSplitMNKernelReorder xnnlGetGemmSplitMNKernelReorderSize; do
    check_symbol artifacts/vendor/libxnnl.so "$s"
done

note "== libxnumd symbols =="
for s in midaSubgraphAddBarrierNode midaSubgraphAddKernelNode \
         midaSubgraphAddL2bAllocNode midaSubgraphAddL2bReadNode \
         midaSubgraphExecToVisualize; do
    check_symbol artifacts/vendor/libxnumd.so "$s"
done

note "== embedded GemmSplitMN kernels =="
if node tools/npu/list-kernels.mjs artifacts/vendor/libxnnl.so --json > /tmp/o3-kernels.json 2>/dev/null; then
    for k in gemmSplitMNEpilogueFusedKernel \
             gemmSplitMNEpilogueFusedWithSiluKernel \
             gemmSplitMNEpilogueFusedWithGwKernel; do
        if grep -q "$k" /tmp/o3-kernels.json; then ok "$k"; else bad "$k"; fi
    done
    node tools/npu/list-kernels.mjs artifacts/vendor/libxnnl.so 2>/dev/null \
        | grep -E 'kernel modules|kernel names' | sed 's/^/  /'
else
    bad "list-kernels.mjs failed"
fi

note "== Nix probe build (offline) =="
if nix build .#probe-npu-matmul --offline 2>/tmp/o3-nix.err; then
    ok "nix build .#probe-npu-matmul"
    if [[ -x result/bin/probe_npu_matmul ]]; then
        ok "$(readlink -f result/bin/probe_npu_matmul)"
        sha256sum result/bin/probe_npu_matmul | sed 's/^/  /'
    else
        bad "result/bin/probe_npu_matmul missing"
    fi
else
    bad "nix build .#probe-npu-matmul"
    tail -5 /tmp/o3-nix.err | sed 's/^/  /'
fi

note "== documentation =="
for d in docs/DEVICE-EXTRACTION.md docs/OFFLINE-READINESS.md \
         docs/NPU-GEMM-NEXT.md docs/NPU-GEMM-VENDOR-PATH.md; do
    if [[ -f "$d" ]]; then ok "$d"; else bad "$d"; fi
done

note
note "Device commands (when adb is back):"
cat <<'EOF'
  nix develop -c adb push result/bin/probe_npu_matmul /data/local/tmp/
  nix develop -c adb shell "/data/local/tmp/o3p_chat.bin --shell 'cd /data/local/tmp; \
    O3NPU_MEM_FLAG=5 O3NPU_OUT_PATTERN=1 O3NPU_BARRIER=1 \
    O3NPU_GEMM=1 O3NPU_GEMM_PARAM3=0 O3NPU_GEMM_PARAM10=5 \
    O3NPU_GEMM_PARAM18=2 O3NPU_TRANSPOSED_W=1 \
    ./probe_npu_matmul --k 576 --m 576 --t 1 2>&1'"
EOF
note
if [[ $fail -eq 0 ]]; then
    note "offline-check: PASS"
else
    note "offline-check: FAIL"
fi
exit $fail
