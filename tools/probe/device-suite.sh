#!/usr/bin/env bash
# One device session, one batch of *discriminating* experiments for GemmSplitMN.
#
#     tools/probe/device-suite.sh [--only NAME ...]
#
# This is the "one unlock, one batch" runner: it pushes the probe (and the CDN
# libxnnl for one experiment), runs the ordered matrix below, and captures
# logcat + dmesg per experiment into artifacts/probe-logs/<stamp>-gemm-suite/.
# Each experiment is independent, so one crash does not stop the batch.
#
# Design rationale (see docs/NPU-GEMM-NEXT.md "判别实验包"):
#   the open question is *whether the GemmSplitMN kernel is dispatched at all*.
#   These experiments are chosen to falsify that, not to sweep parameters:
#     - E1/E2 fix the reference so act=0 (plain GEMM) and act=5 (SiLU) both
#       compare apples to apples; act=0 had never been run after the arg fix.
#     - E3 runs RMSNorm (known-good) then GEMM in one process: if RMSNorm writes
#       and GEMM does not, the problem is the node, not the process.
#     - E4/E5 point a box at VA 0x1.  A real TMA must fault/reject; a silently
#       dropped task still reports success.  This is the decisive discriminator.
#     - E6 changes only the block count (blockN=256) so a multi-block dispatch
#       bug would show up as "single block writes, multi block does not".
#     - E7 re-runs the same GEMM against the model-package libxnnl (66 modules)
#       that MiMo actually uses; it was never retested after the arg-order fix.
#     - E8/E9 resolve names and do a bounded per-module lookup (last, because a
#       full-module scan once wedged the device; O3NPU_FIND_MAX caps it).
#
# Requires: rooted device, the six SELinux rules from apps/o3root applied,
# /data/local/tmp/o3p_chat.bin present (installed by the o3root app).
set -uo pipefail
cd "$(dirname "$0")/../.."

stamp="$(date +%Y%m%d-%H%M%S)"
out="artifacts/probe-logs/${stamp}-gemm-suite"
mkdir -p "$out"

TMP=/data/local/tmp
PROBE=probe_npu_matmul_suite
CDN_LIB=$TMP/libxnnl_cdn.so
ADB=(nix develop -c adb)

only=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) shift; while [[ $# -gt 0 && "$1" != --* ]]; do only+=("$1"); shift; done ;;
        *) echo "usage: $0 [--only NAME ...]" >&2; exit 2 ;;
    esac
done
selected() { # selected <name>
    [[ ${#only[@]} -eq 0 ]] && return 0
    local n; for n in "${only[@]}"; do [[ "$n" == "$1" ]] && return 0; done
    return 1
}

run_root() {  # run_root "<shell command>"
    "${ADB[@]}" shell "$TMP/o3p_chat.bin --shell '$1'" 2>&1 \
        | grep -avE '^  adb:|^  apktool:|^  jadx:|^Android RE shell'
}

dmesg_npu() {  # dmesg_npu <file>  (root via the o3p bridge)
    run_root 'dmesg' 2>/dev/null \
        | grep -aiE 'npu|xr_dfx|hpc|firmware response|LOCKUP' > "$1" || true
}

summary="$out/RESULTS.md"
echo "# GemmSplitMN 判别实验包 — $stamp" > "$summary"
echo > "$summary"
echo "| # | 实验 | rc | 结果行 |" >> "$summary"
echo "|---|---|---|---|" >> "$summary"

# ---------------------------------------------------------------- device state
echo "== device =="
"${ADB[@]}" devices -l | tee "$out/devices.txt"
"${ADB[@]}" shell getprop > "$out/getprop.txt" 2>&1 || true
grep -aE 'ro\.soc|ro\.board|ro\.hardware' "$out/getprop.txt" | head -10
"${ADB[@]}" shell 'ls -lZ /dev/hpc-cdev /dev/hpc-heap /dev/hpc-rpmsg 2>&1' > "$out/dev.txt" 2>&1 || true
"${ADB[@]}" shell 'uname -a; cat /proc/uptime' > "$out/uptime.txt" 2>&1 || true
dmesg_npu "$out/dmesg-start.txt"

# ---------------------------------------------------------------- push payload
if [[ -z "${O3NPU_SKIP_BUILD:-}" ]]; then
    echo "== building probe =="
    tools/probe/build.sh /tmp/probe_npu_matmul_suite
fi
echo "== push =="
"${ADB[@]}" push /tmp/probe_npu_matmul_suite "$TMP/$PROBE" >/dev/null
"${ADB[@]}" shell chmod 755 "$TMP/$PROBE"
if [[ -f artifacts/cdn/libxnnl.so ]]; then
    "${ADB[@]}" push artifacts/cdn/libxnnl.so "$CDN_LIB" >/dev/null
    echo "pushed CDN libxnnl -> $CDN_LIB"
else
    echo "WARN: artifacts/cdn/libxnnl.so missing; E7 will be skipped"
fi

# ---------------------------------------------------------------- preflight
echo "== preflight (root + selinux + o3p bridge) =="
run_root 'id; getenforce; ls -l /data/local/tmp/o3p_chat.bin 2>&1' > "$out/preflight.txt" 2>&1 || true
cat "$out/preflight.txt"
if ! grep -aq 'uid=0' "$out/preflight.txt"; then
    echo "WARN: not root through the o3p bridge; experiments will fail. Apply root + rules first."
fi

# ---------------------------------------------------------------- helpers
row=0
run_exp() { # run_exp <id> <name> <envs> <probe args...>
    local id=$1 name=$2 envs=$3; shift 3
    selected "$id" || return 0
    row=$((row + 1))
    echo
    echo "== [$id] $name =="
    "${ADB[@]}" logcat -b all -c >/dev/null 2>&1 || true
    local cmd="cd $TMP; O3NPU_MEM_FLAG=5 O3NPU_OUT_PATTERN=1 O3NPU_TRACE=1 $envs ./$PROBE $* 2>&1; echo __RC=\$?"
    local rc
    run_root "$cmd" | tee "$out/$id-$name.txt"
    rc="$(grep -ao '__RC=[0-9]*' "$out/$id-$name.txt" | tail -1 | cut -d= -f2)"
    [[ -z "$rc" ]] && rc=99
    "${ADB[@]}" logcat -b all -d > "$out/$id-$name.logcat" 2>&1 || true
    dmesg_npu "$out/$id-$name.dmesg"
    local res
    res="$(grep -aE 'max|NPU-CPU|RESULT |timing |mismatch|FAILED|gemm_name|find-gemm' "$out/$id-$name.txt" | tr '\n' ' ' | cut -c1-180)"
    echo "| $id | $name | $rc | $res |" >> "$summary"
    if [[ -s "$out/$id-$name.dmesg" ]]; then
        echo "  !! NPU/kernel log present, see $id-$name.dmesg" 
    fi
    return 0
}

echo
echo "================ experiment matrix ================"

# E0: operator positive control (RMSNorm must still pass)
run_exp e0 ops-control "" --ops --k 576

# E1: plain GEMM, reference = plain matmul (the MUL_MAT semantic)
run_exp e1 act0-plain "" --gemm --act 0 --k 576 --m 576 --t 1

# E2: SiLU-fused kernel, reference = silu(matmul)
run_exp e2 act5-silu "" --gemm --act 5 --k 576 --m 576 --t 1

# E3: RMSNorm then GEMM in one process (positive control inside the run)
run_exp e3 precheck "" --gemm --act 0 --precheck --k 576 --m 576 --t 1

# E4: weight VA is bogus -> does the firmware actually dereference it?
run_exp e4 poison-w "O3NPU_GEMM_W_VA=0x1" --gemm --act 0 --k 576 --m 576 --t 1

# E5: output VA is bogus -> does the kernel actually write?
run_exp e5 poison-out "O3NPU_GEMM_OUT_VA=0x1" --gemm --act 0 --k 576 --m 576 --t 1

# E6: block geometry sweep (single block vs three blocks vs tiny)
run_exp e6 blocks "" --gemm --act 0 --blocks --t 1

# E7: same GEMM against the model-package libxnnl (66 modules)
if "${ADB[@]}" shell "test -f $CDN_LIB" >/dev/null 2>&1; then
    run_exp e7 cdn-lib "O3NPU_XNNL_LIB=$CDN_LIB" --gemm --act 0 --k 576 --m 576 --t 1
else
    echo "skip E7 (no CDN lib on device)"
fi

# E8: resolve kernel names for act 0/5/6/7 and head variants (read-only)
run_exp e8 gemm-name "" --gemm-name

# E9: bounded per-module lookup, capped to 8 modules (last, can wedge)
run_exp e9 find-gemm "O3NPU_FIND_MAX=8" --find-gemm

dmesg_npu "$out/dmesg-end.txt"

echo
echo "================ done ================"
echo "logs:  $out"
echo "table: $summary"
echo
echo "How to read it:"
echo "  E1 OK           -> MUL_MAT works on the NPU path; go wire the backend."
echo "  E1 MISMATCH but E2 OK -> only the plain kernel is broken; use act=5 + silu."
echo "  E3 rmsnorm PASS + gemm untouched -> node contract, not process state."
echo "  E4/E5 rc=0 & untouched  -> task was NOT dispatched (H-E). Stop tuning params,"
echo "                             decompile SubgraphExecImpl::Instantiate's IB."
echo "  E6 single-block writes only -> multi-block/subgrid dispatch bug."
echo "  E7 OK & E1 MISMATCH -> system libxnnl kernel is version-mismatched; use CDN lib."
