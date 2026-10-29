#!/usr/bin/env bash
# One device session: push the probe, run the experiments, capture logcat.
#
#     tools/probe/device-run.sh [--sweep]
#
# Output goes to artifacts/probe-logs/<timestamp>-npu/.
# Requires: rooted device, the six SELinux rules from apps/o3root applied,
# /data/local/tmp/o3p_chat.bin present (installed by the o3root app).
set -uo pipefail
cd "$(dirname "$0")/../.."

stamp="$(date +%Y%m%d-%H%M%S)"
out="artifacts/probe-logs/${stamp}-npu"
mkdir -p "$out"

ADB=(nix develop -c adb)
run_root() {  # run_root "<shell command>"
    "${ADB[@]}" shell "/data/local/tmp/o3p_chat.bin --shell '$1'" 2>&1 \
        | grep -avE '^  adb:|^  apktool:|^  jadx:|^Android RE shell'
}

echo "== device =="
"${ADB[@]}" devices -l | tee "$out/devices.txt"

echo "== props =="
"${ADB[@]}" shell getprop > "$out/getprop.txt" 2>&1 || true
grep -aE 'ro\.soc|ro\.board|ro\.hardware|ro\.build' "$out/getprop.txt" | head -20

echo "== /dev =="
"${ADB[@]}" shell 'ls -lZ /dev/hpc-cdev /dev/hpc-heap /dev/hpc-rpmsg 2>&1' > "$out/dev.txt" 2>&1 || true
cat "$out/dev.txt"

if [[ ! -f /tmp/probe_npu_matmul ]]; then
    echo "building probe..." && tools/probe/build.sh /tmp/probe_npu_matmul
fi

echo "== push =="
"${ADB[@]}" push /tmp/probe_npu_matmul /data/local/tmp/probe_npu_matmul >/dev/null
"${ADB[@]}" shell chmod 755 /data/local/tmp/probe_npu_matmul

echo "== xnnl create (logcat) =="
"${ADB[@]}" logcat -b all -c >/dev/null 2>&1 || true
run_root 'cd /data/local/tmp; ./probe_npu_matmul --k 64 --m 8 --t 1 2>&1' | tee "$out/probe-small.txt"
"${ADB[@]}" logcat -b all -d > "$out/logcat.txt" 2>&1 || true
grep -aiE 'discrete ELF|boot success|xnnl error|kernel .*not found|hpc' "$out/logcat.txt" | head -20 \
    | tee "$out/logcat-summary.txt"

echo "== sweep =="
"${ADB[@]}" logcat -b all -c >/dev/null 2>&1 || true
run_root 'cd /data/local/tmp; ./probe_npu_matmul --sweep 2>&1' | tee "$out/probe-sweep.txt"
"${ADB[@]}" logcat -b all -d > "$out/logcat-sweep.txt" 2>&1 || true

echo
echo "logs: $out"
echo "next: inspect $out/probe-sweep.txt and $out/logcat-summary.txt"
