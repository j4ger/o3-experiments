#!/usr/bin/env bash
# Collect device-side NPU evidence before taking the device offline.
#
#   tools/device/collect.sh [output-root]
#
# Default output: artifacts/device/<timestamp>/
# Requires: adb (or ADB=/path/to/adb), root bridge /data/local/tmp/o3p_chat.bin
# for dmesg / sysfs reads. Pull failures are recorded, not fatal.

set -uo pipefail
cd "$(dirname "$0")/../.."

find_adb() {
    if [[ -n "${ADB:-}" ]]; then
        printf '%s\n' "$ADB"
    elif command -v adb >/dev/null 2>&1; then
        command -v adb
    else
        local p
        p=$(ls -d /nix/store/*android-tools-*/bin/adb 2>/dev/null | head -1)
        [[ -n "$p" ]] && printf '%s\n' "$p"
    fi
}

ADB_BIN=$(find_adb || true)
if [[ -z "$ADB_BIN" ]]; then
    echo "collect: adb not found; run inside nix develop or set ADB=" >&2
    exit 2
fi
ADB=("$ADB_BIN")

if ! "${ADB[@]}" devices | grep -qE '^[0-9A-Fa-f]+\s+device'; then
    echo "collect: no device in 'device' state" >&2
    "${ADB[@]}" devices -l >&2
    exit 3
fi

stamp=$(date +%Y%m%d-%H%M%S)
root=${1:-"artifacts/device/$stamp"}
mkdir -p "$root"/{vendor,system,tmp,logs,sys,app}
manifest="$root/manifest.json"
: > "$manifest"
first=1

json_escape() {
    sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e $'s/\t/ /g' -e $'s/\r//g' | tr '\n' ' '
}

record() {
    local path="$1" local_path="$2" size="$3" sha="$4" status="$5" err="${6:-}"
    local esc_path esc_local esc_err
    esc_path=$(printf '%s' "$path" | json_escape)
    esc_local=$(printf '%s' "$local_path" | json_escape)
    esc_err=$(printf '%s' "$err" | json_escape)
    if [[ $first -eq 0 ]]; then printf ',\n' >> "$manifest"; fi
    first=0
    printf '{"path":"%s","local":"%s","size":%s,"sha256":"%s","status":"%s","error":"%s"}' \
        "$esc_path" "$esc_local" "$size" "$sha" "$status" "$esc_err" >> "$manifest"
}

pull_one() {
    local dev="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    local err
    if "${ADB[@]}" pull "$dev" "$dst" >/tmp/o3-collect-pull.err 2>&1; then
        local size sha
        size=$(stat -c '%s' "$dst" 2>/dev/null || echo 0)
        sha=$(sha256sum "$dst" 2>/dev/null | awk '{print $1}')
        record "$dev" "$dst" "$size" "$sha" "ok" ""
    else
        err=$(cat /tmp/o3-collect-pull.err 2>/dev/null)
        record "$dev" "$dst" 0 "" "fail" "$err"
    fi
}

pull_glob() {
    local pattern="$1" dst_dir="$2"
    local names name
    names=$("${ADB[@]}" shell "ls -1 $pattern 2>/dev/null" 2>/dev/null || true)
    [[ -n "$names" ]] || return 0
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        pull_one "$name" "$dst_dir/$(basename "$name")"
    done <<< "$names"
}

capture_shell() {
    local label="$1" cmd="$2"
    "${ADB[@]}" shell "$cmd" > "$root/logs/$label.txt" 2>&1 || true
}

echo "== device =="
"${ADB[@]}" devices -l | tee "$root/logs/devices.txt"

echo "== props / runtime snapshots =="
"${ADB[@]}" shell getprop > "$root/logs/getprop.txt" 2>&1 || true
capture_shell "uname" 'uname -a'
capture_shell "dev-hpc" 'ls -lZ /dev/hpc-* 2>&1'
capture_shell "enforce" 'getenforce 2>&1'
capture_shell "interrupts" 'cat /proc/interrupts 2>&1'
capture_shell "iomem" 'cat /proc/iomem 2>&1'
capture_shell "modules" 'ls -d /sys/module/hpc_* /sys/module/npu_* /sys/module/os_cpu_vip_sched 2>&1'
capture_shell "logcat-all" 'logcat -b all -d 2>&1'
capture_shell "logcat-kernel" 'logcat -b kernel -d 2>&1'
if "${ADB[@]}" shell "test -x /data/local/tmp/o3p_chat.bin" >/dev/null 2>&1; then
    "${ADB[@]}" shell "/data/local/tmp/o3p_chat.bin --shell 'dmesg 2>&1'" > "$root/logs/dmesg.txt" 2>&1 || true
fi

echo "== vendor NPU libs =="
for f in \
    libNPURuntime.so libNPURuntime_wrapper.so libNNVXCBinary.so \
    libOpenVXU.so libOvx12VXCBinary.so libOvxGPUVXCBinary.so \
    libnpu_ainr.so libxnnl.so libxnumd.so libxnlm.so \
    libxnn.so libxnn_llm.so libxnn_runtime.so libxnn_cpu.so libxnn_gpu.so \
    libxnn_mapping.so libxnn_op.so libxnnapi.so libNBGlinker.so \
    libOpenVX.so libovxlib.so libVSC.so libOpenCL.so; do
    pull_one "/vendor/lib64/$f" "$root/vendor/$f"
done

echo "== system NPU libs =="
for f in libnpumanager.so libvip_channel.so; do
    pull_one "/system/lib64/$f" "$root/system/$f"
done

echo "== vendor npu etc =="
pull_one "/vendor/etc/npu" "$root/vendor/etc-npu"

echo "== /data/local/tmp evidence =="
for f in collect.tgz libxnnl_model.so; do
    pull_one "/data/local/tmp/$f" "$root/tmp/$f"
done
pull_glob '/data/local/tmp/*.dot' "$root/tmp"
pull_glob '/data/local/tmp/*.log' "$root/tmp"
pull_one "/data/local/tmp/collect" "$root/tmp/collect"

echo "== protected app data (best effort) =="
if "${ADB[@]}" shell "test -x /data/local/tmp/o3p_chat.bin" >/dev/null 2>&1; then
    "${ADB[@]}" shell "/data/local/tmp/o3p_chat.bin --shell 'find /data/user_de/0/com.xiaomi.aiservice/files/download/taiyi -maxdepth 3 -type f 2>&1'" \
        > "$root/app/aiservice-files.txt" 2>&1 || true
    "${ADB[@]}" shell "/data/local/tmp/o3p_chat.bin --shell 'ls -la /data/user_de/0/com.xiaomi.aiservice/files/download/taiyi 2>&1'" \
        > "$root/app/aiservice-ls.txt" 2>&1 || true
fi

printf '\n]\n' >> "$manifest"
echo
echo "collected to $root"
echo "manifest: $manifest"
