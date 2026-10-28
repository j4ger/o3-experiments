#!/usr/bin/env bash
# Grant the shell domain access to /dev/hpc-cdev through KernelSU's own policy
# patcher, keeping SELinux in Enforcing. Requires the temporary root from
# docs/ROOT-CHAIN.md (rootd on @o3p_rootd) to be running.
#
#   tools/root/npu_enable.sh [path/to/ksu-manager.apk]
#
# The rule is exactly the one every NPU client domain holds in the device policy
# (hal_npu, priv_app_36, platform_app, minetd, ...). There is no allowxperm on
# npu_device, so no ioctl whitelist is needed.
set -euo pipefail

RULE='allow shell npu_device:chr_file { append getattr ioctl lock map open read watch watch_reads write };'
MGR="${1:-/tmp/o3root/app/assets/ksu-manager.apk}"
BIN=/data/local/tmp/o3p_chat.bin
KSUD=/data/local/tmp/ksud

if [[ ! -f "$MGR" ]]; then
  echo "ksu-manager.apk not found: $MGR" >&2
  echo "usage: $0 [ksu-manager.apk]   (from github.com/zsw125/o3root app/assets)" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
( cd "$tmp" && jar xf "$MGR" lib/arm64-v8a/libksud.so )
adb push "$tmp/lib/arm64-v8a/libksud.so" "$KSUD" >/dev/null
adb shell "chmod 755 $KSUD"

echo "== check =="
adb shell "$BIN --shell \"$KSUD sepolicy check '$RULE'\""
echo "== patch =="
adb shell "$BIN --shell \"$KSUD sepolicy patch '$RULE'\""
echo "== verify =="
adb shell "$BIN --shell \"(exec 3<>/dev/hpc-cdev) 2>&1 && echo open=OK || echo open=DENIED; getenforce\""
