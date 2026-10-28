#!/usr/bin/env bash
# Extract ksud (KernelSU's userspace CLI, GPL-3.0) from the KernelSU manager APK.
# We do not redistribute it: get the APK from your KernelSU release and run:
#   scripts/fetch-ksud.sh /path/to/KernelSU_*.apk
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/apps/o3root/jniLibs/arm64-v8a/libksud.so"
APK="${1:-}"

if [[ -z "$APK" || ! -f "$APK" ]]; then
  echo "usage: $0 /path/to/KernelSU-manager.apk" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
( cd "$tmp" && unzip -qo "$APK" "lib/arm64-v8a/libksud.so" )
mkdir -p "$(dirname "$DEST")"
cp "$tmp/lib/arm64-v8a/libksud.so" "$DEST"
chmod 644 "$DEST"
echo "installed $DEST ($(stat -c%s "$DEST") bytes)"
