#!/usr/bin/env bash
# Build probe_npu_matmul for arm64 Android.
#
#     tools/probe/build.sh [output-path]
#
# Default output: /tmp/probe_npu_matmul
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/npu/build-env.sh

out="${1:-/tmp/probe_npu_matmul}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

"$CC_A"  -O2 -std=c11   -Wall -Wextra -I tools/npu -c tools/npu/o3npu.c      -o "$tmp/o3npu.o"
"$CXX_A" -O2 -std=c++17 -Wall -Wextra -I tools/npu -c tools/npu/xnnl_shim.cpp -o "$tmp/xnnl_shim.o"
"$CXX_A" -O2 -std=c++17 -Wall -Wextra -I tools/npu \
    tools/probe/probe_npu_matmul.cpp "$tmp/o3npu.o" "$tmp/xnnl_shim.o" \
    -ldl -o "$out"

echo "built $out"
