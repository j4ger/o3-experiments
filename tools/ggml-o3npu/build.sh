#!/usr/bin/env bash
# Build the o3npu ggml backend.
#
#     tools/ggml-o3npu/build.sh android [out.so]   # arm64, for the device
#     tools/ggml-o3npu/build.sh host    [out.so]   # x86_64, for selftest/PPL
#
# Android default output: /tmp/libggml-o3npu.so
# Host    default output: /tmp/libggml-o3npu-host.so
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/npu/build-env.sh

mode="${1:-android}"
out="${2:-}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

inc="-I $LLAMA_CPP_DIR/ggml/include -I $LLAMA_CPP_DIR/ggml/src -I $LLAMA_CPP_DIR/include -I tools/npu"

if [[ "$mode" == "android" ]]; then
    : "${out:=/tmp/libggml-o3npu.so}"
    "$CC_A"  -O2 -std=c11   -Wall -Wextra $inc -fPIC -c tools/npu/o3npu.c      -o "$tmp/o3npu.o"
    "$CXX_A" -O2 -std=c++17 -Wall -Wextra $inc -fPIC -c tools/npu/xnnl_shim.cpp -o "$tmp/xnnl_shim.o"
    "$CXX_A" -shared -fPIC -O2 -std=c++17 -DGGML_BACKEND_DL $inc \
        tools/ggml-o3npu/ggml-o3npu.cpp "$tmp/o3npu.o" "$tmp/xnnl_shim.o" \
        -L "$LLAMA_CPP_DIR/build-dl/bin" -lggml-base -ldl -o "$out"
elif [[ "$mode" == "host" ]]; then
    : "${out:=/tmp/libggml-o3npu-host.so}"
    gcc  -O2 -std=c11   -Wall -Wextra $inc -fPIC -c tools/npu/o3npu.c      -o "$tmp/o3npu.o"
    g++  -O2 -std=c++17 -Wall -Wextra $inc -fPIC -c tools/npu/xnnl_shim.cpp -o "$tmp/xnnl_shim.o"
    g++  -shared -fPIC -O2 -std=c++17 -DGGML_BACKEND_DL $inc \
        tools/ggml-o3npu/ggml-o3npu.cpp "$tmp/o3npu.o" "$tmp/xnnl_shim.o" \
        -L "$LLAMA_CPP_DIR/build-host/bin" -lggml-base -ldl -o "$out"
else
    echo "usage: $0 android|host [out.so]" >&2
    exit 2
fi

echo "built $out"
