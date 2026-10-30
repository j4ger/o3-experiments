#!/usr/bin/env bash
# Build and run the offline selftest: our reference matmul / copy against ggml.
#
#   tools/ggml-o3npu/selftest.sh [out_binary]
#
# Requires a host llama.cpp build (see tools/ggml-o3npu/build.sh host).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../npu/build-env.sh
source "$here/../npu/build-env.sh"

so=${GGML_O3NPU_SO:-/tmp/libggml-o3npu-host.so}
if [ ! -f "$so" ]; then
    bash "$here/build.sh" host "$so"
fi

out=${1:-/tmp/o3npu-selftest}
nix shell nixpkgs#gcc -c g++ -std=c++17 -O2 \
    -I"$LLAMA_CPP_DIR/include" -I"$LLAMA_CPP_DIR/ggml/include" \
    "$here/selftest.cpp" "$so" \
    -L"$LLAMA_CPP_DIR/build-host/bin" \
    -lggml-cpu -lggml -lggml-base -o "$out"

LD_LIBRARY_PATH="$LLAMA_CPP_DIR/build-host/bin:$so:${LD_LIBRARY_PATH:-}" "$out"
