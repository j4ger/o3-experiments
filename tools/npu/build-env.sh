#!/usr/bin/env bash
# Shared build environment for the Android targets.
#
# Source this file; do not execute it:
#     source tools/npu/build-env.sh
#
# Overrides:
#     NDK            path to an NDK root (the directory containing
#                    toolchains/llvm/prebuilt/*/bin)
#     LLAMA_CPP_DIR  llama.cpp checkout (default ~/src/llama.cpp, then /tmp/llama.cpp)

if [[ -z "${NDK:-}" ]]; then
    NDK="$(ls -d /nix/store/*android-sdk-ndk-*/libexec/android-sdk/ndk-bundle 2>/dev/null | head -1)"
fi
if [[ -z "${NDK:-}" || ! -d "$NDK" ]]; then
    echo "build-env: Android NDK not found (set NDK=...)" >&2
    return 1 2>/dev/null || exit 1
fi

ANDROID_BIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
CC_A="$ANDROID_BIN/aarch64-linux-android21-clang"
CXX_A="$ANDROID_BIN/aarch64-linux-android21-clang++"

if [[ -z "${LLAMA_CPP_DIR:-}" ]]; then
    if [[ -d "$HOME/src/llama.cpp" ]]; then
        LLAMA_CPP_DIR="$HOME/src/llama.cpp"
    else
        LLAMA_CPP_DIR=/tmp/llama.cpp
    fi
fi

export NDK ANDROID_BIN CC_A CXX_A LLAMA_CPP_DIR
