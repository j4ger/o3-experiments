#!/usr/bin/env bash
# Fetch the Shizuku 13.1.5 jars used to build chat/ (same versions as o3root).
set -euo pipefail
VER=13.1.5
DIR="$(cd "$(dirname "$0")" && pwd)"
for a in api provider aidl shared; do
  curl -sS -o "$DIR/$a.aar" "https://repo1.maven.org/maven2/dev/rikka/shizuku/$a/$VER/$a-$VER.aar"
  tmp="$(mktemp -d)"
  unzip -oq "$DIR/$a.aar" classes.jar -d "$tmp"
  cp "$tmp/classes.jar" "$DIR/$a-classes.jar"
  rm -rf "$tmp"
done
sha256sum "$DIR"/*-classes.jar
