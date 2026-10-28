#!/usr/bin/env bash
# Builds dev.j4ger.o3root: the privileged tool (CVE payload + KernelSU LKM + ksud
# + OpenVX probe). Kept separate from the chat app so the chat app inherits none
# of it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$ROOT/apps/o3root"
SDK="${ANDROID_BUILD_TOOLS:-$ROOT/poc/sdk/android-16}"
ANDROID_JAR="${ANDROID_JAR:-$ROOT/poc/sdk/platform/android.jar}"
BUILD="$APP/build"
OUT="$APP/o3root.apk"
KEYSTORE="$ROOT/.local/debug.keystore"
SHIZUKU="$ROOT/tools/shizuku"
LIBS=(
  "$APP/jniLibs/arm64-v8a/libo3p.so"
  "$APP/jniLibs/arm64-v8a/libprobex.so"
  "$APP/jniLibs/arm64-v8a/libksud.so"
)

for j in api provider aidl shared; do
  [[ -f "$SHIZUKU/$j-classes.jar" ]] || { echo "missing $SHIZUKU/$j-classes.jar" >&2; exit 1; }
done
for l in "${LIBS[@]}"; do
  if [[ ! -f "$l" ]]; then
    echo "missing $l" >&2
    if [[ "$l" == *libksud.so ]]; then
      echo "  ksud is GPL-3.0 and not redistributed here; get it with:" >&2
      echo "    scripts/fetch-ksud.sh /path/to/KernelSU-manager.apk" >&2
    fi
    exit 1
  fi
done

CP="$SHIZUKU/api-classes.jar:$SHIZUKU/provider-classes.jar:$SHIZUKU/aidl-classes.jar:$SHIZUKU/shared-classes.jar"

rm -rf "$BUILD"
mkdir -p "$BUILD/classes" "$BUILD/dex" "$BUILD/lib/arm64-v8a"
find "$APP/src" -name '*.java' > "$BUILD/sources.txt"
javac -source 8 -target 8 -Xlint:-options \
  -classpath "$ANDROID_JAR:$CP" -d "$BUILD/classes" @"$BUILD/sources.txt"

mapfile -t CLASSES < <(find "$BUILD/classes" -name '*.class')
bash "$SDK/d8" --lib "$ANDROID_JAR" --min-api 34 --output "$BUILD/dex" \
  "${CLASSES[@]}" "$SHIZUKU/api-classes.jar" "$SHIZUKU/provider-classes.jar" \
  "$SHIZUKU/aidl-classes.jar" "$SHIZUKU/shared-classes.jar"

"$SDK/aapt2" link -o "$BUILD/unsigned.apk" -I "$ANDROID_JAR" \
  --manifest "$APP/AndroidManifest.xml" \
  --min-sdk-version 34 --target-sdk-version 37

cp "${LIBS[@]}" "$BUILD/lib/arm64-v8a/"
( cd "$BUILD" && zip -q -j unsigned.apk dex/classes.dex )
( cd "$BUILD" && zip -q unsigned.apk lib/arm64-v8a/libo3p.so lib/arm64-v8a/libprobex.so lib/arm64-v8a/libksud.so )
"$SDK/zipalign" -f 4 "$BUILD/unsigned.apk" "$BUILD/aligned.apk"

mkdir -p "$(dirname "$KEYSTORE")"
if [[ ! -f "$KEYSTORE" ]]; then
  keytool -genkeypair -keystore "$KEYSTORE" -storepass android -keypass android \
    -alias androiddebugkey -dname "CN=Android Debug,O=Android,C=US" \
    -keyalg RSA -keysize 2048 -validity 10000 >/dev/null 2>&1
fi

bash "$SDK/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:android --key-pass pass:android \
  --out "$OUT" "$BUILD/aligned.apk"
ls -lh "$OUT"
