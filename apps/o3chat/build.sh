#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHAT="$ROOT/apps/o3chat"
SDK="${ANDROID_BUILD_TOOLS:-$ROOT/poc/sdk/android-16}"
ANDROID_JAR="${ANDROID_JAR:-$ROOT/poc/sdk/platform/android.jar}"
BUILD="$CHAT/build"
OUT="$CHAT/o3chat.apk"
KEYSTORE="$ROOT/.local/debug.keystore"
SHIZUKU="$ROOT/tools/shizuku"

if [[ ! -x "$SDK/aapt2" || ! -f "$ANDROID_JAR" ]]; then
  echo "missing build tools or android.jar" >&2
  exit 1
fi
for j in api provider aidl shared; do
  [[ -f "$SHIZUKU/$j-classes.jar" ]] || { echo "missing $SHIZUKU/$j-classes.jar (run scripts later: see tools/shizuku)" >&2; exit 1; }
done

SHIZUKU_CP="$SHIZUKU/api-classes.jar:$SHIZUKU/provider-classes.jar:$SHIZUKU/aidl-classes.jar:$SHIZUKU/shared-classes.jar"

rm -rf "$BUILD"
mkdir -p "$BUILD/classes" "$BUILD/dex"

find "$CHAT/src" -name '*.java' > "$BUILD/sources.txt"
javac -source 8 -target 8 -Xlint:-options \
  -classpath "$ANDROID_JAR:$SHIZUKU_CP" \
  -d "$BUILD/classes" \
  @"$BUILD/sources.txt"

mapfile -t CLASSES < <(find "$BUILD/classes" -name '*.class')
bash "$SDK/d8" --lib "$ANDROID_JAR" --min-api 34 \
  --output "$BUILD/dex" "${CLASSES[@]}" \
  "$SHIZUKU/api-classes.jar" "$SHIZUKU/provider-classes.jar" \
  "$SHIZUKU/aidl-classes.jar" "$SHIZUKU/shared-classes.jar"

"$SDK/aapt2" link \
  -o "$BUILD/unsigned.apk" \
  -I "$ANDROID_JAR" \
  -A "$CHAT/assets" \
  --manifest "$CHAT/AndroidManifest.xml" \
  --min-sdk-version 34 \
  --target-sdk-version 37

mkdir -p "$BUILD/lib/arm64-v8a"
# dex must land at the APK root; the payload must keep its lib/<abi>/ path.
( cd "$BUILD" && zip -q -j unsigned.apk dex/classes.dex )

"$SDK/zipalign" -f 4 "$BUILD/unsigned.apk" "$BUILD/aligned.apk"

mkdir -p "$(dirname "$KEYSTORE")"
if [[ ! -f "$KEYSTORE" ]]; then
  keytool -genkeypair \
    -keystore "$KEYSTORE" \
    -storepass android -keypass android \
    -alias androiddebugkey \
    -dname "CN=Android Debug,O=Android,C=US" \
    -keyalg RSA -keysize 2048 -validity 10000 >/dev/null 2>&1
fi

bash "$SDK/apksigner" sign \
  --ks "$KEYSTORE" \
  --ks-pass pass:android \
  --key-pass pass:android \
  --out "$OUT" \
  "$BUILD/aligned.apk"

bash "$SDK/apksigner" verify --print-certs "$OUT" | head -5
ls -lh "$OUT"
