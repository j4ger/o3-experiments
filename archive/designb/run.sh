#!/usr/bin/env bash
# Design B on-device runbook (NOTES.md §20.2 / §23).
# Host-side orchestrator: backup -> clone model dir -> inject DB rows -> test.
#
# Prerequisites:
#   * device connected, adb in PATH (nix develop)
#   * root available from adb shell:  adb shell su -c id  ->  uid=0
#     (KSU/SukiSU: grant the shell UID in the manager app first)
#   * poc/taiyi-authorize-poc.apk built (supports --es api/--es name)
#
# Usage:  ./run.sh           # full inject + test
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$ROOT/tools/designb/work"
TS="$(date +%Y%m%d-%H%M%S)"
PKG="com.xiaomi.aiservice"
SRC_API="vlm"
SRC_NAME="vlm_qwen3_0_7b_f3"
SRC_MODEL_ID="vlm_model_qwen3_0_7b_f3"
API="myllm"
NAME="myllm_f3"
MODEL_ID="myllm_model"
MODEL_VER="1"
APK="$ROOT/poc/taiyi-authorize-poc.apk"

mkdir -p "$WORK"

say() { printf '\n===== %s =====\n' "$*"; }
su_sh() { adb shell su -c "$1"; }

say "preflight"
adb get-state >/dev/null || { echo "no device"; exit 1; }
su_sh "id" | grep -q "uid=0" || { echo "root not available (su -c id != uid 0)"; exit 1; }
[[ -f "$APK" ]] || { echo "missing $APK (run poc/build.sh)"; exit 1; }
su_sh "pm path $PKG" >/dev/null || { echo "$PKG not installed"; exit 1; }

say "locate ai_db"
DBDIR="$(su_sh "find /data/user_de/0/$PKG /data/data/$PKG -name 'ai_db' 2>/dev/null | head -1" | tr -d '\r')"
[[ -n "$DBDIR" ]] || { echo "ai_db not found"; exit 1; }
DBDIR="$(dirname "$DBDIR")"
echo "db dir: $DBDIR"
su_sh "ls -la $DBDIR" | sed 's/^/  /'

say "dump live rows for the record"
su_sh "cp $DBDIR/ai_db /data/local/tmp/ai_db.readonly && chmod 644 /data/local/tmp/ai_db.readonly"
adb pull /data/local/tmp/ai_db.readonly "$WORK/ai_db.pristine-$TS" >/dev/null
adb shell rm -f /data/local/tmp/ai_db.readonly
node - "$WORK/ai_db.pristine-$TS" <<'EOF'
const { DatabaseSync } = require("node:sqlite");
const db = new DatabaseSync(process.argv[2]);
for (const r of db.prepare("SELECT api,name,version,disabled,origin FROM ai_config WHERE api='vlm' OR name LIKE 'vlm%'").all())
  console.log("  ai_config:", JSON.stringify(r));
for (const r of db.prepare("SELECT id,version,type,scheme,unwrapped,subType FROM local_resource WHERE id LIKE 'vlm%'").all())
  console.log("  local_resource:", JSON.stringify(r));
db.close();
EOF

say "backup DB + stat ownership"
su_sh "mkdir -p /data/local/tmp/designb-backup-$TS && cp -a $DBDIR/. /data/local/tmp/designb-backup-$TS/"
DBOWNER="$(su_sh "stat -c '%U:%G' $DBDIR/ai_db")"
DBMODE="$(su_sh "stat -c '%a' $DBDIR/ai_db")"
DBCONTEXT="$(su_sh "stat -c '%C' $DBDIR/ai_db")"
echo "owner=$DBOWNER mode=$DBMODE context=$DBCONTEXT"

say "stop aiservice"
adb shell am force-stop "$PKG" || true
sleep 2
su_sh "ps -A | grep $PKG" || echo "  (no aiservice processes left)"

say "clone model dir -> $MODEL_ID/$MODEL_VER"
TAIYI_MODEL="$(su_sh "find /data/user_de/0/$PKG /data/data/$PKG -type d -path '*taiyi/model/$SRC_MODEL_ID' 2>/dev/null | head -1" | tr -d '\r')"
[[ -n "$TAIYI_MODEL" ]] || { echo "model dir not found"; exit 1; }
SRC_VERDIR="$(su_sh "ls $TAIYI_MODEL | head -1" | tr -d '\r')"
echo "source: $TAIYI_MODEL/$SRC_VERDIR"
MODEL_ROOT="$(dirname "$TAIYI_MODEL")"
su_sh "rm -rf $MODEL_ROOT/$MODEL_ID && mkdir -p $MODEL_ROOT/$MODEL_ID/$MODEL_VER && cp -a $TAIYI_MODEL/$SRC_VERDIR/. $MODEL_ROOT/$MODEL_ID/$MODEL_VER/"
MODEL_DIR="$MODEL_ROOT/$MODEL_ID/$MODEL_VER"
su_sh "test -f $MODEL_DIR/base.json" || { echo "clone incomplete (no base.json)"; exit 1; }
MOWNER="$(su_sh "stat -c '%U:%G' $TAIYI_MODEL/$SRC_VERDIR")"
su_sh "chown -R $MOWNER $MODEL_ROOT/$MODEL_ID && restorecon -R $MODEL_ROOT/$MODEL_ID"
echo "clone at $MODEL_DIR (owner $MOWNER)"

say "pull DB files and inject"
for f in ai_db ai_db-wal ai_db-shm; do
  su_sh "test -f $DBDIR/$f && cp $DBDIR/$f /data/local/tmp/pull_$f && chmod 644 /data/local/tmp/pull_$f" || true
  if adb shell test -f /data/local/tmp/pull_$f; then
    adb pull "/data/local/tmp/pull_$f" "$WORK/$f" >/dev/null
    adb shell rm -f /data/local/tmp/pull_$f
  fi
done
node "$ROOT/tools/designb/inject.js" --db "$WORK/ai_db" \
  --src-api "$SRC_API" --src-name "$SRC_NAME" \
  --api "$API" --name "$NAME" \
  --model-id "$MODEL_ID" --model-version "$MODEL_VER" \
  --model-dir "$MODEL_DIR"

say "push DB back and fix ownership"
adb push "$WORK/ai_db" /data/local/tmp/push_ai_db >/dev/null
su_sh "cp /data/local/tmp/push_ai_db $DBDIR/ai_db && rm -f $DBDIR/ai_db-wal $DBDIR/ai_db-shm /data/local/tmp/push_ai_db && chown $DBOWNER $DBDIR/ai_db && chmod $DBMODE $DBDIR/ai_db && restorecon $DBDIR/ai_db"
echo "pushed; owner $DBOWNER mode $DBMODE context $(su_sh "stat -c '%C' $DBDIR/ai_db")"

say "install PoC and run with api=$API name=$NAME"
adb install -r "$APK" >/dev/null
adb logcat -c
adb shell am start -n com.example.taiyi_authorize_poc/.MainActivity --es api "$API" --es name "$NAME" >/dev/null
sleep 25
adb logcat -d -v time > "$WORK/designb-run-$TS.log"

say "evidence grep"
grep -E "TaiyiPoc|AIConfigurations.*myllm|getAIConfig myllm|BaseVlmPlugin|Stub0|LLMRuntime|CALLBACK|RESP_BODY ans" \
  "$WORK/designb-run-$TS.log" | tail -40 || echo "(no matches — check $WORK/designb-run-$TS.log)"

say "result"
echo "log:    $WORK/designb-run-$TS.log"
echo "backup: /data/local/tmp/designb-backup-$TS  (device)"
echo "pristine db copy: $WORK/ai_db.pristine-$TS"
echo "rollback: ./rollback.sh $TS"
