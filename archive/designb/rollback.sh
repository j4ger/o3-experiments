#!/usr/bin/env bash
# Rollback Design B injection.  Usage: ./rollback.sh <timestamp-from-run.sh>
set -euo pipefail

TS="${1:?usage: ./rollback.sh <timestamp>}"
PKG="com.xiaomi.aiservice"
API="myllm"
MODEL_ID="myllm_model"

say() { printf '\n===== %s =====\n' "$*"; }
su_sh() { adb shell su -c "$1"; }

say "preflight"
adb get-state >/dev/null
su_sh "id" | grep -q "uid=0" || { echo "root not available"; exit 1; }
su_sh "test -d /data/local/tmp/designb-backup-$TS" || { echo "backup /data/local/tmp/designb-backup-$TS not found"; exit 1; }

DBDIR="$(su_sh "find /data/user_de/0/$PKG /data/data/$PKG -name 'ai_db' 2>/dev/null | head -1" | tr -d '\r')"
DBDIR="$(dirname "$DBDIR")"

say "stop aiservice"
adb shell am force-stop "$PKG" || true
sleep 2

say "restore DB from backup"
DBOWNER="$(su_sh "stat -c '%U:%G' /data/local/tmp/designb-backup-$TS/ai_db")"
DBMODE="$(su_sh "stat -c '%a' /data/local/tmp/designb-backup-$TS/ai_db")"
su_sh "cp -a /data/local/tmp/designb-backup-$TS/ai_db $DBDIR/ai_db && rm -f $DBDIR/ai_db-wal $DBDIR/ai_db-shm && chown $DBOWNER $DBDIR/ai_db && chmod $DBMODE $DBDIR/ai_db && restorecon $DBDIR/ai_db"

say "remove cloned model dir"
MODELDIR="$(su_sh "find /data/user_de/0/$PKG /data/data/$PKG -type d -path '*taiyi/model/$MODEL_ID' 2>/dev/null | head -1" | tr -d '\r')"
[[ -n "$MODELDIR" ]] && su_sh "rm -rf $MODELDIR" || echo "(no clone dir found)"

say "verify removed"
su_sh "cp $DBDIR/ai_db /data/local/tmp/chk && chmod 644 /data/local/tmp/chk"
adb pull /data/local/tmp/chk /tmp/designb-chk.db >/dev/null && adb shell rm -f /data/local/tmp/chk
node -e '
const { DatabaseSync } = require("node:sqlite");
const db = new DatabaseSync("/tmp/designb-chk.db");
const n1 = db.prepare("SELECT COUNT(*) n FROM ai_config WHERE api=?").get("myllm").n;
const n2 = db.prepare("SELECT COUNT(*) n FROM local_resource WHERE id=?").get("myllm_model").n;
db.close();
if (Number(n1) || Number(n2)) { console.error("rows still present!", n1, n2); process.exit(1); }
console.log("PASS: injected rows gone");
'
echo "done. aiservice will rebuild its DB state on next start."
