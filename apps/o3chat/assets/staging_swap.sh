#!/system/bin/sh
# staging_swap.sh — keep a shell-controlled file at a download-staging path for
# a bounded window, so that whatever reads that path next (the resource
# installer copies and unzips it) reads our content instead of the genuine
# download.
#
# usage:
#   staging_swap.sh <target> <replacement> <expected_size|0> <duration_ms> <sleep_s>
#
#   target        absolute path the downloader writes, e.g.
#                 .../files/download/taiyi/<type>/<id>/<version>/<fileName>
#   replacement   our package built by tools/staging/mkpack.js
#   expected_size N>0: swap while the path size differs from N (a size-matching
#                     payload keeps the downloader's own size bookkeeping happy);
#                 0  : swap whenever the path content changes (size or mtime)
#   duration_ms   how long to stay armed (must cover download + install copy)
#   sleep_s       poll interval; toybox sleep accepts fractions (e.g. 0.02)
#
# IMPORTANT: the replacement is always COPIED, never hard-linked. A hard link
# would make the target and the replacement share one inode, so a downloader
# that reopens the path would grow our payload too and later swaps would become
# no-ops (observed in tools/staging/swap-sim.js, style "reopen").
#
# Output (greppable from the app):
#   STAGING_ARMED target=.. size=.. dur=..
#   STAGING_SWAP n=.. replaced_size=..
#   STAGING_DONE swaps=.. size=..
set -u

TARGET="${1:-}"
REPL="${2:-}"
SIZE="${3:-0}"
DUR="${4:-3000}"
SLEEP_S="${5:-0.02}"

if [ -z "$TARGET" ] || [ -z "$REPL" ]; then
    echo "STAGING_USAGE"
    exit 2
fi
if [ ! -f "$REPL" ]; then
    echo "STAGING_NO_REPLACEMENT $REPL"
    exit 3
fi

TMP_DIR="$(dirname "$TARGET")"
if [ -d "$TMP_DIR" ]; then
    TMP="$TMP_DIR/.ostage.tmp"
else
    TMP="/data/local/tmp/.ostage.tmp"
fi

MAX=$(( DUR / 20 ))
[ "$MAX" -lt 1 ] && MAX=1

i=0
SWAPS=0
LAST_S=""
LAST_M=""
echo "STAGING_ARMED target=$TARGET size=$SIZE dur=$DUR sleep=$SLEEP_S"

while [ "$i" -lt "$MAX" ]; do
    if [ -e "$TARGET" ]; then
        S="$(stat -c%s "$TARGET" 2>/dev/null || echo 0)"
        M="$(stat -c%Y "$TARGET" 2>/dev/null || echo 0)"
        NEED=0
        if [ "$SIZE" != "0" ]; then
            [ "$S" != "$SIZE" ] && NEED=1
        else
            if [ "$S" != "$LAST_S" ] || [ "$M" != "$LAST_M" ]; then
                NEED=1
            fi
        fi
        if [ "$NEED" = "1" ]; then
            if cp -f "$REPL" "$TMP" 2>/dev/null && mv -f "$TMP" "$TARGET" 2>/dev/null; then
                SWAPS=$((SWAPS + 1))
                LAST_S="$S"
                LAST_M="$M"
                echo "STAGING_SWAP n=$SWAPS replaced_size=$S"
            fi
        fi
    fi
    i=$((i + 1))
    sleep "$SLEEP_S" 2>/dev/null || sleep 1
done

FINAL="$(stat -c%s "$TARGET" 2>/dev/null || echo missing)"
echo "STAGING_DONE swaps=$SWAPS size=$FINAL"
exit 0