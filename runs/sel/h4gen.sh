#!/bin/bash
# D4 SPEC v2.1 §2/§8/§12: new held-out worlds (H4) in ws8, generator and knobs of D2 (p_group 0.4), plan with --exclude-from
# (no project shared with ANY earlier world or earlier H4 world; names only from NEW_NAMES). Plans run one by one so each
# new world joins the exclusion list; render/verify/build/gates run 6 worlds at a time. Resumable: a step is skipped when
# ws8/<id>.<step>.ok exists. Watchdog stops everything at 17:40 ET (DS starts 18:00). Containers removed at exit.
# usage: setsid nohup bash h4gen.sh "1999" > h4gen_tiny.log 2>&1 &      (e2e-tiny world)
#        setsid nohup bash h4gen.sh "$(seq -s' ' 2000 2011)" > h4gen.log 2>&1 &
set -u; cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
exec 9>/tmp/rtm_h4gen.lock; flock -n 9 || { log "already running"; exit 2; }
IDS=$1; O=ws8; mkdir -p $O
END=$(TZ=America/New_York date -d "today 17:40" +%s); [ $(( END - $(date +%s) )) -gt 600 ] || { log "too close to 17:40"; exit 2; }
PG=$(ps -o pgid= $$ | tr -d ' ')
down(){ docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o >/dev/null 2>&1; }
( exec 9>&-; sleep $(( END - $(date +%s) )); log WATCHDOG; down; kill -TERM -- -$PG ) & W=$!
trap 'kill $W 2>/dev/null; down' EXIT
EXCL=ws1,ws3,ws3t,ws4,ws4_bak,ws5p,ws6,ws6e,ws7,ws7e,ws7x,$O
K='{"p_group": 0.4, "p_long": 0.3, "p_dated": 0.15, "p_greet": 0.1, "p_sib": 0.3}'
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303,Qwen3.8-27B-FP8@18304,Qwen3.8-27B-FP8@18305 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
bash wup.sh | tail -1 | grep -q UPOK || { log "FAIL servers"; exit 1; }; log "servers up"
st(){ i=$1; shift; for s in "$@"; do
    [ -f $O/$i.$s.ok ] && continue
    python3 world.py $s --out $O --start $i --n 1 --rounds 240 --par 24 --knobs "$K" --exclude-from $EXCL $A > $O/$i.$s.log 2>&1 || { log "FAIL $i $s"; return 1; }
    touch $O/$i.$s.ok; done; log "world $i: $* done"; }
for i in $IDS; do st $i plan || exit 1; done; log "plans done"
F=0; P=""; n=0; for i in $IDS; do st $i render verify build gates & P="$P $!"; n=$((n+1)); if [ $((n % 6)) = 0 ]; then for p in $P; do wait $p || F=$((F+1)); done; P=""; fi; done
for p in $P; do wait $p || F=$((F+1)); done   # one failed world must not stop the others (11-02: a failure used to take the servers down mid-run)
[ $F = 0 ] && log "H4GENDONE $IDS" || { log "H4GENDONE with $F failed worlds (rerun to resume)"; exit 1; }
