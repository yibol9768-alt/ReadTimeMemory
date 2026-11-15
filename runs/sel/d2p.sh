#!/bin/bash
# D2 tiny pilot: 2 worlds x 80 rounds, all knobs on (docs/D2_SPEC_2026-11-01.md). Containers removed at exit.
cd ~/data/rtm/sel; O=ws5p; mkdir -p $O
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o >/dev/null 2>&1' EXIT
bash wup.sh | tail -1 | grep -q UPOK || { log FAIL up; exit 1; }; log up
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303,Qwen3.8-27B-FP8@18304,Qwen3.8-27B-FP8@18305 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
K='{"p_group": 0.4, "p_long": 0.3, "p_dated": 0.15, "p_greet": 0.1, "p_sib": 0.3}'
one(){ i=$1; for s in plan render verify build gates; do python3 world.py $s --out $O --start $i --n 1 --rounds 80 --par 24 --knobs "$K" $A > $O/$i.$s.log 2>&1 || { log "FAIL $i $s"; return 1; }; done; log "world $i done"; }
one 950 & P1=$!; one 951 & P2=$!; wait $P1 || exit 1; wait $P2 || exit 1
log D2PDONE
