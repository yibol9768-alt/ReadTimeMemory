#!/bin/bash
# D3 stage 1: the 18 frozen worlds (docs/D3_SPEC_2026-11-01.md v2), gates, filter per role, pools. Containers removed at exit.
cd ~/data/rtm/sel; O=ws6; mkdir -p $O
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o rtm-env-rr rtm-env-rr2 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1' EXIT
bash wup.sh | tail -1 | grep -q UPOK || { log FAIL up; exit 1; }; log up
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303,Qwen3.8-27B-FP8@18304,Qwen3.8-27B-FP8@18305 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
K='{"p_group": 0.3, "p_long": 0.3, "p_dated": 0.15, "p_greet": 0.1, "p_sib": 0.3}'
one(){ i=$1; for s in plan render verify build gates; do python3 world.py $s --out $O --start $i --n 1 --rounds 240 --par 24 --knobs "$K" $A > $O/$i.$s.log 2>&1 || { log "FAIL $i $s"; return 1; }; done; log "world $i done"; }
P=""; n=0; for i in $(seq 960 977); do one $i & P="$P $!"; n=$((n+1)); if [ $((n%6)) = 0 ]; then for p in $P; do wait $p || exit 1; done; P=""; fi; done; for p in $P; do wait $p || exit 1; done
docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o >/dev/null 2>&1
python3 wfilter.py $O 960,961,962,963 0 ws6f && python3 wfilter.py $O 964,965,966,967,968,969,970,971 0 ws6v && python3 wfilter.py $O 972,973,974,975 0 ws6x && python3 wfilter.py $O 976 0 ws6p0 && python3 wfilter.py $O 977 0 ws6p1 || { log FAIL filter; exit 1; }
docker start rtm-env-emb >/dev/null; bash ~/data/rtm/env/up_rr.sh
for k in $(seq 80); do curl -s -m 3 localhost:18311/v1/models >/dev/null && curl -s -m 3 localhost:18312/v1/models >/dev/null && curl -s -m 3 localhost:18313/v1/models >/dev/null && break; sleep 15; done
for t in ws6f ws6v ws6x ws6p0 ws6p1; do python3 pool.py env_$t.json --rr-port 18312,18313 --T 600 --par 8 --out pool_$t > pool_$t.log 2>&1 || { log "FAIL pool $t"; exit 1; }; log "pool $t"; done
log D3WDONE
