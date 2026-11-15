#!/bin/bash
# S2 preparation (10-31 night): 16 more worlds, pools for worlds and the two new benchmarks. Must finish before 24:00 ET; ds_guard restarts DeepSeek after that.
cd ~/data/rtm/sel
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303,Qwen3.8-27B-FP8@18304,Qwen3.8-27B-FP8@18305 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
python3 bsub.py || { log FAIL subset; exit 1; }
sed -i 's/^for g in 0 1 2; do/for g in 0 1 2 3 4; do/; s/for p in 18301 18302 18303 18321/for p in 18301 18302 18303 18304 18305 18321/' wup.sh
bash wup.sh || { log FAIL up; exit 1; }
one(){ i=$1; for s in plan render verify build gates; do python3 world.py $s --out ws1 --start $i --n 1 --rounds 240 --par 24 $A > ws1_$i.$s.log 2>&1 || { log "FAIL world $i $s"; return 1; }; done; log "world $i done"; }
for batch in "15 16 17 18 19 20 21 22" "23 24 25 26 27 28 29 30"; do P=""; for i in $batch; do one $i & P="$P $!"; done; for p in $P; do wait $p; done; done
log "worlds done"
docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o >/dev/null 2>&1
W=$(for i in 10 11 12 13 14 $(seq 15 30); do [ -f ws1/g000$i.json ] && echo -n "$i,"; done); python3 wfilter.py ws1 ${W%,} 14 ws2 || { log FAIL filter; exit 1; }
bash ~/data/rtm/env/up_rr.sh
for k in $(seq 80); do curl -s -m 3 localhost:18312/v1/models >/dev/null && curl -s -m 3 localhost:18313/v1/models >/dev/null && break; sleep 15; done
python3 pool.py env_ws2.json --rr-port 18312,18313 --T 600 --par 8 --out pool_ws2 || { log FAIL pool ws2; exit 1; }; log "pool ws2 done"
python3 pool.py q_ep150.json --rr-port 18312,18313 --T 600 --par 8 --out pool_ep150 || { log FAIL pool ep; exit 1; }; log "pool ep done"
python3 pool.py q_ev150.json --rr-port 18312,18313 --T 600 --par 8 --out pool_ev150 || { log FAIL pool ev; exit 1; }; log "pool ev done"
docker rm -f rtm-env-rr rtm-env-rr2 >/dev/null 2>&1
log S2ADONE
