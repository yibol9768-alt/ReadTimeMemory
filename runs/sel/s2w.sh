#!/bin/bash
# S2 stage 1 (11-01, after 08:30 ET, DeepSeek stopped by hand): redraw questions for all 21 worlds, gates, pools.
cd ~/data/rtm/sel
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303,Qwen3.8-27B-FP8@18304,Qwen3.8-27B-FP8@18305 --gem x@1 --oss x@1'
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; M=/home/centuser/data/models
S="--max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 256"
for g in 0 1 2 3 4; do docker rm -f rtm-env-q$g >/dev/null 2>&1; docker run -d --name rtm-env-q$g --gpus "\"device=$g\"" --ipc=host --network host -v $M/Qwen/Qwen3.8-27B-FP8:/model:ro $IMG /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port $((18301+g)) --enable-prefix-caching $S >/dev/null; done
for k in $(seq 80); do ok=1; for p in 18301 18302 18303 18304 18305; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done; [ $ok = 1 ] || { log FAIL up; exit 1; }
mkdir -p ws4; cp ws1/w000*.json ws4/
W="10 11 12 13 14 $(seq 15 30)"
one(){ i=$1; for s in build gates; do python3 world.py $s --out ws4 --start $i --n 1 --rounds 240 --par 24 $A > ws4_$i.$s.log 2>&1 || { log "FAIL world $i $s"; return 1; }; done; log "world $i done"; }
P=""; n=0; for i in $W; do one $i & P="$P $!"; n=$((n+1)); if [ $((n%7)) = 0 ]; then for p in $P; do wait $p || exit 1; done; P=""; fi; done; for p in $P; do wait $p || exit 1; done
docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 >/dev/null 2>&1
python3 wfilter.py ws4 $(echo $W | tr ' ' ',') 14 ws4 || { log FAIL filter; exit 1; }
bash ~/data/rtm/env/up_rr.sh
for k in $(seq 80); do curl -s -m 3 localhost:18312/v1/models >/dev/null && curl -s -m 3 localhost:18313/v1/models >/dev/null && break; sleep 15; done
python3 pool.py env_ws4.json --rr-port 18312,18313 --T 600 --par 8 --out pool_ws4 || { log FAIL pool; exit 1; }
docker rm -f rtm-env-rr rtm-env-rr2 >/dev/null 2>&1; log S2WDONE
