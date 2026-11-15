#!/bin/bash
# Step 3 launcher: one Qwen3.8-27B-FP8 reader per listed GPU (port 18300+g), then a6.py with the given arguments.
# usage: setsid nohup bash a6.sh "0,2,5" --skip RR8 --k 2 > a6_smoke.log 2>&1 &
# Deadline 18:30 EST: the reader containers are removed and the process group is stopped.
set -u
cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
GPUS=$1; shift
exec 9>/tmp/rtm_a6.lock; flock -n 9 || { log "a6.sh already running"; exit 2; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; PFX=rtm-a6-
END=$(TZ=America/New_York date -d "today 18:30" +%s); [ $(( END - $(date +%s) )) -gt 900 ] || { log "too close to deadline"; exit 2; }
PG=$(ps -o pgid= $$ | tr -d ' ')
cleanup(){ timeout 120 bash -c "docker ps -a --format '{{.Names}}' | grep '^$PFX' | xargs -r docker rm -f" >/dev/null 2>&1; }
( exec 9>&-; sleep $(( END - $(date +%s) )); log "WATCHDOG"; cleanup; kill -TERM -- -$PG ) & W=$!
trap 'S=$(pgrep -P $W); kill $W 2>/dev/null; [ -n "$S" ] && kill $S 2>/dev/null; cleanup' EXIT
P=""
for g in ${GPUS//,/ }; do p=$((18300+g))
  ! ss -ltn | grep -q ":$p " || { log "port $p in use"; exit 1; }
  docker run -d --name $PFX$g --gpus "\"device=$g\"" --network host --ipc host -v /home/centuser/data/models/Qwen/Qwen3.8-27B-FP8:/model:ro $IMG /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port $p --enable-prefix-caching --max-model-len 131072 --gpu-memory-utilization 0.9 --max-num-seqs 128 >/dev/null || { log "START FAIL $g"; exit 1; }
  P="$P,$p"; done; P=${P#,}
for k in $(seq 1 90); do ok=1; for p in ${P//,/ }; do curl -sf -m 3 localhost:$p/v1/models | grep -q Qwen || ok=0; done; [ $ok = 1 ] && break; sleep 10; done
[ $ok = 1 ] || { log NOTREADY; exit 1; }; log "ready $P"
python3 a6.py --ports $P "$@"; c=$?; log "A6SHDONE code $c"; exit $c
