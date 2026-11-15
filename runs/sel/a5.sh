#!/bin/bash
# Batch three: reader servers on all 8 GPUs, answer table, adaptive packing. A watchdog removes the readers at 17:40 EST whatever happens, so the GPUs are free before DeepSeek at 18:00.
cd ~/data/rtm/sel; IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
END=$(TZ=America/New_York date -d "today 17:40" +%s); left(){ echo $(( END - $(date +%s) )); }; [ $(left) -gt 300 ] || { log "too late"; exit 1; }
stop(){ for g in 0 1 2 3 4 5 6 7; do docker rm -f rtm-qa-$g >/dev/null 2>&1; done; }
( sleep $(left); pkill -P $$ -f "a5.py" ; for g in 0 1 2 3 4 5 6 7; do docker rm -f rtm-qa-$g >/dev/null 2>&1; done ) & W=$!
trap 'kill $W 2>/dev/null; stop' EXIT
P=""; for g in 0 1 2 3 4 5 6 7; do p=$((18300+g)); P="$P,$p"; docker run -d --name rtm-qa-$g --gpus "\"device=$g\"" --network host --ipc host -v /home/centuser/data/models/Qwen/Qwen3.8-27B-FP8:/model:ro $IMG /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port $p --enable-prefix-caching --max-model-len 131072 --gpu-memory-utilization 0.9 --max-num-seqs 128 >/dev/null || exit 1; done; P=${P#,}
for k in $(seq 1 90); do ok=1; for p in ${P//,/ }; do curl -s -m 3 localhost:$p/v1/models | grep -q Qwen || ok=0; done; [ $ok = 1 ] && break; sleep 10; done; [ $ok = 1 ] || { log NOTREADY; exit 1; }; log ready
python3 a5.py --adaptive --ports $P --workers 56 > a5b.log 2>&1; log "adaptive code $?"
[ $(left) -gt 60 ] && timeout -k 20 $(left) python3 a5.py --ports $P --workers 56 > a5.log 2>&1; log "A5SHDONE table code $?"
