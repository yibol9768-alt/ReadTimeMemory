#!/bin/bash
# independent judging of generated answers. usage: bash judgeans.sh "GPUS for gpt-oss" "GPU for gemma or -" SRC
set -u; cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
GO=$1; GE=$2; SRC=$3; IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; M=/home/centuser/data/models; PFX=rtm-jans2-
trap 'docker ps -a --format "{{.Names}}" | grep "^$PFX" | xargs -r docker rm -f >/dev/null 2>&1' EXIT
start(){ local g=$1 port=$2 dir=$3 name=$4; ss -ltn | grep -q ":$port " && { log "port $port busy"; return 1; }
  docker run -d --name $PFX$g --gpus "\"device=$g\"" --network host --ipc host -v $dir:/model:ro $IMG /model --served-model-name $name --host 127.0.0.1 --port $port --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 128 >/dev/null; }
wait_up(){ for k in $(seq 120); do curl -sf -m3 localhost:$1/v1/models | grep -q "$2" && return 0; sleep 10; done; return 1; }
P=""; for g in ${GO//,/ }; do p=$((18350+g)); start $g $p $M/openai-mirror/gpt-oss-120b gpt-oss-120b || exit 1; P="$P,$p"; done; P=${P#,}
GP=""; for g in ${GE//,/ }; do p=$((18370+g)); start $g $p $M/google/gemma-4-31b-it gemma-4-31b-it || exit 1; GP="$GP,$p"; done; GP=${GP#,}; BJ="gemma-4-31b-it@$GP"
for p in ${P//,/ }; do wait_up $p gpt-oss || { log NOTREADY; exit 1; }; done; for p in ${GP//,/ }; do wait_up $p gemma || { log NOTREADY; exit 1; }; done
log "ready gpt-oss $P gemma ${BJ:-none}"
~/data/rtm/alfenv/bin/python a6judge.py --src $SRC --port $P --beam-judge $BJ --workers 160; log "JUDGE exit $? for $SRC"
