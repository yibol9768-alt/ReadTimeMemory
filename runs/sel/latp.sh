#!/bin/bash
# Step 2 (PLAN_NEXT v2, 11-02): reranking-stage latency, one model per GPU, BEAM dev pools (600 candidates per query).
# Same serving as step 1 (p1.sh). lat.py: 5 warm-up queries, then 50 timed sequential queries, then the same 50 at
# 16 concurrent queries for throughput. GPU: H20 96 GB, one model per GPU, vLLM image as in p1.sh.
# usage: setsid nohup bash lat.sh > lat.log 2>&1 &
set -u
cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
exec 9>/tmp/rtm_latp.lock; flock -n 9 || { log "p1.sh or lat.sh already running"; exit 2; }
unset MASK_DATE
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
M=/home/centuser/data/models; CK=/home/centuser/data/rtm/sel/ckpt; RRD=$M/rerankers; PFX=rtm-lat-
cleanup(){ timeout 120 bash -c "docker ps -a --format '{{.Names}}' | grep '^$PFX' | xargs -r docker rm -f" >/dev/null 2>&1; }
trap cleanup EXIT
free(){ ! ss -ltn 2>/dev/null | grep -q ":$((18400+$1)) " || { log "port $((18400+$1)) in use"; return 1; }; }
start(){ local g=$1 kind=$2 dir=$3 len=${4:-0}; docker rm -f $PFX$g >/dev/null 2>&1; free $g || return 1
  if [ $kind = q ]; then bash serve_sel.sh $PFX$g $g $((18400+g)) $dir | grep -q served
  else docker run -d --name $PFX$g --gpus "\"device=$g\"" --ipc=host --network host -v $dir:/model:ro $IMG /model --served-model-name sel --host 127.0.0.1 --port $((18400+g)) --runner pooling --max-model-len $len --max-num-seqs 256 --gpu-memory-utilization 0.6 >/dev/null; fi; }
up(){ local g=$1 ready=0; for k in $(seq 90); do
    [ "$(docker inspect -f '{{.State.Running}}' $PFX$g 2>/dev/null)" = true ] || { log "container $PFX$g died"; return 1; }
    curl -sf -m 3 localhost:$((18400+g))/v1/models | grep -q '"id":"sel"' && { ready=1; break; }; sleep 10; done; [ $ready = 1 ]; }
one(){ local g=$1 kind=$2 dir=$3 tag=$4 extra=$5 len=${6:-0} c
  [ -e "$dir/config.json" ] || { log "skip $tag: $dir missing"; return 0; }
  start $g $kind $dir $len || { log "START FAIL $tag"; return 1; }
  up $g && python3 lat.py beam_ex.json --pooldir pool --port $((18400+g)) --tag $tag $extra >> lat_$tag.log 2>&1; c=$?
  docker rm -f $PFX$g >/dev/null 2>&1; [ $c = 0 ] && log "done $tag" || log "FAIL $tag"; return $c; }
log start
one 0 p $RRD/bge-reranker-large BGEL "--plain --trunc 512" 512 & a=$!
one 1 p $RRD/bge-reranker-v2-m3 BGEM3 "--plain" 8192 & b=$!
one 2 p $RRD/gte-reranker-modernbert-base GTE "--plain" 8192 & c=$!
X=0; for p in $a $b $c; do wait $p || X=1; done; log "LATDONE code $X"; exit $X
