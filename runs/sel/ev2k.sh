#!/bin/bash
# Step 4 (PLAN_NEXT v2, 11-02): EverMem dev (150 questions) with a 2000-round candidate pool instead of 600.
# Same first stage (Qwen3-Embedding-0.6B + BM25, RRF k=60) with --P 2000, no teacher scores (--T 0). Then A0, CruxRank
# stage 1 (seed 0) and Qwen3-Reranker-8B rank the 2000 pool with infer.py --P 2000 (pointwise, same packing and budgets
# as step 1). Compare with the step-1 P1 files (pool 600): pool gold coverage and session recall at 1k/2k/4k/16k.
# usage: setsid nohup bash ev2k.sh > ev2k.log 2>&1 &
set -u
cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
exec 9>/tmp/rtm_ev2k.lock; flock -n 9 || { log "another GPU script is running"; exit 2; }
unset MASK_DATE; export K1_BUDGETS=1000,2000,4000,16000
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
M=/home/centuser/data/models; CK=/home/centuser/data/rtm/sel/ckpt; PFX=rtm-ev2k-
cleanup(){ timeout 120 bash -c "docker ps -a --format '{{.Names}}' | grep '^$PFX' | xargs -r docker rm -f" >/dev/null 2>&1; }
trap cleanup EXIT
free(){ ! ss -ltn 2>/dev/null | grep -q ":$1 " || { log "port $1 in use"; return 1; }; }
up(){ local name=$1 port=$2 ready=0; for k in $(seq 90); do
    [ "$(docker inspect -f '{{.State.Running}}' $name 2>/dev/null)" = true ] || { log "$name died"; return 1; }
    curl -sf -m 3 localhost:$port/v1/models >/dev/null && { ready=1; break; }; sleep 10; done; [ $ready = 1 ]; }

log start
free 18311 || exit 1
docker run -d --name ${PFX}emb --gpus '"device=4"' --ipc=host --network host -v $M/qwen/Qwen3-Embedding-0.6B:/model:ro $IMG /model --served-model-name Qwen3-Embedding-0.6B --host 127.0.0.1 --port 18311 --runner pooling --convert embed --max-model-len 8192 --gpu-memory-utilization 0.9 >/dev/null || exit 1
up ${PFX}emb 18311 || exit 1
python3 pool.py q_ev150.json --P 2000 --T 0 --par 8 --out pool_ev150_2k > ev2k_pool.log 2>&1 || { log "POOL FAIL"; exit 1; }
docker rm -f ${PFX}emb >/dev/null 2>&1; log "pool done"

run(){ local g=$1 dir=$2 tag=$3 port=$((18420+$1)) c t=P4${3}_ev150; free $port || return 1
  bash serve_sel.sh ${PFX}$g $g $port $dir | grep -q served || { log "START FAIL $tag"; return 1; }
  up ${PFX}$g $port || return 1
  rm -f ctx_${t}_tmp.jsonl
  python3 infer.py q_ev150.json --pooldir pool_ev150_2k --P 2000 --port $port --tag ${t}_tmp --cond 0 --par 16 > inf_$t.log 2>&1 \
    && python3 p1check.py ctx_${t}_tmp.jsonl q_ev150.json pool_ev150_2k 0 2000 >> inf_$t.log 2>&1 \
    && mv -f ctx_${t}_tmp.jsonl ctx_$t.jsonl; c=$?
  docker rm -f ${PFX}$g >/dev/null 2>&1; [ $c = 0 ] && log "done $t" || log "FAIL $t"; return $c; }
# GPUs 4 and 5 only (0-3: generic rerankers, 6-7: 8B step-1 run)
( run 4 $M/qwen/Qwen3-Reranker-0.6B A0 && run 4 $M/qwen/Qwen3-Reranker-8B RR8 ) & a=$!
run 5 $CK/d3_e E0 & b=$!
X=0; for p in $a $b; do wait $p || X=1; done; log "EV2KDONE code $X"; exit $X
