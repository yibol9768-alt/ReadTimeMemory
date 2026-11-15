#!/bin/bash
# EverMem date-masking probe (audit 11-02 item 12): E0 (CruxRank stage 1 seed 0) and A0 rank the EverMem dev pool (600) with
# candidate dates replaced by "-" (fmt.py MASK_DATE). Compared with P1 (unmasked) on session recall. GPUs 0 and 4.
set -u; cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
exec 9>/tmp/rtm_evmask.lock; flock -n 9 || { log "already running"; exit 2; }
export MASK_DATE=1 K1_BUDGETS=1000,2000,4000,8000,16000
M=/home/centuser/data/models; CK=/home/centuser/data/rtm/sel/ckpt; PFX=rtm-evmask-
trap 'docker ps -a --format "{{.Names}}" | grep "^$PFX" | xargs -r docker rm -f >/dev/null 2>&1' EXIT
run(){ local g=$1 dir=$2 tag=$3 port=$((18430+$1)) t=P5${3}M_ev150 c
  ss -ltn | grep -q ":$port " && { log "port $port busy"; return 1; }
  bash serve_sel.sh ${PFX}$g $g $port $dir | grep -q served || { log "START FAIL $tag"; return 1; }
  for k in $(seq 90); do curl -sf -m3 localhost:$port/v1/models >/dev/null && break; sleep 10; done
  rm -f ctx_${t}_tmp.jsonl
  python3 infer.py q_ev150.json --pooldir pool_ev150 --port $port --tag ${t}_tmp --cond 0 --par 16 > inf_$t.log 2>&1 \
    && python3 p1check.py ctx_${t}_tmp.jsonl q_ev150.json pool_ev150 0 600 >> inf_$t.log 2>&1 \
    && mv -f ctx_${t}_tmp.jsonl ctx_$t.jsonl; c=$?
  docker rm -f ${PFX}$g >/dev/null 2>&1; [ $c = 0 ] && log "done $t" || log "FAIL $t"; return $c; }
log start
run 0 $CK/d3_e E0 & a=$!; run 4 $M/qwen/Qwen3-Reranker-0.6B A0 & b=$!
X=0; for p in $a $b; do wait $p || X=1; done; log "EVMASKDONE code $X"
