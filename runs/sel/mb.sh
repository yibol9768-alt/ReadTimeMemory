#!/bin/bash
# Multi-benchmark recall table (10-31). usage: bash mb.sh NAME QFILE IDFILE
# Same first stage for every model: dense+BM25 RRF, P=600; 8B scores = teacher pool. Then each model reranks the pool.
# GPUs: 6,7 = 8B pool; 0,1 / 2,3 = 0.6B models; 4,5? 5 is the embedding service, so MemReranker-4B uses 0,1 after the 0.6B runs.
set -eu
cd ~/data/rtm/sel
N=$1; Q=$2; IDS=$3
export K1_BUDGETS=${MB_BUDGETS:-4000,16000}
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
python3 - "$Q" "$IDS" "q_$N.json" <<'PY'
import json,sys
k=set(open(sys.argv[2]).read().split()); q=[x for x in json.load(open(sys.argv[1])) if x["question_id"] in k]
json.dump(q,open(sys.argv[3],"w")); print("questions",len(q))
PY
docker rm -f rtm-qa-6 rtm-qa-7 >/dev/null 2>&1 || true; sleep 5
bash ~/data/rtm/env/up_rr.sh
for k in $(seq 80); do curl -s -m 3 localhost:18312/v1/models >/dev/null && curl -s -m 3 localhost:18313/v1/models >/dev/null && break; sleep 15; done
log "pool $N"
python3 pool.py q_$N.json --rr-port 18312,18313 --T 600 --par 8 --out pool_$N
docker rm -f rtm-env-rr rtm-env-rr2 >/dev/null 2>&1 || true
python3 cov.py rr8 q_$N.json RR8_$N --pooldir pool_$N
run(){ tag=$1; m=$2; g1=$3; g2=$4; p1=$5; p2=$6
  bash serve_sel.sh rtm-sel$g1 $g1 $p1 $m; bash serve_sel.sh rtm-sel$g2 $g2 $p2 $m; sleep 20
  for k in $(seq 80); do curl -s -m 3 localhost:$p1/v1/models >/dev/null && curl -s -m 3 localhost:$p2/v1/models >/dev/null && break; sleep 15; done
  python3 infer.py q_$N.json --port $p1,$p2 --tag ${tag}_$N --pooldir pool_$N --cond 0 --par 8
  docker rm -f rtm-sel$g1 rtm-sel$g2 >/dev/null 2>&1 || true; log "done $tag"; }
B06=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B; C=/home/centuser/data/rtm/sel/ckpt
run A0 $B06 0 1 18341 18342 &
run T1 $C/t1 2 3 18343 18344 &
run T2X $C/t2x 6 7 18345 18346 &
wait
run MR4 /home/centuser/data/models/qwen/MemReranker-4B 0 1 18341 18342
log "MBDONE $N"
