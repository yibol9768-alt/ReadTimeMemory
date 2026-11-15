#!/bin/bash
# official-prompt baseline for LongMemEval-S 150 (same recipe as the BEAM A0O arm)
set -eu
cd ~/data/rtm/sel
export K1_BUDGETS=1000,2000,4000
B06=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
bash serve_sel.sh rtm-sel0 0 18341 $B06; bash serve_sel.sh rtm-sel1 1 18342 $B06; sleep 20
for k in $(seq 80); do curl -s -m 3 localhost:18341/v1/models >/dev/null && curl -s -m 3 localhost:18342/v1/models >/dev/null && break; sleep 15; done
python3 pool.py q_lmes150.json --rr-port 18341,18342 --rr-model sel --T 600 --par 8 --out pool_lmes150_06
python3 cov.py rr8 q_lmes150.json A0O_lmes150 --pooldir pool_lmes150_06
docker rm -f rtm-sel0 rtm-sel1 >/dev/null 2>&1 || true
python3 mbsum.py lmes150 RR8,A0,A0O,MR4,T2X,T1
echo A0OLMEDONE
