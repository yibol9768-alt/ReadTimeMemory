#!/bin/bash
# LoCoMo development recall: after batch two, pools on all reranker GPUs, then A0, W20, morning fixed and E. Reader not included. Results join the 17:10 batch.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-env-rr rtm-env-rr2 rtm-rrx0 rtm-rrx1 rtm-rrx2 rtm-rrx3 rtm-rrx4 rtm-ev0 rtm-ev1 rtm-ev2 rtm-ev3 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1' EXIT
CK=/home/centuser/data/rtm/sel/ckpt; A0=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
until grep -q D3XDONE d3x.log; do sleep 15; done; sleep 10
docker start rtm-env-emb >/dev/null; bash ~/data/rtm/env/up_rr.sh; bash up_rrx.sh; R=18312,18313,18370,18371,18372,18373,18374
for k in $(seq 80); do ok=1; for p in 18311 ${R//,/ }; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done; [ $ok = 1 ] || { log FAIL rr; exit 1; }
python3 pool.py q_loc150.json --rr-port $R --T 600 --par 32 --out pool_loc150 > pool_loc150.log 2>&1 || { log FAIL pool; exit 1; }; log pool done
docker rm -f rtm-env-rr rtm-env-rr2 rtm-rrx0 rtm-rrx1 rtm-rrx2 rtm-rrx3 rtm-rrx4 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1; sleep 10
srv(){ bash serve_sel.sh rtm-ev$1 $1 $((18360+$1)) $2; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$((18360+$1))/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
P=""; g=0; for x in "$A0 A0X" "$CK/w20 W20X" "$CK/d3_fix D3RFX" "$CK/d3_e D3REX"; do set -- $x; ( srv $g $1 && rm -f ctx_$2_loc150.jsonl && K1_BUDGETS=1000,2000,4000 python3 infer.py q_loc150.json --pooldir pool_loc150 --port $((18360+g)) --tag $2_loc150 --cond 0 --par 16 > inf_$2_loc.log 2>&1 ) & P="$P $!"; g=$((g+1)); done
X=0; for p in $P; do wait $p || X=1; done; log "D3LOCDONE code $X"; exit $X
