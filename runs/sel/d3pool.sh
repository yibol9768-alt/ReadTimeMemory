#!/bin/bash
# EverMem pool size test after the answer table: first stage top 2000 instead of 600, then A0 and morning fixed score the whole pool. Recall only, no training. All containers removed by 17:40 EST.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
stop(){ docker rm -f rtm-env-rr rtm-env-rr2 rtm-rrx0 rtm-rrx1 rtm-rrx2 rtm-rrx3 rtm-rrx4 rtm-ev0 rtm-ev1 rtm-ev2 rtm-ev3 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1; }; trap stop EXIT
END=$(TZ=America/New_York date -d "today 17:40" +%s); left(){ echo $(( END - $(date +%s) )); }
until grep -q A5SHDONE a5sh.log; do [ $(left) -gt 600 ] || { log "no time"; exit 0; }; sleep 15; done; sleep 10
docker start rtm-env-emb >/dev/null; bash ~/data/rtm/env/up_rr.sh; bash up_rrx.sh; R=18312,18313,18370,18371,18372,18373,18374
for k in $(seq 80); do ok=1; for p in 18311 ${R//,/ }; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done; [ $ok = 1 ] || { log FAIL rr; exit 1; }
timeout -k 20 $(( $(left) - 900 )) python3 pool.py q_ev150.json --rr-port $R --P 2000 --T 600 --par 16 --out pool_ev150_2k > pool_ev150_2k.log 2>&1 || { log "pool stopped"; exit 1; }; log pool done
docker rm -f rtm-env-rr rtm-env-rr2 rtm-rrx0 rtm-rrx1 rtm-rrx2 rtm-rrx3 rtm-rrx4 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1; sleep 10
CK=/home/centuser/data/rtm/sel/ckpt; A0=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
srv(){ bash serve_sel.sh rtm-ev$1 $1 $((18360+$1)) $2; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$((18360+$1))/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
P=""; g=0; for x in "$A0 A0P2" "$CK/d3_fix D3FP2"; do set -- $x; ( srv $g $1 && rm -f ctx_$2_ev150.jsonl && K1_BUDGETS=4000 timeout -k 20 $(left) python3 infer.py q_ev150.json --pooldir pool_ev150_2k --P 2000 --port $((18360+g)) --tag $2_ev150 --cond 0 --par 16 > inf_$2.log 2>&1 ) & P="$P $!"; g=$((g+1)); done
X=0; for p in $P; do wait $p || X=1; done; log "D3POOLDONE code $X"
