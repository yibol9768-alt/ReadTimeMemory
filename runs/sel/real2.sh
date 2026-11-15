#!/bin/bash
# Corrected-pack rerun on GPU 5 (11-01 audit): new tags *X, every model through the fixed infer.py. Stops before the pool step needs GPU 5.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-real5 >/dev/null 2>&1' EXIT
CK=/home/centuser/data/rtm/sel/ckpt; A0=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B; P=18351
late(){ grep -q generated d3v8w.log; }
srv(){ bash serve_sel.sh rtm-real5 5 $P $1; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$P/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
inf(){ b=$1; f=$2; pd=$3; t=$4; rm -f ctx_$t.jsonl; K1_BUDGETS=$b python3 infer.py $f --pooldir $pd --port $P --tag $t --cond 0 --par 16 > inf_$t.log 2>&1 || { log "FAIL $t"; exit 1; }; log "done $t"; }
srv $A0 || exit 1; inf 1000,2000,4000 q_lmes150.json pool_lmes150 A0X_lmes150; inf 1000 q_ep150.json pool_ep150 A0X_ep150; inf 4000 q_ev150.json pool_ev150 A0X_ev150
srv $CK/w20 || exit 1; inf 1000,2000,4000 q_lmes150.json pool_lmes150 W20X_lmes150; inf 1000,2000,4000 env_ws6f.json pool_ws6f W20X_ws6f
for x in "d3_fix D3RFX" "d3_e D3REX" "d3_hard D3RHX"; do set -- $x; late && { log "stop for pools"; exit 0; }; srv $CK/$1 || exit 1; inf 1000,2000,4000 q_lmes150.json pool_lmes150 $2_lmes150; done
log REAL2DONE
