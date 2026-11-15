#!/bin/bash
# Descriptive transfer of the D3 arms to real dev sets on the one idle GPU (5), while worlds generate. Small sets for all arms first, BEAM last. Frees the GPU before 14:25 EST for the pool step.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-real5 >/dev/null 2>&1' EXIT
CK=/home/centuser/data/rtm/sel/ckpt; P=18351
late(){ [ $(TZ=America/New_York date +%H%M) -ge 1425 ]; }
srv(){ bash serve_sel.sh rtm-real5 5 $P $CK/$1; for k in $(seq 80); do curl -s -m 3 localhost:$P/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
small(){ t=$1; rm -f ctx_${t}_ep150.jsonl ctx_${t}_ev150.jsonl ctx_${t}_lmes150.jsonl
  K1_BUDGETS=1000 python3 infer.py q_ep150.json --pooldir pool_ep150 --port $P --tag ${t}_ep150 --cond 0 --par 16 &&
  K1_BUDGETS=4000 python3 infer.py q_ev150.json --pooldir pool_ev150 --port $P --tag ${t}_ev150 --cond 0 --par 16 &&
  K1_BUDGETS=1000,2000,4000 python3 infer.py q_lmes150.json --pooldir pool_lmes150 --port $P --tag ${t}_lmes150 --cond 0 --par 16; }
for x in "d3_fix D3RF" "d3_e D3RE" "d3_hard D3RH"; do set -- $x; late && { log "out of time before $2"; exit 0; }; srv $1 || { log FAIL serve; exit 1; }; small $2 > d3real_$2.log 2>&1 || { log "FAIL $2"; exit 1; }; log "small done $2"; done
log SMALLDONE
for x in "d3_fix D3RF" "d3_e D3RE" "d3_hard D3RH"; do set -- $x; late && { log "out of time before BEAM $2"; exit 0; }; srv $1 || exit 1; rm -f ctx_$2_H.jsonl; K1_BUDGETS=4000,16000 timeout 3000 python3 infer.py beam_ex.json --pooldir pool --port $P --tag $2_H --cond 0 --par 16 > d3real_$2_H.log 2>&1; log "BEAM $2 rc $?"; done
log D3REAL1DONE
