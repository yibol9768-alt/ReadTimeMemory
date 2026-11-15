#!/bin/bash
# D3 descriptive transfer check (not a gate): each D3 arm on the S2 dev sets, same pools and budgets as W20 in S2. usage: bash d3real.sh
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 rtm-sel4 rtm-sel5 >/dev/null 2>&1' EXIT
CK=/home/centuser/data/rtm/sel/ckpt; i=0
for a in e fix hard; do for g in $((2*i)) $((2*i+1)); do bash serve_sel.sh rtm-sel$g $g $((18341+g)) $CK/d3_$a; done; i=$((i+1)); done
for k in $(seq 80); do ok=1; for p in 18341 18342 18343 18344 18345 18346; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done
run(){ p=$1; t=$2
  K1_BUDGETS=4000,16000 python3 infer.py beam_ex.json --pooldir pool --port $p --tag ${t}_H --cond 0 --par 8 || return 1
  K1_BUDGETS=1000,2000,4000 python3 infer.py q_lmes150.json --pooldir pool_lmes150 --port $p --tag ${t}_lmes150 --cond 0 --par 8 || return 1
  K1_BUDGETS=1000 python3 infer.py q_ep150.json --pooldir pool_ep150 --port $p --tag ${t}_ep150 --cond 0 --par 8 || return 1
  K1_BUDGETS=4000 python3 infer.py q_ev150.json --pooldir pool_ev150 --port $p --tag ${t}_ev150 --cond 0 --par 8 || return 1; }
rm -f ctx_D3RE_*.jsonl ctx_D3RF_*.jsonl ctx_D3RH_*.jsonl
run 18341,18342 D3RE > d3real_e.log 2>&1 & a=$!; run 18343,18344 D3RF > d3real_f.log 2>&1 & b=$!; run 18345,18346 D3RH > d3real_h.log 2>&1 & c=$!
wait $a && wait $b && wait $c || { log FAIL; exit 1; }
for t in D3RE D3RF D3RH; do python3 ssum.py $t > d3real_$t.txt 2>&1; done; echo "== W20 for reference"; python3 ssum.py W20 > d3real_W20.txt 2>&1
log D3REALDONE
