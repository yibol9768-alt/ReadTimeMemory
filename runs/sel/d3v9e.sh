#!/bin/bash
# D3 v9 batch one: real dev sets for every seed of E and fixed, hard-skip 0, control on its own worlds, corrected W20 and A0 on BEAM, corrected morning arms. One GPU per queue.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'for g in 0 1 2 3 4 5 6 7; do docker rm -f rtm-ev$g >/dev/null 2>&1; done' EXIT
CK=/home/centuser/data/rtm/sel/ckpt; A0=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
srv(){ bash serve_sel.sh rtm-ev$1 $1 $((18360+$1)) $2; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$((18360+$1))/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
inf(){ local g=$1 b=$2 f=$3 pd=$4 t=$5; rm -f ctx_$t.jsonl; K1_BUDGETS=$b python3 infer.py $f --pooldir $pd --port $((18360+g)) --tag $t --cond 0 --par 16 > inf_$t.log 2>&1 || { log "FAIL $t"; return 1; }; log "done $t"; }
real(){ local g=$1 t=$2; inf $g 1000 q_ep150.json pool_ep150 ${t}_ep150 && inf $g 4000 q_ev150.json pool_ev150 ${t}_ev150 && inf $g 1000,2000,4000 q_lmes150.json pool_lmes150 ${t}_lmes150 && inf $g 4000,16000 beam_ex.json pool ${t}_H; }
q(){ local g=$1 m=$2 t=$3; srv $g $m || { log "FAIL serve $t"; return 1; }; real $g $t; }
( q 0 $CK/v9_e0 V9E0 ) & a=$!; ( q 1 $CK/v9_e1 V9E1 ) & b=$!; ( q 2 $CK/v9_e2 V9E2 ) & c=$!; ( q 3 $CK/v9_f0 V9F0 ) & d=$!; ( q 4 $CK/v9_f1 V9F1 ) & e=$!; ( q 5 $CK/v9_f2 V9F2 ) & f=$!
( q 6 $CK/v9_h0 V9H0 && srv 6 $A0 && inf 6 4000,16000 beam_ex.json pool A0X_H && srv 6 $CK/d3_fix && inf 6 4000,16000 beam_ex.json pool D3RFX_H ) & g=$!
( if [ -f v9_pre_ok ]; then for s in 0 1 2; do srv 7 $CK/v9_p$s || exit 1; inf 7 1000,2000,4000 env_ws7ce.json pool_ws7ce V9P${s}_ws7ce || exit 1; done; fi; srv 7 $CK/w20 && inf 7 4000,16000 beam_ex.json pool W20X_H && srv 7 $CK/d3_e && inf 7 4000,16000 beam_ex.json pool D3REX_H && srv 7 $CK/d3_hard && inf 7 4000,16000 beam_ex.json pool D3RHX_H ) & h=$!
R=0; for p in $a $b $c $d $e $f $g $h; do wait $p || R=1; done; [ $R = 0 ] || { log "FAIL some queue"; exit 1; }
if [ -f v9_pre_ok ]; then python3 pcgate3.py post W20_ws7ce V9P0_ws7ce,V9P1_ws7ce,V9P2_ws7ce > d3v9_control.txt 2>&1; else echo "CONTROL INVALID: pre gate failed" > d3v9_control.txt; fi; cat d3v9_control.txt
python3 v9sum.py > d3v9_real.txt 2>&1; v=$?; cat d3v9_real.txt; [ $v = 0 ] || { log "SUMMARY INCOMPLETE"; exit 1; }
log D3V9EVALDONE
