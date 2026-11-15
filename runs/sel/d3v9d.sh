#!/bin/bash
# After d3v9c: real dev sets for hard-skip 0 (GPU 6) and the short-positive pilot (train on 0,1; eval on GPU 7). Same recipe as the D3 arms.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
trap 'docker rm -f rtm-train-s0 rtm-ev6 rtm-ev7 >/dev/null 2>&1' EXIT
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 3000 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
CK=/home/centuser/data/rtm/sel/ckpt
srv(){ bash serve_sel.sh rtm-ev$1 $1 $((18360+$1)) $2; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$((18360+$1))/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
inf(){ local g=$1 b=$2 f=$3 pd=$4 t=$5; rm -f ctx_$t.jsonl; K1_BUDGETS=$b python3 infer.py $f --pooldir $pd --port $((18360+g)) --tag $t --cond 0 --par 16 > inf_$t.log 2>&1 || { log "FAIL $t"; return 1; }; log "done $t"; }
real(){ local g=$1 t=$2; inf $g 1000 q_ep150.json pool_ep150 ${t}_ep150 && inf $g 4000 q_ev150.json pool_ev150 ${t}_ev150 && inf $g 1000,2000,4000 q_lmes150.json pool_lmes150 ${t}_lmes150 && inf $g 4000,16000 beam_ex.json pool ${t}_H; }
until grep -q "D3V9CDONE" d3v9c.log; do sleep 15; done; grep -q "D3V9CDONE code 0" d3v9c.log || { log "d3v9c failed"; exit 1; }
python3 subshort.py tr_v9_short.jsonl 1500 > subshort.log 2>&1 || { log "short pilot not executable"; cat subshort.log; }
( srv 6 $CK/v9_h0 && real 6 V9H0 ) & a=$!
( [ -f tr_v9_short.jsonl ] || exit 0; TRAIN s0 0,1 2 --model /w/ckpt/w20 --lr 1e-5 --epochs 1 --max-d 1100 --w-base 0.5 --seed 0 --data /w/tr_v9_short.jsonl --out /w/ckpt/v9_s0 > v9_tr_s0.log 2>&1 && log short trained && srv 7 $CK/v9_s0 && real 7 V9S0 ) & b=$!
R=0; wait $a || R=1; wait $b || R=1; log "D3V9DDONE code $R"; exit $R
