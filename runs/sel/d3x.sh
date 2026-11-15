#!/bin/bash
# Codex-frozen batch two (11-01 15:00): a) reader on GPU 0; b) seeds 1,2 of the morning fixed and E recipes on the archived data; c) fixed rank reruns for fusion on GPU 1. No result is read before 17:10.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; C=rtm-qa-0; P0=18300
trap 'docker rm -f rtm-qa-0 rtm-train-f1 rtm-train-f2 rtm-train-e1 rtm-train-e2 rtm-ev1 rtm-ev2 rtm-ev3 rtm-ev4 rtm-ev5 >/dev/null 2>&1' EXIT
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 3000 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
CK=/home/centuser/data/rtm/sel/ckpt
base(){ echo "--model /w/ckpt/w20 --lr 1e-5 --epochs 1 --max-d 1100 --w-base 0.5 --seed $1"; }
srv(){ bash serve_sel.sh rtm-ev$1 $1 $((18360+$1)) $2; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$((18360+$1))/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
inf(){ local g=$1 b=$2 f=$3 pd=$4 t=$5; rm -f ctx_$t.jsonl; K1_BUDGETS=$b python3 infer.py $f --pooldir $pd --port $((18360+g)) --tag $t --cond 0 --par 16 > inf_$t.log 2>&1 || { log "FAIL $t"; return 1; }; log "done $t"; }
real(){ local g=$1 t=$2; inf $g 1000 q_ep150.json pool_ep150 ${t}_ep150 && inf $g 4000 q_ev150.json pool_ev150 ${t}_ev150 && inf $g 1000,2000,4000 q_lmes150.json pool_lmes150 ${t}_lmes150 && inf $g 4000,16000 beam_ex.json pool ${t}_H; }
( docker run -d --name $C --gpus '"device=0"' --network host --ipc host -v /home/centuser/data/models/Qwen/Qwen3.8-27B-FP8:/model:ro $IMG /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port $P0 --enable-prefix-caching --max-model-len 131072 --gpu-memory-utilization 0.9 --max-num-seqs 128 >/dev/null || exit 1
  for i in $(seq 1 90); do curl -s -m 3 localhost:$P0/v1/models | grep -q Qwen && break; sleep 10; done; python3 c2.py --port $P0 > c2.log 2>&1 || exit 1; docker rm -f $C >/dev/null 2>&1; log reader done ) & Q=$!
( srv 1 $CK/d3_fix && inf 1 1000 q_ep150.json pool_ep150 D3RFR_ep150 && inf 1 4000 q_ev150.json pool_ev150 D3RFR_ev150 ) & R=$!
TRAIN f1 2,3 2 $(base 1) --data /w/tr_d3_fix.jsonl --out /w/ckpt/d3_fix_s1 > d3_tr_f1.log 2>&1 & a=$!
TRAIN f2 4,5 2 $(base 2) --data /w/tr_d3_fix.jsonl --out /w/ckpt/d3_fix_s2 > d3_tr_f2.log 2>&1 & b=$!
TRAIN e1 6,7 2 $(base 1) --data /w/tr_d3_e.jsonl --out /w/ckpt/d3_e_s1 > d3_tr_e1.log 2>&1 & c=$!
wait $a && wait $b && wait $c || { log FAIL wave1; exit 1; }; log wave1 done
TRAIN e2 6,7 2 $(base 2) --data /w/tr_d3_e.jsonl --out /w/ckpt/d3_e_s2 > d3_tr_e2.log 2>&1 & d=$!
( srv 2 $CK/d3_fix_s1 && real 2 D3F1 ) & e=$!; ( srv 3 $CK/d3_fix_s2 && real 3 D3F2 ) & f=$!; ( srv 4 $CK/d3_e_s1 && real 4 D3E1 ) & g=$!
wait $d || { log FAIL e2; exit 1; }; log e2 trained; ( srv 5 $CK/d3_e_s2 && real 5 D3E2 ) & h=$!
X=0; for p in $Q $R $e $f $g $h; do wait $p || X=1; done; log "D3XDONE code $X"; exit $X
