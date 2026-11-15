#!/bin/bash
# Work that does not depend on the exposure gate: control and hard-skip 0 on GPUs 6,7; corrected BEAM for W20, A0 and the morning arms on GPUs 0-4.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
trap 'docker rm -f rtm-sel6 rtm-sel7 rtm-train-p0 rtm-train-p1 rtm-train-p2 rtm-train-h0 >/dev/null 2>&1; for g in 0 1 2 3 4; do docker rm -f rtm-ev$g >/dev/null 2>&1; done' EXIT
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 3000 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
up(){ for k in $(seq 80); do ok=1; for p in "$@"; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
base(){ echo "--model /w/ckpt/w20 --lr 1e-5 --epochs 1 --max-d 1100 --w-base 0.5 --seed $1"; }
CK=/home/centuser/data/rtm/sel/ckpt; A0=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
srv(){ bash serve_sel.sh rtm-ev$1 $1 $((18360+$1)) $2; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$((18360+$1))/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
inf(){ local g=$1 b=$2 f=$3 pd=$4 t=$5; rm -f ctx_$t.jsonl; K1_BUDGETS=$b python3 infer.py $f --pooldir $pd --port $((18360+g)) --tag $t --cond 0 --par 16 > inf_$t.log 2>&1 || { log "FAIL $t"; return 1; }; log "done $t"; }
( for g in 6 7; do bash serve_sel.sh rtm-sel$g $g $((18341+g)) $CK/w20; done; up 18347 18348 || exit 1
  K1_BUDGETS=1000,2000,4000 python3 infer.py env_ws7ce.json --pooldir pool_ws7ce --port 18347,18348 --tag W20_ws7ce --cond 0 --par 16 > inf_W20_ws7ce.log 2>&1 || exit 1
  docker rm -f rtm-sel6 rtm-sel7 >/dev/null 2>&1; sleep 5; rm -f v9_pre_ok; python3 pcgate3.py pre W20_ws7ce > d3v9_pre.txt 2>&1 && touch v9_pre_ok; cat d3v9_pre.txt
  if [ -f v9_pre_ok ]; then for s in 0 1 2; do TRAIN p$s 6,7 2 $(base $s) --data /w/tr_v9_pc.jsonl --out /w/ckpt/v9_p$s > v9_tr_p$s.log 2>&1 || exit 1; done; log control trained; else log "control invalid: pre gate failed"; fi
  TRAIN h0 6,7 2 $(base 0) --data /w/tr_v9_hs0.jsonl --out /w/ckpt/v9_h0 > v9_tr_h0.log 2>&1 || exit 1; log hs0 trained ) & Q=$!
( srv 0 $CK/w20 && inf 0 4000,16000 beam_ex.json pool W20X_H ) & a=$!; ( srv 1 $A0 && inf 1 4000,16000 beam_ex.json pool A0X_H ) & b=$!
( srv 2 $CK/d3_fix && inf 2 4000,16000 beam_ex.json pool D3RFX_H ) & c=$!; ( srv 3 $CK/d3_e && inf 3 4000,16000 beam_ex.json pool D3REX_H ) & d=$!; ( srv 4 $CK/d3_hard && inf 4 4000,16000 beam_ex.json pool D3RHX_H ) & e=$!
R=0; for p in $Q $a $b $c $d $e; do wait $p || R=1; done; log "D3V9CDONE code $R"; exit $R
