#!/bin/bash
# Control on 120 questions: the user waived the 128-question pre gate on 11-01 14:13 (n=120). Train seeds 0,1,2 on GPUs 2,3, then one GPU per seed on the control eval worlds, then the frozen post gate.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
trap 'docker rm -f rtm-train-p0 rtm-train-p1 rtm-train-p2 rtm-ev2 rtm-ev3 rtm-ev4 >/dev/null 2>&1' EXIT
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 3000 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
CK=/home/centuser/data/rtm/sel/ckpt
srv(){ bash serve_sel.sh rtm-ev$1 $1 $((18360+$1)) $2; sleep 15; for k in $(seq 80); do curl -s -m 3 localhost:$((18360+$1))/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
until grep -q "D3V9CDONE" d3v9c.log; do sleep 15; done
for s in 0 1 2; do TRAIN p$s 2,3 2 --model /w/ckpt/w20 --lr 1e-5 --epochs 1 --max-d 1100 --w-base 0.5 --seed $s --data /w/tr_v9_pc.jsonl --out /w/ckpt/v9_p$s > v9_tr_p$s.log 2>&1 || { log FAIL train $s; exit 1; }; done; log control trained
P=""; for s in 0 1 2; do g=$((s+2)); ( srv $g $CK/v9_p$s || exit 1; rm -f ctx_V9P${s}_ws7ce.jsonl; K1_BUDGETS=1000,2000,4000 python3 infer.py env_ws7ce.json --pooldir pool_ws7ce --port $((18360+g)) --tag V9P${s}_ws7ce --cond 0 --par 16 > inf_V9P$s.log 2>&1 ) & P="$P $!"; done
for p in $P; do wait $p || { log FAIL eval; exit 1; }; done
python3 pcgate3.py post W20_ws7ce V9P0_ws7ce,V9P1_ws7ce,V9P2_ws7ce > d3v9_control.txt 2>&1; r=$?; cat d3v9_control.txt; log "D3V9PDONE code $r"
