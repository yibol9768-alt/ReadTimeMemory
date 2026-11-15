#!/bin/bash
# D4 seed variance: C s1, C s2, P0 s1, P0P1 s1 (same frozen data), then H4 eval of every checkpoint. Stops at 17:30 ET.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; B=/models/qwen/Qwen3-Reranker-0.6B
trap 'docker ps --format "{{.Names}}" | grep -E "^rtm-d4t-|^rtm-d4e-" | xargs -r docker rm -f >/dev/null 2>&1' EXIT
TRAIN(){ name=$1; g=$2; shift 2; timeout 7200 docker run --rm --name rtm-d4t-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node 2 --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
TR="--model $B --lr 1e-5 --max-d 1100 --accum 8 --save-every 100 --epochs 1"
TRAIN C_s1 0,1 $TR --seed 1 --data /w/d4dev/C.jsonl --out /w/ckpt/d4_C_s1 > d4/tr_C_s1.log 2>&1 & a=$!
TRAIN C_s2 2,3 $TR --seed 2 --data /w/d4dev/C.jsonl --out /w/ckpt/d4_C_s2 > d4/tr_C_s2.log 2>&1 & b=$!
TRAIN P0_s1 4,5 $TR --seed 1 --data /w/d4dev/P0.jsonl --out /w/ckpt/d4_P0_s1 > d4/tr_P0_s1.log 2>&1 & c=$!
TRAIN P0P1_s1 6,7 $TR --seed 1 --data /w/d4dev/P0P1.jsonl --out /w/ckpt/d4_P0P1_s1 > d4/tr_P0P1_s1.log 2>&1 & d=$!
for p in $a $b $c $d; do wait $p || { log "FAIL train $p"; exit 1; }; done; log trained
JOBS=(); for a in C_s1 C_s2 P0_s1 P0P1_s1; do v=$([ ${a%%_*} = C ] && echo 1 || echo 2); for s in 100 200 300 400 500 600; do JOBS+=("$a:$s:$v"); done; done
ev(){ g=$1; shift; for j in "$@"; do IFS=: read a s v <<< "$j"; [ "$(TZ=America/New_York date +%H%M)" -ge 1730 ] && { log LATE; return 1; }
  bash serve_sel.sh rtm-d4e-$g $g $((18370+g)) /home/centuser/data/rtm/sel/ckpt/d4_$a/s$s >/dev/null; for k in $(seq 60); do curl -s -m 3 localhost:$((18370+g))/v1/models >/dev/null && break; sleep 10; done
  rm -f ctx_D4${a}s${s}_h4.jsonl; FMT_INST=$v K1_BUDGETS=1000,2000,4000,8000,16000 python3 infer.py env_h4.json --pooldir pool_h4 --port $((18370+g)) --tag D4${a}s${s}_h4 --cond 0 --par 16 > d4/inf_${a}s${s}.log 2>&1 || { log "FAIL infer $a $s"; return 1; }
  docker rm -f rtm-d4e-$g >/dev/null; log "eval $a s$s"; done; }
P=(); for g in 0 1 2 3 4 5 6 7; do sub=(); for ((i=g; i<${#JOBS[@]}; i+=8)); do sub+=("${JOBS[$i]}"); done; ev $g "${sub[@]}" & P+=($!); done
for p in "${P[@]}"; do wait $p; done
S6=100,200,300,400,500,600
python3 d4/d4eval.py C:D4C:$S6 C_s1:D4C_s1:$S6 C_s2:D4C_s2:$S6 P0:D4P0:$S6 P0_s1:D4P0_s1:$S6 P0P1:D4P0P1:$S6 P0P1_s1:D4P0P1_s1:$S6 > d4/sum_seeds.txt 2>&1; cat d4/sum_seeds.txt; log D4SEEDSDONE
