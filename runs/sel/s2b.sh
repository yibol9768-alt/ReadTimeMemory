#!/bin/bash
# S2 (11-01, run inside 08:30-19:10 ET): 20 worlds, uniform, no evolution. Arms W20 plain and W20C with selected rounds. Held-out world 14 and four benchmarks.
set -u
cd ~/data/rtm/sel
while [ -f poolx.running ] || [ -f poolex.running ] || [ -f check.hold ]; do sleep 5; done   # extra rerankers on GPUs 0-4 must be gone first
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
B=/models/qwen/Qwen3-Reranker-0.6B; HB=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B; C=/home/centuser/data/rtm/sel/ckpt
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 18000 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
waitp(){ sleep 20; for k in $(seq 80); do ok=1; for p in "$@"; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
trap 'docker rm -f rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 rtm-sel4 rtm-sel6 rtm-sel7 >/dev/null 2>&1' EXIT
HS=$(cat hs.txt 2>/dev/null || echo 10)   # set by fnchk.py from the checked false-negative rate
K="--ignore-kinds repeat,fu --hard-skip $HS --boost 3"
python3 mkdata.py env_ws4.json pool_ws4 tr_w20.jsonl --no-set $K --steps 8 --ids ws4_tr.txt || { log "FAIL data"; exit 1; }
python3 mkdata.py env_ws4.json pool_ws4 dv_w20.jsonl --no-set $K --steps 2 --ids ws4_dv.txt || { log "FAIL data"; exit 1; }
python3 mkdata.py env_ws4.json pool_ws4 tr_w20c.jsonl --ignore-kinds fu --fact-stage --hard-skip $HS --boost 3 --steps 8 --kmax 20 --p-mixed 0.3 --p-empty 0.15 --ids ws4_tr.txt || { log "FAIL data"; exit 1; }
python3 mkdata.py env_ws4.json pool_ws4 dv_w20c.jsonl --ignore-kinds fu --fact-stage --hard-skip $HS --boost 3 --steps 2 --kmax 20 --p-mixed 0.3 --p-empty 0.15 --ids ws4_dv.txt || { log "FAIL data"; exit 1; }
PH=${1:-all}
if [ "$PH" = base ] || [ "$PH" = all ]; then
  CUDA_VISIBLE_DEVICES=7 python3 wcheck.py ws4 > wcheck_ws4.log 2>&1 || { log "FAIL wcheck, no training"; exit 1; }
  B6=$HB; for g in 0 1; do bash serve_sel.sh rtm-sel$g $g $((18341+g)) $B6; done; waitp 18341 18342 || { log "FAIL serve"; exit 1; }
  for n in ep150 ev150; do
    rm -f ctx_A0_$n.jsonl ctx_A0O_$n.jsonl
    K1_BUDGETS=1000,2000,4000,16000 python3 infer.py q_$n.json --pooldir pool_$n --port 18341,18342 --tag A0_$n --cond 0 --par 8 || { log "FAIL a0 $n"; exit 1; }
    python3 pool.py q_$n.json --rr-port 18341,18342 --rr-model sel --T 600 --par 8 --out pool_${n}_06 || { log "FAIL pool06 $n"; exit 1; }
    K1_BUDGETS=1000,2000,4000,16000 python3 cov.py rr8 q_$n.json A0O_$n --pooldir pool_${n}_06 || { log "FAIL a0o $n"; exit 1; }
  done
  [ -s ctx_A0O_H.jsonl ] && [ -s ctx_A0O_lmes150.jsonl ] || { log "FAIL a0o baselines"; exit 1; }
  rm -f ctx_A0_w14.jsonl ctx_A0M_w14.jsonl
  K1_BUDGETS=4000,16000 python3 infer.py env_ws4.json --ids ws4_dv.txt --pooldir pool_ws4 --port 18341,18342 --tag A0_w14 --cond 0 --par 8 || { log "FAIL a0 w14"; exit 1; }
  MASK_DATE=1 K1_BUDGETS=4000,16000 python3 infer.py env_ws4.json --ids ws4_dv.txt --pooldir pool_ws4 --port 18341,18342 --tag A0M_w14 --cond 0 --par 8 || { log "FAIL a0m w14"; exit 1; }
  rm -f ctx_A0M_lmes150.jsonl; MASK_DATE=1 K1_BUDGETS=1000,2000,4000 python3 infer.py q_lmes150.json --pooldir pool_lmes150 --port 18341,18342 --tag A0M_lmes150 --cond 0 --par 8 || { log "FAIL a0m lme"; exit 1; }
  docker rm -f rtm-sel0 rtm-sel1 >/dev/null 2>&1
  docker rm -f rtm-env-q2 >/dev/null 2>&1; docker run -d --name rtm-env-q2 --gpus '"device=2"' --ipc=host --network host -v /home/centuser/data/models/Qwen/Qwen3.8-27B-FP8:/model:ro $IMG /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port 18303 --enable-prefix-caching --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 256 >/dev/null
  waitp 18303 || { log "FAIL reader"; exit 1; }
  python3 fnchk.py env_ws4.json pool_ws4 ws4_tr.txt || { log "FAIL fnchk"; exit 1; }; docker rm -f rtm-env-q2 >/dev/null 2>&1
  python3 bud.py || { log "FAIL budgets"; exit 1; }; log BASEDONE
fi
[ "$PH" = base ] && exit 0
# all 8 GPUs (user 11-01): the embedding service leaves GPU 5 for training and evaluation, restarted at the end
docker stop rtm-env-emb >/dev/null 2>&1; trap 'docker rm -f rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 rtm-sel4 rtm-sel5 rtm-sel6 rtm-sel7 >/dev/null 2>&1; docker start rtm-env-emb >/dev/null 2>&1' EXIT
TRAIN w20 0,1,2,3 4 --model $B --data /w/tr_w20.jsonl --dev /w/dv_w20.jsonl --out /w/ckpt/w20 --lr 1e-5 --epochs 2 --eval-every 200 --max-d 1100 > w20.log 2>&1 & P1=$!
TRAIN w20c 4,5,6,7 4 --model $B --data /w/tr_w20c.jsonl --dev /w/dv_w20c.jsonl --out /w/ckpt/w20c --lr 1e-5 --epochs 2 --eval-every 200 --max-d 1100 --max-q 7000 > w20c.log 2>&1 & P2=$!
for p in $P1 $P2; do wait $p || { log "FAIL train"; exit 1; }; done; log "train done"
BE=$(python3 -c "import json;print(json.load(open('budgets.json'))['ep150'] or 1000)"); BV=$(python3 -c "import json;print(json.load(open('budgets.json'))['ev150'] or 1000)")
# W20C is run separately per main budget: its conditioning state must match the budget it is scored at
cb(){ q=$1; pd=$2; tag=$3; b=$4; p=$5; shift 5; K1_BUDGETS=$b python3 infer.py $q --pooldir $pd --port $p --tag $tag --cond 1 --per-step 4 --budget $b --par 8 "$@"; }
evc(){ p=$1; t=W20C
  cb env_ws4.json pool_ws4 ${t}_w14 4000 $p --ids ws4_dv.txt || return 1
  cb beam_ex.json pool ${t}_H16 16000 $p || return 1; cb beam_ex.json pool ${t}_H4 4000 $p || return 1; cat ctx_${t}_H16.jsonl ctx_${t}_H4.jsonl > ctx_${t}_H.jsonl
  cb q_lmes150.json pool_lmes150 ${t}_lmes150 2000 $p || return 1
  python3 infer.py beam_ex.json --port $p --tag W20CP_H --cond 0 --par 8 || return 1
  cb q_ep150.json pool_ep150 ${t}_ep150 $BE $p || return 1; cb q_ev150.json pool_ev150 ${t}_ev150 $BV $p || return 1; log "$t eval done"; }
evp(){ p=$1; t=W20; o="--cond 0"
  K1_BUDGETS=4000,16000 python3 infer.py env_ws4.json --ids ws4_dv.txt --pooldir pool_ws4 --port $p --tag ${t}_w14 $o --par 8 || return 1
  python3 infer.py beam_ex.json --port $p --tag ${t}_H $o --par 8 || return 1
  K1_BUDGETS=1000,2000,4000 python3 infer.py q_lmes150.json --pooldir pool_lmes150 --port $p --tag ${t}_lmes150 $o --par 8 || return 1
  MASK_DATE=1 python3 infer.py beam_ex.json --port $p --tag W20M_H $o --par 8 || return 1
  MASK_DATE=1 K1_BUDGETS=1000,2000,4000 python3 infer.py q_lmes150.json --pooldir pool_lmes150 --port $p --tag W20M_lmes150 $o --par 8 || return 1
  MASK_DATE=1 K1_BUDGETS=4000,16000 python3 infer.py env_ws4.json --ids ws4_dv.txt --pooldir pool_ws4 --port $p --tag W20M_w14 $o --par 8 || return 1
  K1_BUDGETS=$BE python3 infer.py q_ep150.json --pooldir pool_ep150 --port $p --tag ${t}_ep150 $o --par 8 || return 1
  K1_BUDGETS=$BV python3 infer.py q_ev150.json --pooldir pool_ev150 --port $p --tag ${t}_ev150 $o --par 8 || return 1; log "$t eval done"; }
rm -f ctx_W20_*.jsonl ctx_W20C_*.jsonl ctx_W20M_*.jsonl
s(){ bash serve_sel.sh rtm-sel$1 $1 $2 $3; }
s 0 18341 $C/w20; s 1 18342 $C/w20; s 2 18343 $C/w20; s 3 18344 $C/w20c; s 4 18345 $C/w20c; s 5 18346 $C/w20c; s 6 18347 $C/w20c; s 7 18348 $C/w20c; waitp 18341 18342 18343 18344 18345 18346 18347 18348 || { log "FAIL serve"; exit 1; }
evp 18341,18342,18343 & P1=$!
evc 18344,18345,18346,18347,18348 & P2=$!
for p in $P1 $P2; do wait $p || { log "FAIL eval"; exit 1; }; done
python3 ssum.py W20 W20C > ssum.log 2>&1 || { log "FAIL ssum"; exit 1; }
log S2DONE
