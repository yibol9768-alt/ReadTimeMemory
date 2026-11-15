#!/bin/bash
# Transfer tests (10-31, decided after REVIEW_SERIES; audit fixes applied). Parallel arms; 27B readers on GPUs 0-4 are stopped first, restart them for QA.
#  A0  untrained Qwen3-Reranker-0.6B, fmt prompt;  A0O untrained, official prompt
#  T1  pointwise, trained on BEAM-100K (upper-bound test);  T2 pointwise, trained on existing synthetic data only (transfer)
# Eval: BEAM-500K dev, 299 ids, P=600. Success rule: PROGRESS 10-31 (txsum.py). QA launched separately after recall.
set -eu
cd ~/data/rtm/sel
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 14400 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
# SERVE MODEL G1 G2 P1 P2 ; UNSERVE G1 G2
SERVE(){ bash serve_sel.sh rtm-sel$2 $2 $4 $1; bash serve_sel.sh rtm-sel$3 $3 $5 $1; sleep 20; for k in $(seq 80); do curl -s -m 3 localhost:$4/v1/models >/dev/null && curl -s -m 3 localhost:$5/v1/models >/dev/null && return 0; sleep 15; done; log "serve timeout"; exit 1; }
UNSERVE(){ docker rm -f rtm-sel$1 rtm-sel$2 >/dev/null 2>&1 || true; }
# CONSIST MODEL GROUPS GPU PORT (model served on PORT on another GPU)
CONSIST(){ docker run --rm --name rtm-consist$3 --gpus "\"device=$3\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint python3 $IMG /w/consist2.py $1 /w/$2 $4; }
trap 'docker rm -f rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 rtm-sel6 rtm-sel7 rtm-consist0 rtm-consist2 rtm-consist6 >/dev/null 2>&1 || true' EXIT
until grep -q VARDONE ~/data/rtm/env/run_var.log; do pgrep -f "bash run_var.sh" >/dev/null || { grep -q VARDONE ~/data/rtm/env/run_var.log || { log "run_var died"; exit 1; }; }; sleep 30; done
log "var done"
for f in env_v3p.json env_v4p.json env_var_V0.json env_var_VA.json env_var_VB.json env_var_VC.json; do [ -s $f ] || { log "missing $f"; exit 1; }; done
~/data/rtm/alfenv/bin/python ~/data/rtm/beam/beam2lme.py ~/data/rtm/beam/beam/data/100K-00000-of-00001.parquet all beam100k_g.json c
bash ~/data/rtm/env/up_rr.sh
for k in $(seq 80); do curl -s -m 3 localhost:18312/v1/models >/dev/null && curl -s -m 3 localhost:18313/v1/models >/dev/null && break; sleep 15; done
python3 pool.py beam100k_g.json --rr-port 18312,18313 --T 600 --par 8 --out pool_c
docker rm -f rtm-env-rr rtm-env-rr2 >/dev/null 2>&1
python3 - <<PY
import json
q=json.load(open("beam100k_g.json")); dev={"c19","c20"}
open("c_tr.txt","w").write("\n".join(x["question_id"] for x in q if x["question_id"].split("_")[0] not in dev))
open("c_dv.txt","w").write("\n".join(x["question_id"] for x in q if x["question_id"].split("_")[0] in dev))
k=set(open('ex_ids.txt').read().split()); json.dump([x for x in json.load(open('beam500k_g.json')) if x['question_id'] in k], open('beam_ex.json','w'))
PY
python3 mkdata.py beam100k_g.json pool_c tr_b100.jsonl --no-set --gold-rounds --hard-skip 10 --steps 8 --boost 3 --ids c_tr.txt
python3 mkdata.py beam100k_g.json pool_c dv_b100.jsonl --no-set --gold-rounds --hard-skip 10 --steps 4 --ids c_dv.txt
: > tr_env_all.jsonl
for x in "env_v3p.json pool_v3p" "env_v4p.json pool_v4p" "env_var_V0.json pool_var_V0" "env_var_VA.json pool_var_VA" "env_var_VB.json pool_var_VB" "env_var_VC.json pool_var_VC"; do set -- $x
  python3 mkdata.py $1 $2 tmp_e.jsonl --no-set --hard-skip 10 --split-not test --steps 8 --boost 3
  cat tmp_e.jsonl >> tr_env_all.jsonl
done
python3 - <<PY
import json
r=[l for l in open("tr_env_all.jsonl")]; ids=sorted({json.loads(l)["qid"].split("_")[0] for l in r})
dv=set(ids[::5]); tr=[l for l in r if json.loads(l)["qid"].split("_")[0] not in dv]; d=[l for l in r if json.loads(l)["qid"].split("_")[0] in dv]
open("tr_env.jsonl","w").writelines(tr); open("dv_env.jsonl","w").writelines(d); print("env train groups",len(tr),"dev",len(d),"worlds",ids)
PY
wc -l tr_b100.jsonl dv_b100.jsonl tr_env.jsonl dv_env.jsonl
export K1_BUDGETS=4000,16000
B06=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
# --- end-to-end tiny run (rule: before scaling)
log "tiny e2e"
head -24 tr_b100.jsonl > tiny_tr.jsonl; head -8 dv_b100.jsonl > tiny_dv.jsonl
TRAIN tiny 6,7 2 --model /models/qwen/Qwen3-Reranker-0.6B --data /w/tiny_tr.jsonl --dev /w/tiny_dv.jsonl --out /w/ckpt/tiny --lr 1e-5 --epochs 1 --accum 2 --max-d 1100
SERVE /home/centuser/data/rtm/sel/ckpt/tiny 6 7 18331 18332
rm -f ctx_TINY_H.jsonl; python3 infer.py beam_ex.json --port 18331,18332 --tag TINY_H --cond 0 --limit 5 --par 4
UNSERVE 6 7
python3 -c "
import json
r=[json.loads(l) for l in open('ctx_TINY_H.jsonl')]; print('tiny rows',len(r),[ (x['qid'],x['budget'],x['gold_in'],x['gold_total']) for x in r[:4]])"
# --- three parallel arms (GPU 6,7 baselines; 0,1 T1; 2,3,4 T2)
log "tiny ok; waiting for ARMS_GO (audit gate)"
until [ -f ARMS_GO ]; do sleep 20; done
docker rm -f rtm-qa-0 rtm-qa-1 rtm-qa-2 rtm-qa-3 rtm-qa-4 >/dev/null 2>&1 || true
sleep 10
arm_base(){
  log "baselines"
  SERVE $B06 6 7 18331 18332
  docker rm -f rtm-sel6 >/dev/null 2>&1
  CONSIST /models/qwen/Qwen3-Reranker-0.6B dv_b100.jsonl 6 18332
  SERVE $B06 6 7 18331 18332
  python3 infer.py beam_ex.json --port 18331,18332 --tag A0_H --cond 0 --par 8
  python3 pool.py beam_ex.json --rr-port 18331,18332 --rr-model sel --T 600 --par 8 --out pool_h06
  python3 cov.py rr8 beam_ex.json A0O_H --pooldir pool_h06
  UNSERVE 6 7; log "baselines done"; echo BASEDONE; }
arm_t1(){
  log "train T1"
  TRAIN t1 0,1 2 --model /models/qwen/Qwen3-Reranker-0.6B --data /w/tr_b100.jsonl --dev /w/dv_b100.jsonl --out /w/ckpt/t1 --lr 1e-5 --epochs 2 --eval-every 100 --max-d 1100
  SERVE /home/centuser/data/rtm/sel/ckpt/t1 0 1 18341 18342
  docker rm -f rtm-sel0 >/dev/null 2>&1
  CONSIST /w/ckpt/t1 dv_b100.jsonl 0 18342
  SERVE /home/centuser/data/rtm/sel/ckpt/t1 0 1 18341 18342
  python3 infer.py beam_ex.json --port 18341,18342 --tag T1_H --cond 0 --par 8
  UNSERVE 0 1; log "T1 eval done"; echo T1DONE; }
arm_t2(){
  log "train T2"
  TRAIN t2 2,3,4 3 --model /models/qwen/Qwen3-Reranker-0.6B --data /w/tr_env.jsonl --dev /w/dv_env.jsonl --out /w/ckpt/t2 --lr 1e-5 --epochs 2 --eval-every 100 --max-d 1100
  SERVE /home/centuser/data/rtm/sel/ckpt/t2 2 3 18343 18344
  python3 infer.py beam_ex.json --port 18343,18344 --tag T2_H --cond 0 --par 8
  UNSERVE 2 3; log "T2 eval done"; echo T2DONE; }
arm_base > tx_base.log 2>&1 &
PB=$!
arm_t1 > tx_t1.log 2>&1 &
P1=$!
arm_t2 > tx_t2.log 2>&1 &
P2=$!
wait $PB || log "BASE ARM FAILED"
wait $P1 || log "T1 ARM FAILED"
wait $P2 || log "T2 ARM FAILED"
python3 txsum.py T1_H,T2_H
echo TXDONE
