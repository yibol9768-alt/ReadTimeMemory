#!/bin/bash
# Transfer tests (10-31, decided after REVIEW_SERIES; audit fixes applied). GPUs 6,7 only.
#  A0  untrained Qwen3-Reranker-0.6B, fmt prompt;  A0O untrained, official prompt
#  T1  pointwise, trained on BEAM-100K (upper-bound test);  T2 pointwise, trained on existing synthetic data only (transfer)
# Eval: BEAM-500K dev, 299 ids, P=600. Success rule: PROGRESS 10-31 (txsum.py). QA launched separately after recall.
set -eu
cd ~/data/rtm/sel
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
TRAIN(){ name=$1; shift; timeout 14400 docker run --rm --name rtm-train-$name --gpus '"device=6,7"' --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node 2 --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
SERVE(){ bash serve_sel.sh rtm-sel6 6 18331 $1; bash serve_sel.sh rtm-sel7 7 18332 $1; sleep 20; for k in $(seq 80); do curl -s -m 3 localhost:18331/v1/models >/dev/null && curl -s -m 3 localhost:18332/v1/models >/dev/null && return 0; sleep 15; done; log "serve timeout"; exit 1; }
UNSERVE(){ docker rm -f rtm-sel6 rtm-sel7 >/dev/null 2>&1 || true; }
CONSIST(){ docker run --rm --name rtm-consist --gpus '"device=6"' --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint python3 $IMG /w/consist2.py $1 /w/$2 18332; }
trap 'docker rm -f rtm-sel6 rtm-sel7 rtm-env-rr rtm-env-rr2 rtm-consist >/dev/null 2>&1 || true' EXIT
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
TRAIN tiny --model /models/qwen/Qwen3-Reranker-0.6B --data /w/tiny_tr.jsonl --dev /w/tiny_dv.jsonl --out /w/ckpt/tiny --lr 1e-5 --epochs 1 --accum 2 --max-d 1100
SERVE /home/centuser/data/rtm/sel/ckpt/tiny
rm -f ctx_TINY_H.jsonl; python3 infer.py beam_ex.json --port 18331,18332 --tag TINY_H --cond 0 --limit 5 --par 4
UNSERVE
python3 -c "
import json
r=[json.loads(l) for l in open('ctx_TINY_H.jsonl')]; print('tiny rows',len(r),[ (x['qid'],x['budget'],x['gold_in'],x['gold_total']) for x in r[:4]])"
# --- baselines + HF/vLLM consistency
log "baselines"
SERVE $B06
docker rm -f rtm-sel6 >/dev/null 2>&1
CONSIST /models/qwen/Qwen3-Reranker-0.6B dv_b100.jsonl
SERVE $B06
python3 infer.py beam_ex.json --port 18331,18332 --tag A0_H --cond 0 --par 8
python3 pool.py beam_ex.json --rr-port 18331,18332 --rr-model sel --T 600 --par 8 --out pool_h06
python3 cov.py rr8 beam_ex.json A0O_H --pooldir pool_h06
UNSERVE
# --- T1 first: if it shows a problem we save the T2 time
log "train T1"
TRAIN t1 --model /models/qwen/Qwen3-Reranker-0.6B --data /w/tr_b100.jsonl --dev /w/dv_b100.jsonl --out /w/ckpt/t1 --lr 1e-5 --epochs 2 --eval-every 100 --max-d 1100
SERVE /home/centuser/data/rtm/sel/ckpt/t1
docker rm -f rtm-sel6 >/dev/null 2>&1
CONSIST /w/ckpt/t1 dv_b100.jsonl
SERVE /home/centuser/data/rtm/sel/ckpt/t1
python3 infer.py beam_ex.json --port 18331,18332 --tag T1_H --cond 0 --par 8
UNSERVE
python3 txsum.py T1_H
echo T1DONE
log "train T2"
TRAIN t2 --model /models/qwen/Qwen3-Reranker-0.6B --data /w/tr_env.jsonl --dev /w/dv_env.jsonl --out /w/ckpt/t2 --lr 1e-5 --epochs 2 --eval-every 100 --max-d 1100
SERVE /home/centuser/data/rtm/sel/ckpt/t2
python3 infer.py beam_ex.json --port 18331,18332 --tag T2_H --cond 0 --par 8
UNSERVE
python3 txsum.py T1_H,T2_H
echo TXDONE
