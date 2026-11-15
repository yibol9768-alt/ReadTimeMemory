#!/bin/bash
# S1b (10-31): first world-trained selector. 4 worlds train, world 14 held out. Question-free arms measure non-linear shortcuts.
#  W1 fresh 0.6B on worlds 10-13; NOQW same data with the question hidden; NOQB BEAM-100K data with the question hidden.
set -u
cd ~/data/rtm/sel
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
B=/models/qwen/Qwen3-Reranker-0.6B; HB=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B; C=/home/centuser/data/rtm/sel/ckpt
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 14400 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
waitp(){ sleep 20; for k in $(seq 80); do ok=1; for p in "$@"; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
trap 'docker rm -f rtm-sel3 rtm-sel4 rtm-sel7 >/dev/null 2>&1' EXIT
K="--no-set --ignore-kinds repeat,fu --hard-skip 10 --boost 3"
python3 mkdata.py env_ws1.json pool_ws1 tr_w1.jsonl $K --steps 8 --ids ws1_tr.txt || { log "FAIL data"; exit 1; }
python3 mkdata.py env_ws1.json pool_ws1 dv_w1.jsonl $K --steps 2 --ids ws1_dv.txt || { log "FAIL data"; exit 1; }
python3 - <<PY
import json, re
for a, b in (("tr_w1", "trq_w1"), ("dv_w1", "dvq_w1"), ("tr_b100", "trq_b100"), ("dv_b100", "dvq_b100")):
    with open(b + ".jsonl", "w") as f:
        for l in open(a + ".jsonl"):
            r = json.loads(l); r["query"] = re.sub(r"<Query>: .*?\n<Selected>", "<Query>: (hidden)\n<Selected>", r["query"], flags=re.S); r["query"] = re.sub(r"\(asked on [^)]*\)", "(asked later)", r["query"]); f.write(json.dumps(r) + "\n")
    Q = [json.loads(l)["query"] for l in open(b + ".jsonl")]
    assert not any("(asked on" in q or "<Query>: (hidden)" not in q for q in Q), b
PY
[ $? = 0 ] || { log "FAIL hide"; exit 1; }
if [ "${1:-}" = tiny ]; then
  head -48 tr_w1.jsonl > tiny_w.jsonl; head -8 dv_w1.jsonl > tiny_wd.jsonl
  TRAIN w1t 3 1 --model $B --data /w/tiny_w.jsonl --dev /w/tiny_wd.jsonl --out /w/ckpt/w1t --lr 1e-5 --epochs 1 --eval-every 4 --max-d 1100 || { log "FAIL tiny train"; exit 1; }
  bash serve_sel.sh rtm-sel3 3 18343 $C/w1t; waitp 18343 || { log "FAIL tiny serve"; exit 1; }
  K1_BUDGETS=4000,16000 python3 infer.py env_ws1.json --ids ws1_dv.txt --pooldir pool_ws1 --port 18343 --tag W1T_w14 --cond 0 --par 4 --limit 3 || { log "FAIL tiny infer"; exit 1; }
  wc -l tr_w1.jsonl dv_w1.jsonl; log TINYDONE; exit 0
fi
TRAIN noqw 0,1 2 --model $B --data /w/trq_w1.jsonl --dev /w/dvq_w1.jsonl --out /w/ckpt/noqw --lr 1e-5 --epochs 1 --eval-every 100 --max-d 1100 > noqw.log 2>&1 & P1=$!
TRAIN noqb 2,6 2 --model $B --data /w/trq_b100.jsonl --dev /w/dvq_b100.jsonl --out /w/ckpt/noqb --lr 1e-5 --epochs 1 --eval-every 100 --max-d 1100 > noqb.log 2>&1 & P2=$!
TRAIN w1 3,4 2 --model $B --data /w/tr_w1.jsonl --dev /w/dv_w1.jsonl --out /w/ckpt/w1 --lr 1e-5 --epochs 2 --eval-every 100 --max-d 1100 > w1.log 2>&1 & P3=$!
for p in $P1 $P2 $P3; do wait $p || { log "FAIL train"; exit 1; }; done
log "train done"
bash serve_sel.sh rtm-sel3 3 18343 $C/w1; bash serve_sel.sh rtm-sel4 4 18344 $C/w1; bash serve_sel.sh rtm-sel7 7 18347 $HB; waitp 18343 18344 18347 || { log "FAIL serve"; exit 1; }
rm -f ctx_W1_w14.jsonl ctx_A0_w14.jsonl ctx_W1M_w14.jsonl ctx_A0M_w14.jsonl ctx_W1_H.jsonl ctx_W1_lmes150.jsonl
export K1_BUDGETS=4000,16000
I="python3 infer.py env_ws1.json --ids ws1_dv.txt --pooldir pool_ws1 --cond 0"
$I --port 18343,18344 --tag W1_w14 --par 8 & P1=$!
$I --port 18347 --tag A0_w14 --par 4 & P2=$!
for p in $P1 $P2; do wait $p || { log "FAIL w14"; exit 1; }; done
MASK_DATE=1 $I --port 18343,18344 --tag W1M_w14 --par 8 & P1=$!
MASK_DATE=1 $I --port 18347 --tag A0M_w14 --par 4 & P2=$!
for p in $P1 $P2; do wait $p || { log "FAIL w14 mask"; exit 1; }; done
python3 infer.py beam_ex.json --port 18343,18344 --tag W1_H --cond 0 --par 8 || { log "FAIL beam"; exit 1; }
K1_BUDGETS=1000,2000,4000 python3 infer.py q_lmes150.json --pooldir pool_lmes150 --port 18343,18344 --tag W1_lmes150 --cond 0 --par 8 || { log "FAIL lme"; exit 1; }
log S1BDONE
