#!/bin/bash
# D4 seed 0: wait for full FN judging -> verdict -> four arms (frozen table) -> train (2 GPUs each) -> H4 eval of every checkpoint -> summary.
# Stops itself at 17:30 ET; d4/release.sh frees the GPUs at 17:38.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
SEED=${SEED:-0}; SUF=$([ "$SEED" = 0 ] && echo "" || echo "_s$SEED")
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; B=/models/qwen/Qwen3-Reranker-0.6B
late(){ [ "$(TZ=America/New_York date +%H%M)" -ge 1730 ]; }
trap 'docker ps --format "{{.Names}}" | grep -E "^rtm-d4t-|^rtm-d4e-" | xargs -r docker rm -f >/dev/null 2>&1' EXIT
if [ "$SEED" = 0 ]; then
  until grep -q FNALLEXIT d4/fnall.log; do late && { log "LATE waiting for judge"; exit 1; }; sleep 60; done
  grep -q "FNALLEXIT 0" d4/fnall.log || { log "FAIL judge: $(tail -2 d4/fnall.log)"; exit 1; }
  python3 fnfilter.py verdict --pairs d4/cands.json --verdicts d4/fn_v.jsonl --out d4/fn_support.json || { log FAIL verdict; exit 1; }
  log "verdict $(python3 -c 'import json;print(json.load(open("d4/fn_support.json"))["counts"])')"
  docker ps --format "{{.Names}}" | grep "^rtm-d4-" | xargs -r docker rm -f >/dev/null
  cd d4dev; ln -sf ../d4/cands.json cands.json; ln -sf ../d4/fn_support.json fn_support.json
  python3 mkdata.py env_ws4.json pool_ws4 C.jsonl --group-table D4_FREEZE.json --arm C --inst v1 --cands cands.json --shared-fn-support fn_support.json > mk_C.log 2>&1 &
  python3 mkdata.py env_ws4.json pool_ws4 I.jsonl --group-table D4_FREEZE.json --arm I --inst v2 --cands cands.json --shared-fn-support fn_support.json > mk_I.log 2>&1 &
  python3 mkdata.py env_ws4.json pool_ws4 P0.jsonl --group-table D4_FREEZE.json --arm P0 --inst v2 --cands cands.json --fn-support fn_support.json --qtext qtext.json > mk_P0.log 2>&1 &
  python3 mkdata.py env_ws4.json pool_ws4 P0P1.jsonl --group-table D4_FREEZE.json --arm P0P1 --inst v2 --cands cands.json --fn-support fn_support.json --qtext qtext.json --neg-mix > mk_P0P1.log 2>&1 &
  wait; for a in C I P0 P0P1; do [ -s $a.jsonl ] || { log "FAIL data $a: $(tail -3 mk_$a.log)"; exit 1; }; done
  python3 - <<'PY' || { cd ..; log "FAIL arms differ"; exit 1; }
import json, hashlib
R = {a: json.load(open(f"{a}.jsonl.drops.json")) for a in ("C", "I", "P0", "P0P1")}
h = {a: hashlib.md5(json.dumps(r["selected"], sort_keys=True).encode()).hexdigest() for a, r in R.items()}
n = {a: sum(1 for _ in open(f"{a}.jsonl")) for a in R}; print("selected md5", h, "rows", n)
assert len(set(h.values())) == 1 and len(set(n.values())) == 1
PY
  cd ..; log "data ready $(wc -l < d4dev/C.jsonl) groups per arm"
fi
TRAIN(){ name=$1; g=$2; shift 2; timeout 14400 docker run --rm --name rtm-d4t-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node 2 --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
TR="--model $B --lr 1e-5 --max-d 1100 --accum 8 --save-every 100 --seed $SEED"
TRAIN C$SUF 0,1 $TR --epochs 1 --data /w/d4dev/C.jsonl --out /w/ckpt/d4_C$SUF > d4/tr_C$SUF.log 2>&1 & p1=$!
TRAIN P0$SUF 4,5 $TR --epochs 1 --data /w/d4dev/P0.jsonl --out /w/ckpt/d4_P0$SUF > d4/tr_P0$SUF.log 2>&1 & p3=$!
TRAIN P0P1$SUF 6,7 $TR --epochs 1 --data /w/d4dev/P0P1.jsonl --out /w/ckpt/d4_P0P1$SUF > d4/tr_P0P1$SUF.log 2>&1 & p4=$!
p2=""; [ "$SEED" = 0 ] && { TRAIN I 2,3 $TR --epochs 0.5 --data /w/d4dev/I.jsonl --out /w/ckpt/d4_I > d4/tr_I.log 2>&1 & p2=$!; }
for p in $p1 $p2 $p3 $p4; do wait $p || { log "FAIL train (pid $p)"; exit 1; }; done; log "trained seed $SEED"
late && { log "LATE after training"; exit 1; }
# eval: every checkpoint on H4 (all 604 questions), own instruction (C v1, others v2)
JOBS=(); for a in C P0 P0P1; do for s in 100 200 300 400 500 600; do JOBS+=("$a$SUF:$s:$([ $a = C ] && echo 1 || echo 2)"); done; done
[ "$SEED" = 0 ] && for s in 100 200 300; do JOBS+=("I:$s:2"); done
ev(){ g=$1; shift; for j in "$@"; do IFS=: read a s v <<< "$j"; d=/home/centuser/data/rtm/sel/ckpt/d4_$a/s$s; [ -d $d ] || d=/home/centuser/data/rtm/sel/ckpt/d4_$a
  bash serve_sel.sh rtm-d4e-$g $g $((18370+g)) $d >/dev/null; ok=0; for k in $(seq 60); do curl -s -m 3 localhost:$((18370+g))/v1/models >/dev/null && { ok=1; break; }; sleep 10; done
  [ $ok = 1 ] || { log "FAIL serve $a $s"; return 1; }; rm -f ctx_D4${a}s${s}_h4.jsonl
  FMT_INST=$v K1_BUDGETS=1000,2000,4000,8000,16000 python3 infer.py env_h4.json --pooldir pool_h4 --port $((18370+g)) --tag D4${a}s${s}_h4 --cond 0 --par 16 > d4/inf_${a}s${s}.log 2>&1 || { log "FAIL infer $a $s"; return 1; }
  docker rm -f rtm-d4e-$g >/dev/null; log "eval $a s$s"; done; }
P=(); for g in 0 1 2 3 4 5 6 7; do sub=(); for ((i=g; i<${#JOBS[@]}; i+=8)); do sub+=("${JOBS[$i]}"); done; ev $g "${sub[@]}" & P+=($!); done
X=0; for p in "${P[@]}"; do wait $p || X=1; done; [ $X = 0 ] || { log "FAIL eval"; exit 1; }
A="C:D4C$SUF:100,200,300,400,500,600 P0:D4P0$SUF:100,200,300,400,500,600 P0P1:D4P0P1$SUF:100,200,300,400,500,600"; [ "$SEED" = 0 ] && A="$A I:D4I:100,200,300"
python3 d4/d4eval.py $A > d4/sum$SUF.txt 2>&1; cat d4/sum$SUF.txt; log "D4RUNDONE seed $SEED"
