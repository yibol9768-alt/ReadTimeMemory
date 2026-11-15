#!/bin/bash
# D3 v9 (spec v8, v9): waits for the worlds and pools, builds data, pre-training gates, trains E and fixed (3 seeds), control (3 seeds), hard-skip 0 pilot. Evaluation is d3v9e.sh.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
trap 'docker rm -f rtm-sel6 rtm-sel7 rtm-train-e0 rtm-train-e1 rtm-train-e2 rtm-train-f0 rtm-train-f1 rtm-train-f2 rtm-train-p0 rtm-train-p1 rtm-train-p2 rtm-train-h0 >/dev/null 2>&1' EXIT
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 3000 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
up(){ for k in $(seq 80); do ok=1; for p in "$@"; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
for t in ep150 ev150; do for a in W20 D3RF D3RE D3RH; do [ -s ctx_${a}X_$t.jsonl ] || cp ctx_${a}_$t.jsonl ctx_${a}X_$t.jsonl || exit 1; done; done   # single-budget files, unaffected by the packing bug
until grep -q D3V8WDONE d3v8w.log; do grep -q FAIL d3v8w.log && { log "world step failed"; exit 1; }; sleep 20; done; log worlds and pools ready
HS=$(cat hs.txt 2>/dev/null || echo 10); MK="--no-set --ignore-kinds repeat,fu --hard-skip $HS --boost 3 --steps 8"
python3 mkdata.py env_ws7x.json pool_ws7x v9_all_fix.jsonl $MK > mk_v9f.log 2>&1 && python3 mkdata.py env_ws7e.json pool_ws7e v9_all_e.jsonl $MK > mk_v9e.log 2>&1 || { log FAIL mk; exit 1; }
python3 -c "import json; open('ws7ct_aggmix.txt','w').write('\n'.join(q['question_id'] for q in json.load(open('env_ws7ct.json')) if q['question_type']=='aggregation' and len(q.get('carriers',[]))>1))"
python3 mkdata.py env_ws7ct.json pool_ws7ct v9_all_pc.jsonl $MK --ids ws7ct_aggmix.txt > mk_v9p.log 2>&1 || { log FAIL mkpc; exit 1; }
python3 mkdata.py env_ws6x.json pool_ws6x v9_all_hs0.jsonl --no-set --ignore-kinds repeat,fu --hard-skip 0 --boost 3 --steps 8 > mk_v9h.log 2>&1 || { log FAIL mkh; exit 1; }
NF=$(wc -l < v9_all_fix.jsonl); N=$(( NF / 2 / 100 * 100 )); [ $N -gt 2000 ] && N=2000; log "groups fix $NF e $(wc -l < v9_all_e.jsonl) -> N $N pc $(wc -l < v9_all_pc.jsonl)"; [ $N -ge 100 ] || { log "NOT EXECUTABLE N"; exit 4; }
python3 subs.py v9_all_fix.jsonl tr_v9_fix.jsonl $N && python3 subq.py v9_all_e.jsonl tr_v9_fix.jsonl tr_v9_e.jsonl || { log "NOT EXECUTABLE quota"; exit 4; }
python3 subs.py v9_all_pc.jsonl tr_v9_pc.jsonl 375 || { log "NOT EXECUTABLE pc"; exit 4; }
python3 subs.py v9_all_hs0.jsonl tr_v9_hs0.jsonl 1500 || { log "NOT EXECUTABLE hs0"; exit 4; }
python3 gatev9.py > d3v9_gate.txt 2>&1; g=$?; cat d3v9_gate.txt; [ $g = 0 ] || { log "GATE FAIL: stop before main training"; exit 3; }
STEPS=$(( N / 2 / 4 )); SA="$(( STEPS / 4 )),$(( STEPS / 2 ))"; log "steps $STEPS save at $SA"
base(){ echo "--model /w/ckpt/w20 --lr 1e-5 --epochs 1 --max-d 1100 --w-base 0.5 --seed $1"; }
CK=/home/centuser/data/rtm/sel/ckpt
# GPUs 6,7: W20 on the control eval worlds first (the control needs its pre gate), then control seeds, then hard-skip 0
( for g in 6 7; do bash serve_sel.sh rtm-sel$g $g $((18341+g)) $CK/w20; done; up 18347 18348 || exit 1
  K1_BUDGETS=1000,2000,4000 python3 infer.py env_ws7ce.json --pooldir pool_ws7ce --port 18347,18348 --tag W20_ws7ce --cond 0 --par 16 > inf_W20_ws7ce.log 2>&1 || exit 1
  docker rm -f rtm-sel6 rtm-sel7 >/dev/null 2>&1; sleep 5; rm -f v9_pre_ok; python3 pcgate3.py pre W20_ws7ce > d3v9_pre.txt 2>&1 && touch v9_pre_ok; cat d3v9_pre.txt
  if [ -f v9_pre_ok ]; then for s in 0 1 2; do TRAIN p$s 6,7 2 $(base $s) --data /w/tr_v9_pc.jsonl --out /w/ckpt/v9_p$s > v9_tr_p$s.log 2>&1 || exit 1; done; else log "control invalid: pre gate failed, control skipped"; fi
  TRAIN h0 6,7 2 $(base 0) --data /w/tr_v9_hs0.jsonl --out /w/ckpt/v9_h0 > v9_tr_h0.log 2>&1 ) & Q=$!
TRAIN e0 0,1 2 $(base 0) --save-at $SA --data /w/tr_v9_e.jsonl --out /w/ckpt/v9_e0 > v9_tr_e0.log 2>&1 & a=$!
TRAIN f0 2,3 2 $(base 0) --save-at $SA --data /w/tr_v9_fix.jsonl --out /w/ckpt/v9_f0 > v9_tr_f0.log 2>&1 & b=$!
TRAIN e1 4,5 2 $(base 1) --save-at $SA --data /w/tr_v9_e.jsonl --out /w/ckpt/v9_e1 > v9_tr_e1.log 2>&1 & c=$!
wait $a && wait $b && wait $c || { log FAIL wave1; exit 1; }; log wave1 done
TRAIN f1 0,1 2 $(base 1) --save-at $SA --data /w/tr_v9_fix.jsonl --out /w/ckpt/v9_f1 > v9_tr_f1.log 2>&1 & a=$!
TRAIN e2 2,3 2 $(base 2) --save-at $SA --data /w/tr_v9_e.jsonl --out /w/ckpt/v9_e2 > v9_tr_e2.log 2>&1 & b=$!
TRAIN f2 4,5 2 $(base 2) --save-at $SA --data /w/tr_v9_fix.jsonl --out /w/ckpt/v9_f2 > v9_tr_f2.log 2>&1 & c=$!
wait $a && wait $b && wait $c || { log FAIL wave2; exit 1; }; log wave2 done
wait $Q || { log FAIL control queue; exit 1; }
log D3V9TRAINDONE
sleep 10; [ -z "$(docker ps --format "{{.Names}}" | grep -E "rtm-(train|sel|ev)")" ] || { log "containers still up, no eval"; exit 1; }
bash d3v9e.sh > d3v9e.log 2>&1; r=$?; tail -30 d3v9e.log; log "eval code $r"
exit $r
