#!/bin/bash
# D3 stage 2b: E-arm worlds (same seeds as the fixed-mix worlds, knobs from d3_map.json), training data, equal N, hard mining, four arms, evaluation.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
trap 'docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o rtm-env-rr rtm-env-rr2 rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 rtm-sel4 rtm-sel5 rtm-sel6 rtm-sel7 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1' EXIT
TRAIN(){ name=$1; g=$2; np=$3; shift 3; timeout 7200 docker run --rm --name rtm-train-$name --gpus "\"device=$g\"" --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node $np --master_port $((29600+RANDOM%300)) /w/train.py "$@"; }
up(){ for k in $(seq 80); do ok=1; for p in "$@"; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
K=$(python3 -c "import json; print(json.dumps(json.load(open('d3_map.json'))['knobs']))"); WT=$(python3 -c "import json; print(json.load(open('d3_map.json'))['weights'])")
log "E knobs $K weights $WT"
# 1. E worlds: same plan seeds 972-975 as the fixed-mix worlds, other directory, only the knobs differ
bash wup.sh | tail -1 | grep -q UPOK || { log FAIL up; exit 1; }
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303,Qwen3.8-27B-FP8@18304,Qwen3.8-27B-FP8@18305 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
mkdir -p ws6e; for i in 972 973 974 975; do python3 reknob.py ws6/w00$i.json ws6e/w00$i.json "$K" || exit 1; done
one(){ i=$1; for s in render verify build gates; do python3 world.py $s --out ws6e --start $i --n 1 --rounds 240 --par 24 --knobs "$K" $A > ws6e/$i.$s.log 2>&1 || { log "FAIL e $i $s"; return 1; }; done; }
M0='{"p_group": 0.3, "p_long": 0.3, "p_dated": 0.15, "p_greet": 0.1, "p_sib": 0.3}'
two(){ i=$1; for s in plan render verify build gates; do python3 world.py $s --out ws6 --start $i --n 1 --rounds 240 --par 24 --knobs "$M0" $A > ws6/$i.$s.log 2>&1 || { log "FAIL pc $i $s"; return 1; }; done; }   # extra control worlds 978 (train) 979 (eval), spec v5
P=""; for i in 972 973 974 975; do one $i & P="$P $!"; done; two 978 & P="$P $!"; two 979 & P="$P $!"; for p in $P; do wait $p || exit 1; done
python3 wfilter.py ws6 976,978 0 ws6pc0 && python3 wfilter.py ws6 977,979 0 ws6pc1 || exit 1
docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o >/dev/null 2>&1
python3 wfilter.py ws6e 972,973,974,975 0 ws6e || exit 1
docker start rtm-env-emb >/dev/null; bash ~/data/rtm/env/up_rr.sh; up 18311 18312 18313 || { log FAIL rr; exit 1; }
for t in ws6e ws6pc0 ws6pc1; do python3 pool.py env_$t.json --rr-port 18312,18313 --T 600 --par 8 --out pool_$t > pool_$t.log 2>&1 || { log "FAIL pool $t"; exit 1; }; done
docker rm -f rtm-env-rr rtm-env-rr2 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null; log "E worlds ready"
# 2. training groups, same recipe as W20
HS=$(cat hs.txt 2>/dev/null || echo 10); MK="--no-set --ignore-kinds repeat,fu --hard-skip $HS --boost 3 --steps 8"
python3 mkdata.py env_ws6x.json pool_ws6x all_fix.jsonl $MK > mk_fix.log 2>&1 && python3 mkdata.py env_ws6e.json pool_ws6e all_e.jsonl $MK > mk_e.log 2>&1 || { log FAIL mkdata; exit 1; }
python3 -c "import json; open('ws6pc0_group.txt','w').write('\n'.join(q['question_id'] for q in json.load(open('env_ws6pc0.json')) if q.get('carriers') == ['group']))"
python3 mkdata.py env_ws6pc0.json pool_ws6pc0 all_pc.jsonl $MK --ids ws6pc0_group.txt > mk_pc.log 2>&1 || { log FAIL mkpc; exit 1; }
NF=$(wc -l < all_fix.jsonl); NE=$(wc -l < all_e.jsonl); N=$(( (NF / 2) / 100 * 100 )); [ $N -gt 2000 ] && N=2000
log "groups fix $NF e $NE -> N $N"; [ $N -ge 100 ] && [ $NE -ge $N ] || { log "NOT EXECUTABLE: too few groups (fix $NF, e $NE, N $N)"; exit 4; }
python3 subs.py all_fix.jsonl tr_d3_fix.jsonl $N && python3 subs.py all_e.jsonl tr_d3_e.jsonl $N --weights $WT || { log FAIL subs; exit 1; }
NP=$(( N / 4 )); [ $(wc -l < all_pc.jsonl) -ge $NP ] || NP=$(wc -l < all_pc.jsonl); python3 subs.py all_pc.jsonl tr_d3_pc.jsonl $NP || exit 1
C=/home/centuser/data/rtm/sel/ckpt/w20; for g in 0 1 2 3; do bash serve_sel.sh rtm-sel$g $g $((18341+g)) $C; done; up 18341 18342 18343 18344 || exit 1
python3 gloss.py all_fix.jsonl 18341,18342,18343,18344 loss_fix.txt --par 32 || { log FAIL gloss; exit 1; }
python3 subs.py all_fix.jsonl tr_d3_hard.jsonl $N --hard loss_fix.txt || exit 1
docker rm -f rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 >/dev/null 2>&1; log "data ready N=$N pc=$NP"
# 3. train smoke then four arms from W20, identical settings
TR="--model /w/ckpt/w20 --lr 1e-5 --epochs 1 --max-d 1100 --w-base 0.5 --seed 0"
head -64 tr_d3_fix.jsonl > tr_d3_smoke.jsonl; t0=$(date +%s); TRAIN smoke 0,1 2 $TR --data /w/tr_d3_smoke.jsonl --out /w/ckpt/d3_smoke > d3_smoke.log 2>&1 || { log FAIL smoke; exit 1; }; t1=$(date +%s); rm -rf ckpt/d3_smoke
EST=$(( (t1 - t0) * N / 64 )); log "smoke $((t1-t0))s for 64 groups incl. load; projected per arm ${EST}s"; [ $EST -le 900 ] || { log "NOT EXECUTABLE: projected training ${EST}s per arm exceeds the 15 min budget"; exit 4; }
TRAIN e 0,1 2 $TR --data /w/tr_d3_e.jsonl --out /w/ckpt/d3_e > d3_tr_e.log 2>&1 & P1=$!
TRAIN fix 2,3 2 $TR --data /w/tr_d3_fix.jsonl --out /w/ckpt/d3_fix > d3_tr_fix.log 2>&1 & P2=$!
TRAIN hard 4,5 2 $TR --data /w/tr_d3_hard.jsonl --out /w/ckpt/d3_hard > d3_tr_hard.log 2>&1 & P3=$!
TRAIN pc 6,7 2 $TR --data /w/tr_d3_pc.jsonl --out /w/ckpt/d3_pc > d3_tr_pc.log 2>&1 & P4=$!
for p in $P1 $P2 $P3 $P4; do wait $p || { log FAIL train; exit 1; }; done; log trained
# 4. evaluation: frozen eval worlds once; control world 977 for the training control
CK=/home/centuser/data/rtm/sel/ckpt
bash serve_sel.sh rtm-sel0 0 18341 $CK/w20; bash serve_sel.sh rtm-sel1 1 18342 $CK/w20; bash serve_sel.sh rtm-sel2 2 18343 $CK/d3_e; bash serve_sel.sh rtm-sel3 3 18344 $CK/d3_e
bash serve_sel.sh rtm-sel4 4 18345 $CK/d3_fix; bash serve_sel.sh rtm-sel5 5 18346 $CK/d3_hard; bash serve_sel.sh rtm-sel6 6 18347 $CK/d3_pc; bash serve_sel.sh rtm-sel7 7 18348 $CK/d3_pc
up 18341 18342 18343 18344 18345 18346 18347 18348 || exit 1
ev(){ p=$1; t=$2; f=$3; rm -f ctx_${t}.jsonl; K1_BUDGETS=1000,2000,4000 python3 infer.py env_$f.json --pooldir pool_$f --port $p --tag $t --cond 0 --par 8 > inf_$t.log 2>&1; }
ev 18341,18342 W20_ws6v ws6v & a=$!; ev 18343,18344 D3E_ws6v ws6v & b=$!; ev 18345 D3F_ws6v ws6v & c=$!; ev 18346 D3H_ws6v ws6v & d=$!; ev 18347,18348 D3P_ws6pc1 ws6pc1 & e=$!
wait $a && wait $b && wait $c && wait $d && wait $e || { log FAIL eval; exit 1; }
ev 18341,18342 W20_ws6pc1 ws6pc1 || exit 1
T=$(python3 -c "import json; print(','.join(json.load(open('d3_map.json'))['target']))")
python3 pcgate.py train env_ws6pc1.json 1000,2000,4000 W20_ws6pc1 D3P_ws6pc1 > d3_pc_train.txt; r=$?; cat d3_pc_train.txt
python3 d3eval.py env_ws6v.json 1000,2000,4000 $T W20_ws6v E=D3E_ws6v FIX=D3F_ws6v HARD=D3H_ws6v > d3_eval.txt; e=$?; tail -3 d3_eval.txt
[ $r = 0 ] || { log "PROCESS INVALID: training control failed; main result not interpretable"; exit 3; }
[ $e = 0 ] || { log "EVAL INVALID (code $e)"; exit 3; }
log D3BDONE
