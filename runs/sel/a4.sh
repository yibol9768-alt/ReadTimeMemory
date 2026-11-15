#!/bin/bash
# A4 per docs/A4_SPEC_2026-11-01.md. usage: bash a4.sh smoke|disc|indep. GPUs: 5 embedding, 1-3 A0, 4,6,7 W20 (0 is A3). Containers removed at exit.
cd ~/data/rtm/sel; ST=$1; D=a4_$ST; mkdir -p $D
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-sel1 rtm-sel2 rtm-sel3 rtm-sel4 rtm-sel6 rtm-sel7 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1' EXIT
case $ST in
  smoke) L=q_lmes150.json; B=beam_ex.json; N=3;;
  disc)  L=q_lmes150.json; B=beam_ex.json; N=0;;
  indep) L=q_dv_lme_raw.json; B=q_dv_beam.json; N=0;;
esac
python3 - $L $B $N $D <<'PY' || exit 1
import json, sys, interv as V
L, B, N, D = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
has = lambda q: any(t.get("has_answer") for t in V.turns(q))
def cut(f):
    Q = [q for q in json.load(open(f)) if has(q)]
    if N:   # smoke: questions from one or two histories so the run is small
        Q = Q[:N]
    return Q
l, b = cut(L), cut(B); json.dump(l, open(f"{D}/lme_orig.json", "w")); json.dump(b, open(f"{D}/beam_orig.json", "w"))
for n in ("swapgold", "swapother", "insbtw", "insalt"):
    o, g = V.run(n, l); json.dump(o, open(f"{D}/lme_{n}.json", "w")); json.dump(g, open(f"{D}/lme_{n}.json.log.json", "w")); print(n, len(o))
PY
docker start rtm-env-emb >/dev/null; HB=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B; C=/home/centuser/data/rtm/sel/ckpt/w20
for g in 1 2 3; do bash serve_sel.sh rtm-sel$g $g $((18340+g)) $HB; done; for g in 4 6 7; do bash serve_sel.sh rtm-sel$g $g $((18340+g)) $C; done
for k in $(seq 80); do ok=1; for p in 18311 18341 18342 18343 18344 18346 18347; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done; [ $ok = 1 ] || { log FAIL up; exit 1; }
log up
for f in $D/lme_*.json $D/beam_orig.json; do case $f in *.log.json) continue;; esac; p=${f%.json}; python3 pool.py $f --T 0 --par 16 --out ${p}_pool > ${p}_pool.log 2>&1 || { log "FAIL pool $f"; exit 1; }; done
log pools done
docker stop rtm-env-emb >/dev/null
run(){ port=$1; t=$2
  for f in $D/lme_*.json; do case $f in *.log.json) continue;; esac; n=$(basename ${f%.json}); K1_BUDGETS=2000 python3 infer.py $f --pooldir ${f%.json}_pool --port $port --tag ${t}_a4${ST}_$n --cond 0 --par 8 || return 1; done
  K1_BUDGETS=4000 python3 infer.py $D/beam_orig.json --pooldir $D/beam_orig_pool --port $port --tag ${t}_a4${ST}_beam_orig --cond 0 --par 8 || return 1
  MASK_DATE=1 K1_BUDGETS=4000 python3 infer.py $D/beam_orig.json --pooldir $D/beam_orig_pool --port $port --tag ${t}_a4${ST}_beam_mask --cond 0 --par 8 || return 1; }
rm -f ctx_A0_a4${ST}_*.jsonl ctx_W20_a4${ST}_*.jsonl
run 18341,18342,18343 A0 > $D/a0.log 2>&1 & A=$!; run 18344,18346,18347 W20 > $D/w20.log 2>&1 & W=$!
wait $A || { log FAIL a0; exit 1; }; wait $W || { log FAIL w20; exit 1; }
log A4DONE $ST
