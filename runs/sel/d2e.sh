#!/bin/bash
# D2 end-to-end on the pilot worlds: filter, pools (8B teacher), mkdata stats, A0 and W20 selection at 4k. Containers removed at exit.
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-env-rr rtm-env-rr2 rtm-sel1 rtm-sel2 rtm-sel3 rtm-sel4 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1' EXIT
python3 wfilter.py ws5p 950,951 951 ws5p || exit 1
docker start rtm-env-emb >/dev/null; bash ~/data/rtm/env/up_rr.sh
for k in $(seq 80); do curl -s -m 3 localhost:18311/v1/models >/dev/null && curl -s -m 3 localhost:18312/v1/models >/dev/null && curl -s -m 3 localhost:18313/v1/models >/dev/null && break; sleep 15; done
rm -rf pool_ws5p; python3 pool.py env_ws5p.json --rr-port 18312,18313 --T 600 --par 8 --out pool_ws5p > pool_ws5p.log 2>&1 || { log FAIL pool; exit 1; }; log pool done
docker rm -f rtm-env-rr rtm-env-rr2 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null
HS=$(cat hs.txt 2>/dev/null || echo 10); python3 mkdata.py env_ws5p.json pool_ws5p tr_ws5p.jsonl --no-set --ignore-kinds repeat,fu --hard-skip $HS --boost 3 --steps 8 > mk_ws5p.log 2>&1 || { log FAIL mkdata; exit 1; }
HB=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B; C=/home/centuser/data/rtm/sel/ckpt/w20
bash serve_sel.sh rtm-sel1 1 18341 $HB; bash serve_sel.sh rtm-sel2 2 18342 $HB; bash serve_sel.sh rtm-sel3 3 18343 $C; bash serve_sel.sh rtm-sel4 4 18344 $C
for k in $(seq 80); do ok=1; for p in 18341 18342 18343 18344; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done
rm -f ctx_A0_ws5p.jsonl ctx_W20_ws5p.jsonl
K1_BUDGETS=4000 python3 infer.py env_ws5p.json --pooldir pool_ws5p --port 18341,18342 --tag A0_ws5p --cond 0 --par 8 > inf_a0_ws5p.log 2>&1 & A=$!
K1_BUDGETS=4000 python3 infer.py env_ws5p.json --pooldir pool_ws5p --port 18343,18344 --tag W20_ws5p --cond 0 --par 8 > inf_w20_ws5p.log 2>&1 & B=$!
wait $A || { log FAIL a0; exit 1; }; wait $B || { log FAIL w20; exit 1; }
K1_BUDGETS=4000 python3 cov.py rr8 env_ws5p.json A0O_ws5p --pooldir pool_ws5p > /dev/null 2>&1
log D2EDONE
