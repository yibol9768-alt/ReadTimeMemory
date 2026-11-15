#!/bin/bash
# helper (11-01 10:15): extra 8B rerankers on idle GPUs 0-4 to finish D3 pools in parallel with d3w.sh; removed before d3a needs GPUs 0-3
cd ~/data/rtm/sel; trap "docker rm -f rtm-rrx0 rtm-rrx1 rtm-rrx2 rtm-rrx3 rtm-rrx4 >/dev/null 2>&1" EXIT
bash up_rrx.sh
for k in $(seq 80); do ok=1; for p in 18370 18371 18372 18373 18374; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done
python3 -c "import json; I=[q[\"question_id\"] for q in json.load(open(\"env_ws6v.json\"))]; open(\"ws6v_back.txt\",\"w\").write(\"\n\".join(I[len(I)//2:]))"
X=18370,18371,18372,18373,18374
python3 pool.py env_ws6v.json --rr-port $X --T 600 --par 10 --out pool_ws6v --ids ws6v_back.txt > poolx_v.log 2>&1 &
python3 pool.py env_ws6x.json --rr-port $X --T 600 --par 10 --out pool_ws6x > poolx_x.log 2>&1 &
python3 pool.py env_ws6p0.json --rr-port $X --T 600 --par 6 --out pool_ws6p0 > poolx_p0.log 2>&1 &
python3 pool.py env_ws6p1.json --rr-port $X --T 600 --par 6 --out pool_ws6p1 > poolx_p1.log 2>&1 &
wait; echo POOLXDONE
