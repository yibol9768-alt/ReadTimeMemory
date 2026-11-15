#!/bin/bash
cd ~/data/rtm/sel
docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-g rtm-env-o >/dev/null 2>&1
python3 wfilter.py ws1 10,11,12,13,14 14 ws1 || { echo FAIL filter; exit 1; }
bash ~/data/rtm/env/up_rr.sh
for k in $(seq 80); do curl -s -m 3 localhost:18312/v1/models >/dev/null && curl -s -m 3 localhost:18313/v1/models >/dev/null && break; sleep 15; done
python3 pool.py env_ws1.json --rr-port 18312,18313 --T 600 --par 8 --out pool_ws1 || { echo FAIL pool; exit 1; }
docker rm -f rtm-env-rr rtm-env-rr2 >/dev/null 2>&1
~/data/rtm/alfenv/bin/python wcheck.py ws1 || { echo FAIL check; exit 1; }
echo S1ADONE
