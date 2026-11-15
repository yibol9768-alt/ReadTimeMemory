#!/bin/bash
# rerun of the baseline arm after the consistency gate stopped it (ranking passed; dP limit too tight)
set -eu
cd ~/data/rtm/sel
export K1_BUDGETS=4000,16000
B06=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
bash serve_sel.sh rtm-sel6 6 18331 $B06
sleep 20
for k in $(seq 80); do curl -s -m 3 localhost:18331/v1/models >/dev/null && curl -s -m 3 localhost:18332/v1/models >/dev/null && break; sleep 15; done
python3 infer.py beam_ex.json --port 18331,18332 --tag A0_H --cond 0 --par 8
python3 pool.py beam_ex.json --rr-port 18331,18332 --rr-model sel --T 600 --par 8 --out pool_h06
python3 cov.py rr8 beam_ex.json A0O_H --pooldir pool_h06
docker rm -f rtm-sel6 rtm-sel7 >/dev/null 2>&1 || true
echo "[$(TZ=America/New_York date +%H:%M)] baselines done"; echo BASEDONE
