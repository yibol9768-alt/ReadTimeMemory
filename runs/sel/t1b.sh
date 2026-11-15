#!/bin/bash
# T1 evaluation after the consistency gate stopped the arm (only low-tail rank noise, see PROGRESS)
set -eu
cd ~/data/rtm/sel
export K1_BUDGETS=4000,16000
bash serve_sel.sh rtm-sel0 0 18341 /home/centuser/data/rtm/sel/ckpt/t1
bash serve_sel.sh rtm-sel1 1 18342 /home/centuser/data/rtm/sel/ckpt/t1
sleep 20
for k in $(seq 80); do curl -s -m 3 localhost:18341/v1/models >/dev/null && curl -s -m 3 localhost:18342/v1/models >/dev/null && break; sleep 15; done
python3 infer.py beam_ex.json --port 18341,18342 --tag T1_H --cond 0 --par 8
docker rm -f rtm-sel0 rtm-sel1 >/dev/null 2>&1 || true
echo "[$(TZ=America/New_York date +%H:%M)] T1 eval done"; echo T1DONE
