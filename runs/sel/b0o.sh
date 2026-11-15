#!/bin/bash
# B0 options probe (10-31): untrained 0.6B on GPU 4, T1 on GPUs 6,7. No training.
cd ~/data/rtm/sel
B06=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B
trap 'docker rm -f rtm-sel4 rtm-sel6 rtm-sel7 >/dev/null 2>&1' EXIT
waitp(){ sleep 20; for k in $(seq 80); do ok=1; for p in "$@"; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && return 0; sleep 15; done; return 1; }
run(){ t=$1; p=$2
  python3 cond_probe3.py beam_ex.json hard_ids.txt pool --port $p --out opt_${t}b.json > opt_${t}b.log 2>&1 || echo "FAIL $t beam"
  python3 cond_probe3.py q_lmes150.json ids_lmes150.txt pool_lmes150 --port $p --out opt_${t}l.json --min-gold 2 > opt_${t}l.log 2>&1 || echo "FAIL $t lme"; }
bash serve_sel.sh rtm-sel4 4 18347 $B06; bash serve_sel.sh rtm-sel6 6 18348 /home/centuser/data/rtm/sel/ckpt/t1; bash serve_sel.sh rtm-sel7 7 18349 /home/centuser/data/rtm/sel/ckpt/t1
waitp 18347 18348 18349 || { echo FAIL serve; exit 1; }
python3 cond_probe3.py beam_ex.json hard_ids.txt pool --port 18347 --out opt_tiny.json --limit 3 > opt_tiny.log 2>&1 || { echo FAIL tiny; cat opt_tiny.log; exit 1; }
run a0 18347 &
run t1 18348,18349 &
wait; echo B0ODONE
