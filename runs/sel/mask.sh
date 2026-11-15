#!/bin/bash
# Candidate-date mask probe (10-31): T1 and untrained 0.6B on BEAM-500K dev with dates hidden from candidates.
cd ~/data/rtm/sel
trap 'docker rm -f rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 >/dev/null 2>&1' EXIT
B06=/home/centuser/data/models/qwen/Qwen3-Reranker-0.6B; T1=/home/centuser/data/rtm/sel/ckpt/t1
bash serve_sel.sh rtm-sel0 0 18341 $T1; bash serve_sel.sh rtm-sel1 1 18342 $T1; bash serve_sel.sh rtm-sel2 2 18343 $B06; bash serve_sel.sh rtm-sel3 3 18344 $B06
sleep 20; for k in $(seq 80); do ok=1; for p in 18341 18342 18343 18344; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done
[ $ok = 1 ] || { echo FAIL serve; exit 1; }
rm -f ctx_T1M_H.jsonl ctx_A0M_H.jsonl
MASK_DATE=1 python3 infer.py beam_ex.json --port 18341,18342 --tag T1M_H --cond 0 --par 8 || echo FAIL t1 &
MASK_DATE=1 python3 infer.py beam_ex.json --port 18343,18344 --tag A0M_H --cond 0 --par 8 || echo FAIL a0 &
wait; echo MASKDONE
