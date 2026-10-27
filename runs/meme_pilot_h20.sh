#!/bin/bash
# MEME pilot 1, open models only (user 10-23: stop using luna). Waits for the H20 window (11:05 ET).
# 1) Qwen judges the existing luna answers (state, full); 2) DeepSeek-V4.1-Flash reader for both conds, Qwen judge.
cd "$(dirname "$0")/.."; export PYTHONPATH=.
T=$(TZ=America/New_York date -d "today 11:05" +%s); while [ $(date +%s) -lt $T ]; do sleep 120; done
curl -s -m 10 -o /dev/null http://127.0.0.1:18003/v1/models || ssh -fN -o ExitOnForwardFailure=yes -L 18003:127.0.0.1:18003 h20
D=/tmp/claude-1000/-mnt-c-Users-liuyibo-Desktop-auto-llm-driving/9d982558-f9f7-476e-89c7-66eeb913ad1e/scratchpad/meme/nofiller.json
J=Qwen3.8-27B-FP8
for c in state full; do V3_MODEL=gpt-6-luna V3_PROXY_PORT=18319 V3_PLUS_REAL=1 python3 -m mem.stateq judge --cond $c --suffix _p1 --judge $J; done
export V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9
python3 -m mem.stateq run --data $D --ids pl_002,pl_003,pl_004,pl_005,pl_006,pl_007,pl_008,pl_009,pl_010,pl_011,sw_002,sw_003,sw_004,sw_005,sw_006,sw_007,sw_008,sw_009,sw_010,sw_011 --cond state --suffix _p1 --par 1
python3 -m mem.stateq run --data $D --ids pl_002,pl_003,pl_004,pl_005,pl_006,pl_007,pl_008,pl_009,pl_010,pl_011,sw_002,sw_003,sw_004,sw_005,sw_006,sw_007,sw_008,sw_009,sw_010,sw_011 --cond full --suffix _p1 --par 1
for c in state full; do python3 -m mem.stateq judge --cond $c --suffix _p1 --judge $J; done
echo "all done $(TZ=America/New_York date +%H:%M)"
