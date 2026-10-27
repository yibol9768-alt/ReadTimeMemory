#!/bin/bash
# MEME pilot 1 (state queries vs full context), 20 held-out episodes (dev = pl_001, sw_001), reader luna via Plus.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_PROXY_PORT=18319 V3_PLUS_REAL=1 V3_MODEL=gpt-6-luna
D=/tmp/claude-1000/-mnt-c-Users-liuyibo-Desktop-auto-llm-driving/9d982558-f9f7-476e-89c7-66eeb913ad1e/scratchpad/meme/nofiller.json
python3 -m mem.stateq run --data $D --ids pl_002,pl_003,pl_004,pl_005,pl_006,pl_007,pl_008,pl_009,pl_010,pl_011,sw_002,sw_003,sw_004,sw_005,sw_006,sw_007,sw_008,sw_009,sw_010,sw_011 --cond state --suffix _p1 --par 4 &
python3 -m mem.stateq run --data $D --ids pl_002,pl_003,pl_004,pl_005,pl_006,pl_007,pl_008,pl_009,pl_010,pl_011,sw_002,sw_003,sw_004,sw_005,sw_006,sw_007,sw_008,sw_009,sw_010,sw_011 --cond full --suffix _p1 --par 4
wait
for c in state full; do python3 -m mem.stateq judge --cond $c --suffix _p1 --judge gpt-6-luna; done
python3 -c "from mem import llm; print('plus', llm.plus_used())"
