#!/bin/bash
# Phase diagram, horizon axis: BEAM-1M (82-92 sessions of 20 turns), 5 conversations, sol and luna, budget 8000.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
for m in gpt-6.1-sol gpt-6-luna; do
  V3_MODEL=$m python3 -m mem.longdrift build --split 1M --convs 0,1,2,3,4 --par 2 &&
  V3_MODEL=$m python3 -m mem.longdrift report --split 1M --convs 0,1,2,3,4
done
