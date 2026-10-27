#!/bin/bash
# PROTOCOL change 10-16: BEAM-1M dev = rows 0-4 (5 convs, 100 q). Build lychee for rows 2-4, then rag / lychee
# (+ hindsight once its contexts are exported) for 3 generations. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m
python3 -m mem.beam build --convs 5 --par 3 || exit 1
for g in 1 2 3; do
  V3_GEN=$g python3 -m mem.beam answer --convs 5 --conds rag,lychee,hindsight --par 6 && V3_GEN=$g python3 -m mem.beam judge --convs 5 --par 8
done
