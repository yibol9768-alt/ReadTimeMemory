#!/bin/bash
# Interim check of the DeepSeek (1017) runs: luna-judge answers written so far, then compare conditions on shared questions.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 V3_OUT_ARM=ds41_1017
for S in 1m:5 100k:10; do
  for g in 1 2 3; do
    f=runs/$([ ${S%:*} = 1m ] && echo beam1m || echo beam)/ds41_1017/answers$([ $g = 1 ] || echo _g$g).jsonl
    [ -s $f ] && V3_GEN=$g V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam judge --convs ${S#*:} --par 8 > /dev/null 2>&1
  done
done
python3 runs/interim_ds.py
