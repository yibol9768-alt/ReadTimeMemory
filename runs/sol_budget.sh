#!/bin/bash
# Phase diagram, budget axis: drift (roll vs rebuild) for gpt-6.1-sol and gpt-6-luna at 4000 and 2000 characters.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
for b in 4000 2000; do
  for m in gpt-6.1-sol gpt-6-luna; do
    V3_MODEL=$m python3 -m mem.drift build --convs 0,2,4,5,7,8 --par 2 --budget $b &&
    V3_MODEL=$m python3 -m mem.drift answer --convs 0,2,4,5,7,8 --par 3 --budget $b &&
    V3_MODEL=$m python3 -m mem.drift report --convs 0,2,4,5,7,8 --budget $b
  done
done
