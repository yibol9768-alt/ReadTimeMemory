#!/bin/bash
# After dev1m5.sh: Hindsight 1M contexts read by our reader, 5 dev convs, 3 gens.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m
while kill -0 2409456 2>/dev/null; do sleep 60; done
for g in 1 2 3; do
  V3_GEN=$g python3 -m mem.beam answer --convs 5 --conds hindsight --par 6 && V3_GEN=$g python3 -m mem.beam judge --convs 5 --par 8
done
