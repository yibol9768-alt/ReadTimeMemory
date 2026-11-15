#!/bin/bash
# H39 confirmation: lychee_cf + cf_hindsight on contradiction + knowledge_update, gens 2-3 (gen 1 done), 100K and 1M. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
for g in 2 3; do
  V3_BEAM_SPLIT=100k V3_GEN=$g python3 -m mem.beam answer --convs 10 --conds lychee_cf,cf_hindsight --par 8 && V3_BEAM_SPLIT=100k V3_GEN=$g python3 -m mem.beam judge --convs 10 --par 8
  V3_BEAM_SPLIT=1m V3_GEN=$g python3 -m mem.beam answer --convs 5 --conds lychee_cf --par 8 && V3_BEAM_SPLIT=1m V3_GEN=$g python3 -m mem.beam judge --convs 5 --par 8
done
