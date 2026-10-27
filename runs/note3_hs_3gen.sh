#!/bin/bash
# H41f gens 2-3.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_ONLY_ROUTE=POINT
for g in 2 3; do for S in 1m:5 100k:10; do
  V3_GEN=$g V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam answer --convs ${S#*:} --conds note3_hs~m0 --par 8 || exit 1
  V3_GEN=$g V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam judge --convs ${S#*:} --par 8 || exit 1
done; done
