#!/bin/bash
# H41c pilot: note3_ly~m0 (notes + standing instructions) on point questions, gen 1, 1M (63 q) and 100K (128 q). luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_ONLY_ROUTE=POINT
for S in 1m:5 100k:10; do
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam answer --convs ${S#*:} --conds note3_ly~m0 --par 8 || exit 1
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam judge --convs ${S#*:} --par 8 || exit 1
done
