#!/bin/bash
# H41e pilot: note15_ly~m0 (15000-char notes) on AREA questions, gen 1, 1M (37) + 100K (72). luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_ONLY_ROUTE=AREA
for S in 1m:5 100k:10; do
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam answer --convs ${S#*:} --conds note15_ly~m0 --par 8 || exit 1
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam judge --convs ${S#*:} --par 8 || exit 1
done
