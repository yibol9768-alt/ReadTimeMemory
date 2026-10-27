#!/bin/bash
# H41g pilot: note3_deep~m0 (notes + full 60K records) on all questions, gen 1, 1M + 100K. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
for S in 1m:5 100k:10; do
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam answer --convs ${S#*:} --conds note3_deep~m0 --par 8 || exit 1
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam judge --convs ${S#*:} --par 8 || exit 1
done
