#!/bin/bash
# H41b prompt-independence: note_ly with the minimal reader prompt on point questions, gen 1, 1M + 100K (notes cached). luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_ONLY_ROUTE=POINT
for S in 1m:5 100k:10; do
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam answer --convs ${S#*:} --conds note_ly --par 8 || exit 1
  V3_BEAM_SPLIT=${S%:*} python3 -m mem.beam judge --convs ${S#*:} --par 8 || exit 1
done
