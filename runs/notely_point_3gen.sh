#!/bin/bash
# H41b: point questions -> note_ly~m0 (evidence notes from 60K lychee records + 8K view). 1M gens 2-3 (gen1 exists), 100K gens 1-3. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_ONLY_ROUTE=POINT
S=$1
if [ "$S" = 1m ]; then GENS="2 3"; N=5; else GENS="1 2 3"; N=10; fi
for g in $GENS; do
  V3_BEAM_SPLIT=$S V3_GEN=$g python3 -m mem.beam answer --convs $N --conds note_ly~m0 --par 8 || exit 1
  V3_BEAM_SPLIT=$S V3_GEN=$g python3 -m mem.beam judge --convs $N --par 8 || exit 1
done
