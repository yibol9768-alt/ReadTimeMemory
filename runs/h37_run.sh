#!/bin/bash
# H37: asp_full (all aspect digests) on router-AREA questions, 1M dev5 + 100K dev10, 3 gens. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
# (build finished; 1M conv 0 regrouped with retry at 11:25)
ls runs/beam1m/pro6/aspect/asp_4.json runs/beam/pro6/aspect/asp_18.json || exit 1
for g in 1 2 3; do
  for sp in "1m 5" "100k 10"; do set -- $sp
    V3_BEAM_SPLIT=$1 V3_GEN=$g python3 -m mem.beam answer --convs $2 --conds asp_full --par 6 &&
    V3_BEAM_SPLIT=$1 V3_GEN=$g python3 -m mem.beam judge --convs $2 --par 8 || exit 1
  done
done
