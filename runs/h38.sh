#!/bin/bash
# H38: non-adjudicating ledger (mem/nadj.py), 100K dev10 then 1M dev5; lychee reader on it, 3 gens. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
for sp in "100k 10" "1m 5"; do set -- $sp
  V3_BEAM_SPLIT=$1 python3 -m mem.nadj build --convs $2 --par 5 || exit 1
  for g in 1 2 3; do
    V3_BEAM_SPLIT=$1 V3_GEN=$g python3 -m mem.beam answer --convs $2 --conds nadj --par 6 &&
    V3_BEAM_SPLIT=$1 V3_GEN=$g python3 -m mem.beam judge --convs $2 --par 8 || exit 1
  done
done
