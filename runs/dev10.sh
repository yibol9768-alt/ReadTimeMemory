#!/bin/bash
# PROTOCOL change 10-16: BEAM-100K dev = 10 convs. Baselines on the 4 new convs (2,3,9,11), 3 generations, luna.
# Existing 6 convs are cached. Hindsight (AMB units 3,4,10,12) runs in parallel via amb_beam_dev.sh.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.beam build --convs 10 --par 4 || exit 1
for g in 1 2 3; do
  V3_GEN=$g python3 -m mem.beam answer --convs 10 --conds full,rag,lychee --par 6 && V3_GEN=$g python3 -m mem.beam judge --convs 10 --par 8
done
python3 -m mem.beam report --convs 10
