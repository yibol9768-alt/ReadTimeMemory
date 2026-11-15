#!/bin/bash
# gens 2-3 for the area-question depth/representation comparison (router AREA only, 37 q, 1M). luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m V3_ONLY_ROUTE=AREA
for g in 2 3; do
  V3_GEN=$g python3 -m mem.beam answer --convs 5 --conds hindsight~m0,hs60~m0,lychee_b60~m0,lychee_b120~m0 --par 8 || exit 1
  V3_GEN=$g python3 -m mem.beam judge --convs 5 --par 8 || exit 1
done
