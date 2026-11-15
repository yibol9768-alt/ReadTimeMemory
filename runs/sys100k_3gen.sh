#!/bin/bash
# 100K gens 2-3 for the composite: area -> lychee_b60~m0, point -> lychee_cf~m0 (conflict check on all), plus hindsight~m0 on all 200. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=100k H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
for g in 2 3; do
  V3_GEN=$g V3_ONLY_ROUTE=AREA python3 -m mem.beam answer --convs 10 --conds lychee_b60~m0 --par 8 || exit 1
  V3_GEN=$g V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs 10 --conds lychee_cf~m0 --par 8 || exit 1
  V3_GEN=$g python3 -m mem.beam answer --convs 10 --conds hindsight~m0 --par 8 || exit 1
  V3_GEN=$g python3 -m mem.beam judge --convs 10 --par 8 || exit 1
done
