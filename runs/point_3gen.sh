#!/bin/bash
# gens 2-3 for point questions (router POINT, 63 q, 1M): lychee_cf~m0 (conflict check on every ability) and hindsight~m0. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m V3_ONLY_ROUTE=POINT H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
for g in 2 3; do
  V3_GEN=$g python3 -m mem.beam answer --convs 5 --conds lychee_cf~m0,hindsight~m0 --par 8 || exit 1
  V3_GEN=$g python3 -m mem.beam judge --convs 5 --par 8 || exit 1
done
