#!/bin/bash
# gens 2-3 of cf_hindsight~m0 on router-POINT questions, 100K then 1M. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_ONLY_ROUTE=POINT H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
for g in 2 3; do
  V3_GEN=$g V3_BEAM_SPLIT=100k python3 -m mem.beam answer --convs 10 --conds cf_hindsight~m0 --par 8 && V3_GEN=$g V3_BEAM_SPLIT=100k python3 -m mem.beam judge --convs 10 --par 8 || exit 1
  V3_GEN=$g V3_BEAM_SPLIT=1m python3 -m mem.beam answer --convs 5 --conds cf_hindsight~m0 --par 8 && V3_GEN=$g V3_BEAM_SPLIT=1m python3 -m mem.beam judge --convs 5 --par 8 || exit 1
done
