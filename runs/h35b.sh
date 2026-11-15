#!/bin/bash
# H35b: wide_mix on all abilities (no oracle routing), 1M dev5 + 100K dev10, 3 gens. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. H35_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
for g in 1 2 3; do
  for sp in "1m 5" "100k 10"; do set -- $sp
    V3_BEAM_SPLIT=$1 V3_GEN=$g python3 -m mem.beam answer --convs $2 --conds wide_mix --par 6 &&
    V3_BEAM_SPLIT=$1 V3_GEN=$g python3 -m mem.beam judge --convs $2 --par 8 || exit 1
  done
done
