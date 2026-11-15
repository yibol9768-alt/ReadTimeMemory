#!/bin/bash
# ours60 under the minimal reader prompt: area -> lychee_b60, point -> lychee_cf (conflict check on all abilities), gens 1-3, 100K and 1M. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
for g in 1 2 3; do
  for sp in "100k 10" "1m 5"; do set -- $sp
    V3_GEN=$g V3_BEAM_SPLIT=$1 V3_ONLY_ROUTE=AREA python3 -m mem.beam answer --convs $2 --conds lychee_b60 --par 8 || exit 1
    V3_GEN=$g V3_BEAM_SPLIT=$1 V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs $2 --conds lychee_cf --par 8 || exit 1
    V3_GEN=$g V3_BEAM_SPLIT=$1 python3 -m mem.beam judge --convs $2 --par 8 || exit 1
  done
done
