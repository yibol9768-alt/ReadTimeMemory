#!/bin/bash
# Held-out test (V3_TEST=1): main comparison on BEAM, luna reader + luna judge, 3 gens, same as dev.
# $1 = 100k|1m. note3_deep~m0, hindsight~m0 on all; cf_hindsight~m0 on router-POINT questions.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_TEST=1 V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 V3_BEAM_SPLIT=$1
export H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
N=10; P=${TP:-8}
for g in ${GENS:-1 2 3}; do
  export V3_GEN=$g
  python3 -m mem.beam answer --convs $N --conds note3_deep~m0,hindsight~m0 --par $P || exit 1
  V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs $N --conds cf_hindsight~m0 --par $P || exit 1
  python3 -m mem.beam judge --convs $N --par 8 || exit 1
done
