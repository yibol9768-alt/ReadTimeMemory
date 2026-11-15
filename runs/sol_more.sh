#!/bin/bash
# gpt-6.1-sol reader, gens 2-3 on one BEAM scale ($1 = 1m|100k), started after that scale's ablation gen1 ($2 = its PID):
# main comparison (note3_deep, hindsight, Hindsight + conflict check on point questions), then the note-part ablation. luna judge.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_PROXY_PORT=18318 V3_PRO_STOP=101 V3_BEAM_SPLIT=$1 H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
N=$([ "$1" = 1m ] && echo 5 || echo 10)
while kill -0 $2 2>/dev/null; do sleep 60; done
for g in 2 3; do
  export V3_GEN=$g
  V3_MODEL=gpt-6.1-sol python3 -m mem.beam answer --convs $N --conds note3_deep~m0,hindsight~m0 --par ${SP:-16} || exit 1
  V3_MODEL=gpt-6.1-sol V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs $N --conds cf_hindsight~m0 --par ${SP:-16} || exit 1
  V3_MODEL=gpt-6.1-sol python3 -m mem.beam answer --convs $N --conds nd_p1234~m0,nd_p1345~m0,nd_p1245~m0,nd_p1235~m0,nd_p1~m0 --par ${SP:-16} || exit 1
  V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6.1-sol python3 -m mem.beam judge --convs $N --par 12 || exit 1
done
