#!/bin/bash
# Note-part ablation with gpt-6.1-sol as the whole read side (gen1), luna judge; $1 = 1m|100k. Full note3_deep~m0 sol gen1 already exists.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_PROXY_PORT=18318 V3_PRO_STOP=101 V3_BEAM_SPLIT=$1 H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
N=$([ "$1" = 1m ] && echo 5 || echo 10)
V3_MODEL=gpt-6.1-sol python3 -m mem.beam answer --convs $N --conds nd_p1234~m0,nd_p1345~m0,nd_p1245~m0,nd_p1235~m0,nd_p1~m0 --par 8 || exit 1
V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6.1-sol python3 -m mem.beam judge --convs $N --par 8
