#!/bin/bash
# gpt-6.1-sol reader pilot, gen 1, one scale ($1): whole read side on sol (plan, notes, conflict check, answer); memories are
# the luna-built records / Hindsight contexts; judge stays luna. note3_deep (our single read) vs hindsight / Hindsight + cf.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_PROXY_PORT=18318 V3_PRO_STOP=95 V3_BEAM_SPLIT=$1 H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
N=$([ "$1" = 1m ] && echo 5 || echo 10)
V3_MODEL=gpt-6.1-sol python3 -m mem.beam answer --convs $N --conds note3_deep~m0,hindsight~m0 --par 8 || exit 1
V3_MODEL=gpt-6.1-sol V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs $N --conds cf_hindsight~m0 --par 8 || exit 1
V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6.1-sol python3 -m mem.beam judge --convs $N --par 8
