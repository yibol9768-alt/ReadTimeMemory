#!/bin/bash
# Cross-reader check on BEAM-1M dev (100 q), gen 1: the whole read side (query plan, notes, conflict check, answer) on
# DeepSeek v4.1-flash (H20, 2 concurrent); memories are the same luna-built records / Hindsight contexts; judge stays luna.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_BEAM_SPLIT=1m H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
while [ "$(TZ=America/New_York date +%H%M)" -lt 1101 ]; do sleep 30; done
DS="V3_MODEL=deepseek-v4.1-flash"
env $DS V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs 5 --conds note3_ly~m0,lychee_cf~m0,cf_hindsight~m0 --par 2 || exit 1
env $DS V3_ONLY_ROUTE=AREA python3 -m mem.beam answer --convs 5 --conds lychee_b60~m0 --par 2 || exit 1
env $DS python3 -m mem.beam answer --convs 5 --conds hindsight~m0 --par 2 || exit 1
V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 V3_OUT_ARM=ds41 python3 -m mem.beam judge --convs 5 --par 8
