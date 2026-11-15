#!/bin/bash
# sys1 without label routing for the conflict check: lychee_cf on every ability (gen 1), 100K and 1M. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
V3_BEAM_SPLIT=100k python3 -m mem.beam answer --convs 10 --conds lychee_cf --par 8 && V3_BEAM_SPLIT=100k python3 -m mem.beam judge --convs 10 --par 8
V3_BEAM_SPLIT=1m python3 -m mem.beam answer --convs 5 --conds lychee_cf --par 8 && V3_BEAM_SPLIT=1m python3 -m mem.beam judge --convs 5 --par 8
