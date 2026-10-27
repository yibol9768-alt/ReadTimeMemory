#!/bin/bash
# Held-out test, luna reader: record retrieval + CC baseline (lychee_cf~m0 on all questions), one gen ($2), then judge.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_TEST=1 V3_PROXY_PORT=${PORT:-18318} V3_PLUS_REAL=${PLUS_REAL:-0} V3_MODEL=gpt-6-luna V3_PRO_STOP=101 V3_BEAM_SPLIT=$1 V3_GEN=$2
export H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
python3 -m mem.beam answer --convs 10 --conds lychee_cf~m0 --par 8 || exit 1
python3 -m mem.beam judge --convs 10 --par 8
