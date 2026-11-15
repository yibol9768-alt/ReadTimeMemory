#!/bin/bash
# H41b on LongMemEval-S dev (62 q, all questions, no router): note_ly~m0 vs lychee_cf~m0 (cf on all types) vs lychee~m0; gen 1. luna + official LME judge.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=lme H39_AB=knowledge-update,knowledge-update_abs,multi-session,multi-session_abs,single-session-assistant,single-session-preference,single-session-user,single-session-user_abs,temporal-reasoning,temporal-reasoning_abs
python3 -m mem.beam answer --conds note_ly~m0,lychee_cf~m0,lychee~m0 --par 8 && python3 -m mem.beam judge --par 8
