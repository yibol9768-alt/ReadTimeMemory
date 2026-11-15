#!/bin/bash
# H41b on LME with the minimal reader prompt: note_ly, lychee_cf (gen 1).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=lme H39_AB=knowledge-update,knowledge-update_abs,multi-session,multi-session_abs,single-session-assistant,single-session-preference,single-session-user,single-session-user_abs,temporal-reasoning,temporal-reasoning_abs
python3 -m mem.beam answer --conds note_ly,lychee_cf --par 8 && python3 -m mem.beam judge --par 8
