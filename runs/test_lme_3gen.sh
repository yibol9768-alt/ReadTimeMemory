#!/bin/bash
# Held-out LME-S test (151 q, V3_TEST=1): minimal reader prompt, luna reader + luna judge (official LME prompt), 3 gens.
# Same conditions as dev (lme_final.sh + hindsight): note3_deep, lychee, hindsight.
while kill -0 958150 2>/dev/null; do sleep 60; done   # LME test records (restarted build) must be complete
grep -q Traceback runs/test_logs/build_lme2.log && { echo "LME build failed"; exit 1; }
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_TEST=1 V3_PROXY_PORT=18318 V3_PRO_STOP=101 V3_BEAM_SPLIT=lme V3_MODEL=gpt-6-luna
export H39_AB=knowledge-update,knowledge-update_abs,multi-session,multi-session_abs,single-session-assistant,single-session-preference,single-session-user,single-session-user_abs,temporal-reasoning,temporal-reasoning_abs
for g in ${GENS:-1 2 3}; do
  export V3_GEN=$g
  python3 -m mem.beam answer --conds note3_deep,lychee,hindsight --par 8 || exit 1
  python3 -m mem.beam judge --par 8 || exit 1
done
