#!/bin/bash
# Held-out test, another reader for the whole read side (memories/contexts shared with pro6 via symlinks), luna judge.
# $1 = 100k|1m|lme. Env: R = reader model (default gpt-6.1-sol), GENS, P (par), STOP (Pro stop %), V3_DS_VER for DeepSeek.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_TEST=1 V3_PROXY_PORT=18318 V3_BEAM_SPLIT=$1
R=${R:-gpt-6.1-sol}; P=${P:-8}; export V3_PRO_STOP=${STOP:-101}
ARM=$(V3_MODEL=$R python3 -c "from mem import llm; print(llm.ARM)")
D=runs/$( [ $1 = 100k ] && echo beam_test || ([ $1 = 1m ] && echo beam1m_test || echo lme_test))
A=$D/$ARM; mkdir -p $A
[ -e $A/ext ] || ln -s ../pro6/ext $A/ext
[ -e $A/lychee ] || ln -s ../pro6/lychee $A/lychee
[ -e $D/pro6/route.json ] && [ ! -e $A/route.json ] && cp $D/pro6/route.json $A/
if [ $1 = lme ]; then
  export H39_AB=knowledge-update,knowledge-update_abs,multi-session,multi-session_abs,single-session-assistant,single-session-preference,single-session-user,single-session-user_abs,temporal-reasoning,temporal-reasoning_abs
else
  export H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
fi
for g in ${GENS:-1 2 3}; do
  export V3_GEN=$g
  if [ $1 = lme ]; then
    V3_MODEL=$R python3 -m mem.beam answer --conds note3_deep,lychee,hindsight --par $P || exit 1
  else
    V3_MODEL=$R python3 -m mem.beam answer --convs 10 --conds note3_deep~m0,hindsight~m0,lychee_cf~m0 --par $P || exit 1
    V3_MODEL=$R V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs 10 --conds cf_hindsight~m0 --par $P || exit 1
  fi
  [ -n "$NOJUDGE" ] || V3_MODEL=gpt-6-luna V3_OUT_ARM=$ARM V3_PRO_STOP=101 python3 -m mem.beam judge $([ $1 = lme ] || echo --convs 10) --par 8 || exit 1
done
