#!/bin/bash
# Held-out test with gpt-6.1-sol as reader (memories/contexts shared with pro6 via symlinks), luna judge, 3 gens.
# $1 = 100k|1m|lme. Same conditions as test_beam_3gen.sh / test_lme_3gen.sh.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_TEST=1 V3_PROXY_PORT=18318 V3_PRO_STOP=101 V3_BEAM_SPLIT=$1
D=runs/$( [ $1 = 100k ] && echo beam_test || ([ $1 = 1m ] && echo beam1m_test || echo lme_test))
A=$D/pro_gpt-6.1-sol; mkdir -p $A
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
    V3_MODEL=gpt-6.1-sol python3 -m mem.beam answer --conds note3_deep,lychee,hindsight --par 8 || exit 1
    V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6.1-sol python3 -m mem.beam judge --par 8 || exit 1
  else
    V3_MODEL=gpt-6.1-sol python3 -m mem.beam answer --convs 10 --conds note3_deep~m0,hindsight~m0 --par 8 || exit 1
    V3_MODEL=gpt-6.1-sol V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs 10 --conds cf_hindsight~m0 --par 8 || exit 1
    V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6.1-sol python3 -m mem.beam judge --convs 10 --par 8 || exit 1
  fi
done
