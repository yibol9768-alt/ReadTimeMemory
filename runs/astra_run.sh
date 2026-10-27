#!/bin/bash
# gpt-6-astra as the whole read side (user 10-20: "astra sol 都跑一下"); memories unchanged, luna judge.
# $1 = 1m|100k|lme, $2 = gens (e.g. "1" or "2 3"). Stops at Pro weekly 99% (V3_PRO_STOP; astra is the first to stop).
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_PROXY_PORT=18318 V3_PRO_STOP=99 V3_BEAM_SPLIT=$1
P=${AP:-10}
if [ $1 = lme ]; then
  export H39_AB=knowledge-update,knowledge-update_abs,multi-session,multi-session_abs,single-session-assistant,single-session-preference,single-session-user,single-session-user_abs,temporal-reasoning,temporal-reasoning_abs
else
  export H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
  N=$([ $1 = 1m ] && echo 5 || echo 10)
fi
for g in $2; do
  export V3_GEN=$g
  if [ $1 = lme ]; then
    V3_MODEL=gpt-6-astra python3 -m mem.beam answer --conds note3_deep,lychee,hindsight --par $P || exit 1
    V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6-astra V3_PRO_STOP=101 python3 -m mem.beam judge --par 8 || exit 1
  else
    V3_MODEL=gpt-6-astra python3 -m mem.beam answer --convs $N --conds note3_deep~m0,hindsight~m0 --par $P || exit 1
    V3_MODEL=gpt-6-astra V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs $N --conds cf_hindsight~m0 --par $P || exit 1
    V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6-astra V3_PRO_STOP=101 python3 -m mem.beam judge --convs $N --par 8 || exit 1
  fi
done
