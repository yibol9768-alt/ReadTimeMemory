#!/bin/bash
# Note-part ablation with DeepSeek-V4.1-Flash 1017 (n03) as the whole read side, gen1, 2 in flight (user 10-19); 1M then 100K; luna judge.
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_DS_VER=1017 V3_NO_WINDOW=1 H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
for S in ${SCALES:-1m 100k}; do
  for t in 1 2 3 4 5; do    # a single ssh hiccup must not stop the run
    m=$(timeout 60 ssh -o ConnectTimeout=20 h20 'docker inspect -f "{{range .Mounts}}{{.Source}} {{end}}" dsv41-agent-candidate-v9' 2>/dev/null) && [ -n "$m" ] && break
    sleep 60
  done
  echo "$m" | grep -q "DeepSeek-V4.1-Flash-1017" || { echo "n03 weights changed or unreachable: stop"; exit 1; }
  N=$([ $S = 1m ] && echo 5 || echo 10)
  V3_BEAM_SPLIT=$S V3_MODEL=deepseek-v4.1-flash python3 -m mem.beam answer --convs $N --conds nd_p1234~m0,nd_p1345~m0,nd_p1245~m0,nd_p1235~m0,nd_p1~m0 --par 2 || exit 1
  V3_BEAM_SPLIT=$S V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 V3_OUT_ARM=ds41_1017 python3 -m mem.beam judge --convs $N --par 8 || exit 1
done
