#!/bin/bash
# DeepSeek reader (n03 weights DeepSeek-V4.1-Flash-1017), full: one BEAM scale ($1 = 1m|100k), gens 1-3, all main
# conditions, one answer process at a time with par 32 (two scales = 64 in flight), no time window (user 10-17).
# Memories are the luna-built records / Hindsight contexts; judge stays luna (official BEAM prompt).
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_BEAM_SPLIT=$1 V3_DS_VER=1017 V3_NO_WINDOW=1 H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning
N=$([ "$1" = 1m ] && echo 5 || echo 10)
check() {   # stop if n03 no longer serves the 1017 weights
  for t in 1 2 3; do
    m=$(timeout 60 ssh -o ConnectTimeout=20 h20 'docker inspect -f "{{range .Mounts}}{{.Source}} {{end}}" dsv41-agent-candidate-v9' 2>/dev/null) && break
    sleep 30
  done
  echo "$(date) weights: $m" | grep -o "DeepSeek-V4.1-Flash-[0-9]*" || true
  echo "$m" | grep -q "DeepSeek-V4.1-Flash-1017" || { echo "n03 weights changed or unreachable: stop"; exit 1; }
}
for g in 1 2 3; do
  export V3_GEN=$g
  check
  V3_MODEL=deepseek-v4.1-flash V3_ONLY_ROUTE=POINT python3 -m mem.beam answer --convs $N --conds note3_ly~m0,lychee_cf~m0,cf_hindsight~m0 --par 32 || exit 1
  V3_MODEL=deepseek-v4.1-flash V3_ONLY_ROUTE=AREA python3 -m mem.beam answer --convs $N --conds lychee_b60~m0 --par 32 || exit 1
  V3_MODEL=deepseek-v4.1-flash python3 -m mem.beam answer --convs $N --conds hindsight~m0,note3_deep~m0 --par 32 || exit 1
  check
  V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 V3_OUT_ARM=ds41_1017 python3 -m mem.beam judge --convs $N --par 8 || exit 1
done
