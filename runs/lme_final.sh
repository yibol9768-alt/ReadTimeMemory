#!/bin/bash
# LongMemEval-S dev (62 q), minimal reader prompt, gens 1-3. $1 = reader (luna|ds|sol), $2 = conditions.
# Memories are the luna-built records / Hindsight contexts; the whole read side runs on the reader; judge = luna (official LME prompt).
# DeepSeek: n03 weights 1017, 2 requests in flight (user 10-19).
cd "$(dirname "$0")/.." && export PYTHONPATH=. V3_PROXY_PORT=18318 V3_PRO_STOP=101 V3_BEAM_SPLIT=lme H39_AB=knowledge-update,knowledge-update_abs,multi-session,multi-session_abs,single-session-assistant,single-session-preference,single-session-user,single-session-user_abs,temporal-reasoning,temporal-reasoning_abs
case $1 in
  luna) M=gpt-6-luna; P=8; ARM=pro6 ;;
  sol)  M=gpt-6.1-sol; P=8; ARM=pro_gpt-6.1-sol ;;
  ds)   M=deepseek-v4.1-flash; P=2; ARM=ds41_1017; export V3_DS_VER=1017 V3_NO_WINDOW=1 ;;
esac
for g in 1 2 3; do
  export V3_GEN=$g
  V3_MODEL=$M python3 -m mem.beam answer --conds $2 --par $P || exit 1
  V3_MODEL=gpt-6-luna V3_OUT_ARM=$ARM python3 -m mem.beam judge --par 8 || exit 1
done
