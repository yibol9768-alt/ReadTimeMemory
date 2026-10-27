#!/bin/bash
# Test set: Hindsight 1.0.0 banks (self-hosted server :8888, luna extraction) via AMB, retrieval only.
# $1 = 100k|1m. 100K test units = our convs 1,4,5,6,7,8,12,13,15,19 (+1); 1M test = rows 5-14 = AMB units (conversation_id) 6-15 of beam1m_test_amb.json.
source "$(dirname "$0")/amb_env.sh"; R=/home/liuyibo/auto-llm-driving/ReadTimeMemory
if [ "$1" = 100k ]; then export BEAM_DATA_PATH=$R/data/beam/beam100k_amb.json; UNITS=${UNITS:-"2 5 6 7 8 9 13 14 16 20"}; N=hs_test_u
else export BEAM_DATA_PATH=$R/data/beam/beam1m_test_amb.json; UNITS=${UNITS:-"6 7 8 9 10 11 12 13 14 15"}; N=hs_test_1m_u; fi
one() {
  [ -n "$(ls ~/v3venvs/amb_out/beam/$N$2/retrieval/*.json 2>/dev/null)" ] && return 0
  cd ~/v3venvs/amb && timeout 14400 amb run --dataset beam --split $1 --memory hindsight-http --mode retrieval --llm openai \
    --unit $2 --output-dir ~/v3venvs/amb_out --name $N$2 > $R/runs/test_logs/hs_$1_u$2.log 2>&1
}
export -f one; export R N
for u in $UNITS; do echo "$1 $u"; done | xargs -P ${HP:-3} -n 2 bash -c 'one $0 $1'
