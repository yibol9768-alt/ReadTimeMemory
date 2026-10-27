#!/bin/bash
# BEAM-1M dev (BEAM 1M rows 0,1 = AMB units 1,2) in AMB via the self-hosted Hindsight 1.0.0 server; luna everywhere.
source "$(dirname "$0")/amb_env.sh"
export BEAM_DATA_PATH=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/V3/data/beam/beam1m_dev_amb.json
MEM=${1:-hindsight-http}
cd ~/v3venvs/amb
for u in ${UNITS:-1 2}; do
  amb run --dataset beam --split 1m --memory $MEM --unit $u --output-dir ~/v3venvs/amb_out --name ${MEM}_1m_u$u
done
