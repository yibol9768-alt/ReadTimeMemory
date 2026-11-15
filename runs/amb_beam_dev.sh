#!/bin/bash
# Baseline reproduction in AMB (PROTOCOL.md): BEAM-100K dev units (our convs 0,10,14,16,17,18 = AMB ids 1,11,15,17,18,19).
# Answer, judge and Hindsight extraction all on gpt-6-luna. Usage: amb_beam_dev.sh <memory> [units]
source "$(dirname "$0")/amb_env.sh"
MEM=$1; UNITS=${2:-"1 11 15 17 18 19"}
cd ~/v3venvs/amb
for u in $UNITS; do
  amb run --dataset beam --split 100k --memory $MEM --unit $u --output-dir ~/v3venvs/amb_out --name ${MEM}_u$u
done
