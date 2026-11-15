#!/bin/bash
# BEAM 100K then 1M (sequential), LME in parallel.
cd "$(dirname "$0")/.."
(bash runs/test_sol.sh 100k > runs/test_logs/sol_100k.log 2>&1; echo "sol 100k exit $?" >> runs/test_logs/sol_chain.log
 bash runs/test_sol.sh 1m > runs/test_logs/sol_1m.log 2>&1; echo "sol 1m exit $?" >> runs/test_logs/sol_chain.log) &
bash runs/test_sol.sh lme > runs/test_logs/sol_lme.log 2>&1; echo "sol lme exit $?" >> runs/test_logs/sol_chain.log
wait
