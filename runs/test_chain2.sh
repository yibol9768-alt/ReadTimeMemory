#!/bin/bash
# After test_chain.sh: 1M route + 3 gens once the 1M build is done; LME 3 gens once the LME Hindsight contexts exist.
cd "$(dirname "$0")/.." && export PYTHONPATH=.
( while kill -0 758503 2>/dev/null; do sleep 60; done     # test_chain.sh ends when both the 1M build and LME Hindsight end
  grep -q "build_1m exit 0" runs/test_logs/chain.log || { echo "1m build failed"; exit 1; }
  V3_TEST=1 V3_BEAM_SPLIT=1m V3_MODEL=gpt-6-luna V3_PROXY_PORT=18318 V3_PRO_STOP=101 python3 runs/h35c_route.py --convs 10 > runs/test_logs/route_1m.log 2>&1 &&
  runs/test_beam_3gen.sh 1m > runs/test_logs/beam_1m_3gen.log 2>&1; echo "1m 3gen exit $?" ) &
( while kill -0 758503 2>/dev/null; do sleep 60; done
  grep -q "hs_lme exit 0" runs/test_logs/chain.log && [ -s runs/lme_test/pro6/ext/hindsight/ctx.jsonl ] || { echo "lme hindsight failed"; exit 1; }
  runs/test_lme_3gen.sh > runs/test_logs/lme_3gen.log 2>&1; echo "lme 3gen exit $?" ) &
wait
