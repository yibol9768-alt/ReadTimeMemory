#!/bin/bash
# Memory-limited chaining (15 GB host): 1M build after the LME build ends; LME Hindsight after the 1M Hindsight ends.
cd "$(dirname "$0")/.." && export PYTHONPATH=.
( while kill -0 741442 2>/dev/null; do sleep 60; done
  V3_TEST=1 V3_BEAM_SPLIT=1m V3_MODEL=gpt-6-luna V3_PROXY_PORT=18318 V3_PRO_STOP=101 python3 -m mem.beam build --convs 10 --par 5 > runs/test_logs/build_1m.log 2>&1
  echo "build_1m exit $?" ) &
( while kill -0 757307 2>/dev/null; do sleep 60; done
  runs/hs_lme_test.sh > runs/test_logs/hs_lme_driver.log 2>&1
  echo "hs_lme exit $?" ) &
wait
