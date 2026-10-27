#!/bin/bash
# DeepSeek queue (2 in flight): wait for BEAM-1M gen3 (PID 324554), then LME hindsight gens 1-3, then BEAM-100K gen3.
cd "$(dirname "$0")/.."
while kill -0 324554 2>/dev/null; do sleep 60; done
runs/lme_final.sh ds hindsight && GENS=3 runs/ds_par2.sh 100k
