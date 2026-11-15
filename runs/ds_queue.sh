#!/bin/bash
# DeepSeek queue, 2 in flight total (user 10-19): wait for the LME run, then BEAM gen3 at 1M then 100K.
cd "$(dirname "$0")/.."
while kill -0 303607 2>/dev/null; do sleep 60; done
GENS=3 runs/ds_par2.sh 1m && GENS=3 runs/ds_par2.sh 100k
