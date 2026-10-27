#!/bin/bash
# 10-15: GPT back on, Pro only (gpt-6-luna on 18318, arm pro6). Plus is not used.
# BEAM-100K main run (6 conversations x 20 questions; rag / lychee / lychee_roll) in parallel with the drift redo.
# V3 keeps ~7 requests in flight on Pro (V2 shares the proxy; ~20 total before 429s).
# Pro is at 95% weekly; the user allowed using it up and spending credits on experiments, so no stop line.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
( python3 -m mem.drift build --par 3 && python3 -m mem.drift answer --par 3 && python3 -m mem.drift report ) > runs/pro6_drift.log 2>&1 &
python3 -m mem.beam build --convs 6 --par 3 > runs/pro6_beam.log 2>&1 &&
python3 -m mem.beam answer --convs 6 --par 6 >> runs/pro6_beam.log 2>&1 &&
python3 -m mem.beam judge --par 6 >> runs/pro6_beam.log 2>&1 &&
python3 -m mem.beam report --convs 6 >> runs/pro6_beam.log 2>&1
wait
