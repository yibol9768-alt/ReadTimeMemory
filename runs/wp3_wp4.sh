#!/bin/bash
# 10-15 H20 window: WP3 roll-up records (vs Lychee-lite, 80-question sample + 30 held-out multi-hop) and
# WP4 drift redone on DeepSeek. DeepSeek end to end, no GPT. Waits for 11:00 US Eastern; stops itself after 20:00.
cd "$(dirname "$0")/.." && export V3_MODEL=deepseek-v4.1-flash V3_NO_GPT=1 PYTHONPATH=.
until [ "$(TZ=America/New_York date +%H%M)" -ge 1102 ] && [ "$(TZ=America/New_York date +%H)" -lt 20 ]; do sleep 120; done
( python3 -m mem.drift build --par 3 && python3 -m mem.drift answer --par 3 ) > runs/ds41_drift.log 2>&1 &
python3 -m mem.rollup build --par 5 > runs/ds41_rollup.log 2>&1
python3 -m mem.rollup answer --par 6 >> runs/ds41_rollup.log 2>&1
wait
python3 -m mem.locomo_exp judge --par 6
python3 -m mem.rollup report
python3 -m mem.drift report
