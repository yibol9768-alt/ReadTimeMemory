#!/bin/bash
# WP3 on LoCoMo, Pro gpt-6-luna (arm pro6): Lychee-lite encode, roll-ups, then lychee vs lychee_roll on the
# 80-question sample + 30 held-out multi-hop questions. rec_sess baselines for pro6 already exist (runs/locomo).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.lychee build --per-cat 50 --par 3 &&
python3 -m mem.rollup build --par 3 &&
python3 -m mem.rollup answer --par 4 &&
python3 -m mem.locomo_exp judge --par 4 &&
python3 -m mem.rollup report
