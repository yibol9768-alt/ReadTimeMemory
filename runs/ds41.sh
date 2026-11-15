#!/bin/bash
# WP1 on the H20 DeepSeek (deepseek-v4.1-flash), DeepSeek end to end (write, read, judge). No GPT calls.
# 80-question sample (20 per category). Stops by itself outside 11:00-20:00 US Eastern.
cd "$(dirname "$0")/.." && export V3_MODEL=deepseek-v4.1-flash V3_NO_GPT=1 PYTHONPATH=.
python3 -m mem.lychee build --par 6 > runs/ds41_lychee_build.log 2>&1 &
python3 -m mem.locomo_exp build --kinds rec_sess --per-cat 20 --par 3 > runs/ds41_recsess_build.log 2>&1
python3 -m mem.locomo_exp ranks --per-cat 20 --par 6
python3 -m mem.locomo_exp answer --per-cat 20 --par 6 --conds rec_sess,hyb_mq
wait
python3 -m mem.lychee answer --per-cat 20 --par 6
python3 -m mem.locomo_exp judge --par 6
python3 -m mem.locomo_exp report --per-cat 20
