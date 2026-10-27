#!/bin/bash
# V3 on the Plus account (fingerprinted as gpt-5.6-luna), small sample: 20 questions per category (the first 20
# of the same seed-0 shuffle, so a subset of the 50-per-category set) and the four conditions that decide the idea.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18319 PYTHONPATH=.
set -e
C=rec_sess,once,hyb_mq,view+hyb_mq
python3 -m mem.locomo_exp answer --per-cat 20 --par 8 --conds $C
python3 -m mem.locomo_exp judge --par 8
python3 -m mem.locomo_exp report --per-cat 20
