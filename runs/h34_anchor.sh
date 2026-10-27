#!/bin/bash
# H34 anchored records (mem/anchor.py) on BEAM-100K dev: one re-encoding with quotes; anc vs anc_noq vs lychee.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.anchor build --par 6 && python3 -m mem.beam answer --conds anc,anc_noq --par 8 && \
python3 -m mem.beam judge --par 8 && python3 -m mem.beam report
