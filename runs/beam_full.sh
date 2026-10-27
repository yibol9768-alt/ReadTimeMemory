#!/bin/bash
# Baseline reproduction (PROTOCOL.md): full-context reader on BEAM-100K dev (6 convs, 120 questions), luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.beam answer --conds full --par 4 && python3 -m mem.beam judge --par 6 && python3 -m mem.beam report
