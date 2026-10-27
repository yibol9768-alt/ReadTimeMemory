#!/bin/bash
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
V3_BEAM_SPLIT=1m python3 -m mem.beam answer --convs 5 --conds lychee_cf --par 8 && V3_BEAM_SPLIT=1m python3 -m mem.beam judge --convs 5 --par 8
V3_BEAM_SPLIT=100k python3 -m mem.beam answer --convs 10 --conds cf_hindsight --par 8 && V3_BEAM_SPLIT=100k python3 -m mem.beam judge --convs 10 --par 8
