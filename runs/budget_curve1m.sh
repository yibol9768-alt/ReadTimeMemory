#!/bin/bash
# pipeline x budget curve on 1M, gen 1, mem0 prompt, no conflict check: Lychee records vs Hindsight memories at 8K / 20K / 60K / ~124K. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m V3_ALL_ROUTES=1
python3 -m mem.beam answer --convs 5 --conds lychee~m0,hs8n~m0,hs20~m0,lychee_b60~m0,hs60~m0,lychee_b120~m0 --par 8 && python3 -m mem.beam judge --convs 5 --par 8
