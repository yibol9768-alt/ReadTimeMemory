#!/bin/bash
# Error analysis of BEAM-100K dev baselines (luna as the presence checker).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.erran run --conds lychee,hindsight,rag,lightmem,full --par 6
