#!/bin/bash
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m
python3 -m mem.erran run --conds lychee,hindsight,rag --par 8 && python3 -m mem.erran report --conds lychee,hindsight,rag
