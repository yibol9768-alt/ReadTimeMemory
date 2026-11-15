#!/bin/bash
# BEAM-1M dev (rows 0,1; 40 questions), luna, same reader + official judge as 100K.
# hindsight = AMB/Hindsight-1.0.0 retrieved contexts (exported to runs/beam1m/pro6/ext/hindsight), re-read by our reader.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m
C=${1:-hindsight,rag,lychee,lychee_b20,lyx_b20_nonb,lyx_b20,rag_b20}
python3 -m mem.beam build --par 2 && python3 -m mem.beam answer --conds $C --par 8 && \
python3 -m mem.beam judge --par 8 && python3 -m mem.beam report
