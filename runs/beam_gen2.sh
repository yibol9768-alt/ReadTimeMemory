#!/bin/bash
# Second independent generation (V3_GEN=2: every LLM call - query plan, reader, judge - re-sampled) on BEAM-100K dev,
# to measure generation noise before trusting any 3-7 point difference (PROTOCOL 5).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_GEN=2
C=${1:-lychee,lychee_b20,lyx_b20,rag,rag_b20,hindsight}
python3 -m mem.beam answer --conds $C --par 6 && python3 -m mem.beam judge --par 6 && python3 -m mem.beam report
