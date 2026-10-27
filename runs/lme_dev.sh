#!/bin/bash
# LongMemEval-S dev (62 questions stratified by type, seed 0; mem/beam.py lme_split), luna reader + official LME judge.
# Same conditions / reader prompt as BEAM (question date given to the reader as "Current date").
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=lme
C=${1:-full,rag,lychee,lychee_b20,lyx_b20,rag_b20}
python3 -m mem.beam build --par 16 && \
python3 -m mem.beam answer --conds $C --par 8 && python3 -m mem.beam judge --par 8 && python3 -m mem.beam report
