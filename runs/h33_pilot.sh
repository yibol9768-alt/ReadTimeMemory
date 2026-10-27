#!/bin/bash
# H33 pilot (PROGRESS 2026-10-15 night): records as an index into the raw dialogue, BEAM-100K dev 120 q, luna.
# Ladder: lychee (8K) -> lychee_b20 (same method, 20K) -> lyx_b20_nonb (+ source turns of every record, 1500 chars)
#         -> lyx_b20 (+ the turns before/after each source); control rag_b20 (raw retrieval at 20K).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
C=lychee_b20,lyx_b20_nonb,lyx_b20,rag_b20
python3 -m mem.beam contexts --conds $C && python3 -m mem.beam answer --conds $C --par 8 && \
python3 -m mem.beam judge --par 8 && python3 -m mem.beam report
