#!/bin/bash
# Evaluation power (PROGRESS 10-16 00:0x): 3 independent generations for the BEAM-100K dev comparisons.
# gen2 for anc/anc_noq/full (missing), then gen3 for every main condition.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
V3_GEN=2 python3 -m mem.beam answer --conds anc,anc_noq,full --par 6 && V3_GEN=2 python3 -m mem.beam judge --par 8
C=lychee,lychee_b20,rag,rag_b20,hindsight,anc,anc_noq,full
V3_GEN=3 python3 -m mem.beam answer --conds $C --par 6 && V3_GEN=3 python3 -m mem.beam judge --par 8
