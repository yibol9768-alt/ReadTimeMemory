#!/bin/bash
# LME dev gens 2-3, minimal reader prompt: note3_ly, note_ly, lychee. luna + official LME judge.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=lme
for g in 2 3; do
  V3_GEN=$g python3 -m mem.beam answer --conds note3_ly,note_ly,lychee --par 8 || exit 1
  V3_GEN=$g python3 -m mem.beam judge --par 8 || exit 1
done
