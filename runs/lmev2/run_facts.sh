#!/bin/bash
# 10-29 pilot (dev split only, 105 ids: 45 abs + 60 answerable, seed 1030):
#   A1 replay2_bm25  = replay+bm25 on page-class keyframes (record numbers masked)
#   A2 replay2_facts = A1 + program-computed fact sheet (reader compiles 1-4 census queries; program counts/lists)
# Reader Qwen3.5-9B TP2 on H20 GPUs 6+7 (18025). A0 replay_bm25 and A4 hybrid reuse the existing full runs.
cd "$(dirname "$0")/../.."; export PYTHONPATH=. V3_MODEL=Qwen3.5-9B V3_NO_WINDOW=1 V3_LME_EXPAND_MODEL=Qwen3.5-9B
R=runs/lmev2/run_lme.py; I=runs/lmev2/pilot_facts_ids.txt; U=${READER_PORT:-18025}
for m in replay2_facts replay2_bm25; do
  V3_READER9_URL=http://127.0.0.1:$U/v1/chat/completions python3 $R answer --method $m --tag pf --ids-file $I --effort medium --par 14
done
echo "[$(TZ=America/New_York date +%H:%M)] answered"
