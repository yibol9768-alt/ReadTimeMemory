#!/bin/bash
# LME-V2 Small, official reader Qwen3.5-9B (thinking on, H20 GPUs 0-3 rtm-reader), all 451 questions.
# Store v2 (iframe bids parsed) + query expansion by the same 9B model (also used by the BM25 baseline).
cd "$(dirname "$0")/../.."; export PYTHONPATH=. V3_MODEL=Qwen3.5-9B V3_NO_WINDOW=1 V3_LME_EXPAND_MODEL=Qwen3.5-9B
for m in uistore bm25 hybrid; do python3 runs/lmev2/run_lme.py answer --method $m --tag q9 --effort medium --par 32; done
python3 runs/lmev2/run_lme.py report
echo "[$(TZ=America/New_York date +%H:%M)] q9 answered"
