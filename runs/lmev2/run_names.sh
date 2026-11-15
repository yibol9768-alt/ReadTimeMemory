#!/bin/bash
# 10-29 pilot: replay_bm25 + name check (only names stated in the question, found/not found), 40 questions
# (16 abs, 12 static, 12 dynamic; seed 1029). Readers 18021 / 18025 (GPUs 0+1, 6+7).
cd "$(dirname "$0")/../.."; export PYTHONPATH=. V3_MODEL=Qwen3.5-9B V3_NO_WINDOW=1 V3_LME_EXPAND_MODEL=Qwen3.5-9B
R=runs/lmev2/run_lme.py
V3_READER9_URL=http://127.0.0.1:18021/v1/chat/completions python3 $R answer --method replay_names --tag pna --ids-file runs/lmev2/pilot_names_ids.txt --types abs --effort medium --par 12 &
V3_READER9_URL=http://127.0.0.1:18025/v1/chat/completions python3 $R answer --method replay_names --tag pnn --ids-file runs/lmev2/pilot_names_ids.txt --types static-environment,dynamic-environment --effort medium --par 12 &
wait; echo "[$(TZ=America/New_York date +%H:%M)] answered $(cat runs/lmev2/out/pn?_replay_names.jsonl | wc -l)"
