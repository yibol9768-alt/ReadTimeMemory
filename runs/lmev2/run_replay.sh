#!/bin/bash
# 10-28: replay memory (reconstructed ordered pages + replayed steps), alone and + BM25. Qwen3.5-9B readers on H20
# GPUs 1,2,5,6 (one replica each, ports 1802x); judge Qwen3.8-27B on GPU 7 (18011). Containers removed at the end.
cd "$(dirname "$0")/../.."; export PYTHONPATH=. V3_MODEL=Qwen3.5-9B V3_NO_WINDOW=1 V3_LME_EXPAND_MODEL=Qwen3.5-9B
R=runs/lmev2/run_lme.py; NA=static-environment,dynamic-environment,procedure,errors-gotchas
go() { V3_READER9_URL=http://127.0.0.1:$1/v1/chat/completions python3 $R answer --method $2 --tag $4 --types $3 --effort medium --par 48; }
go 18021 replay $NA q9n & go 18022 replay abs q9a & go 18025 replay_bm25 $NA q9n & go 18026 replay_bm25 abs q9a & wait
for m in replay replay_bm25; do cat runs/lmev2/out/q9n_$m.jsonl runs/lmev2/out/q9a_$m.jsonl > runs/lmev2/out/q9_$m.jsonl && rm runs/lmev2/out/q9n_$m.jsonl runs/lmev2/out/q9a_$m.jsonl; done
echo "[$(TZ=America/New_York date +%H:%M)] answered"
until curl -sf -m 5 http://127.0.0.1:18011/health >/dev/null; do sleep 15; done
for m in replay replay_bm25; do python3 $R judge --method $m --tag q9; done
python3 $R report | grep q9
echo "[$(TZ=America/New_York date +%H:%M)] done"
