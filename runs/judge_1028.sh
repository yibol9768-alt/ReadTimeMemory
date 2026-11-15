#!/bin/bash
# 10-28: Qwen3.8-27B judge on H20 GPUs 4,5 (window). LME-V2 q9 LLM-judged questions (4 methods) + MEME hybrid v4 (set slots). Judge removed at the end.
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_NO_WINDOW=1
curl -s -m 5 -o /dev/null http://127.0.0.1:18011/health || ssh -fN -o ExitOnForwardFailure=yes -L 18011:127.0.0.1:18011 h20
until curl -sf -m 5 http://127.0.0.1:18011/health >/dev/null; do sleep 15; done
for m in uistore bm25 hybrid none; do V3_MODEL=Qwen3.5-9B python3 runs/lmev2/run_lme.py judge --method $m --tag q9; done
V3_MODEL=Qwen3.5-9B python3 runs/lmev2/run_lme.py report | grep q9
export V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9
for s in _p1c _tnfc; do python3 -m mem.stateq judge --cond hybrid --suffix $s --judge Qwen3.8-27B-FP8 2>&1 | grep "judge Qwen"; done
ssh h20 'docker rm -f rtm-judge >/dev/null'
echo "[$(TZ=America/New_York date +%H:%M)] judged, judge removed"
