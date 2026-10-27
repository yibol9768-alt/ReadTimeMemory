#!/bin/bash
# Qwen3.8-27B (H20 GPUs 0-3, rtm-judge) judges the DeepSeek-reader answers of MEME pilot 1, all four conditions.
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
for c in state full full_fair full_rules; do python3 -m mem.stateq judge --cond $c --suffix _p1 --judge Qwen3.8-27B-FP8; done
echo "judge done $(TZ=America/New_York date +%H:%M)"
