#!/bin/bash
# MEME pilot 1 with DeepSeek-V4.1-Flash reader (2 concurrent, DeepSeek time). 4 conds incl. fair controls.
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
D=/tmp/claude-1000/-mnt-c-Users-liuyibo-Desktop-auto-llm-driving/9d982558-f9f7-476e-89c7-66eeb913ad1e/scratchpad/meme/nofiller.json
IDS=$(cat runs/meme/pilot_ids.txt)
for c in state full full_fair full_rules; do python3 -m mem.stateq run --data $D --ids $IDS --cond $c --suffix _p1 --par 2; done
echo "reader done $(TZ=America/New_York date +%H:%M)"
