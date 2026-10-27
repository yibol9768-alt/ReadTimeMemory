#!/bin/bash
# MEME hybrid v4 (set-valued slots: OBS mode add/replace). Pilot 20 (_p1c) then test 78 (_tnfc, post-hoc fix). DeepSeek 2 concurrent.
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
M=runs/meme/data
python3 -m mem.stateq run --data $M/nofiller.json --ids $(cat runs/meme/pilot_ids.txt) --cond hybrid --suffix _p1c --par 2
echo "[$(TZ=America/New_York date +%H:%M)] pilot done"
python3 -m mem.stateq run --data $M/nofiller.json --ids $(cat runs/meme/test_ids.txt) --cond hybrid --suffix _tnfc --par 2
echo "[$(TZ=America/New_York date +%H:%M)] test done"
