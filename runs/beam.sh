#!/bin/bash
# 10-15 H20 window: BEAM-100K, 6 conversations x 20 questions, rag / lychee / lychee_roll, DeepSeek end to end.
# Waits for 11:00 US Eastern (runs alongside wp3_wp4.sh); every step resumes from the cache if cut off at 20:00.
cd "$(dirname "$0")/.." && export V3_MODEL=deepseek-v4.1-flash V3_NO_GPT=1 PYTHONPATH=.
until [ "$(TZ=America/New_York date +%H%M)" -ge 1103 ] && [ "$(TZ=America/New_York date +%H)" -lt 20 ]; do sleep 120; done
python3 -m mem.beam build --convs 6 --par 4 &&
python3 -m mem.beam answer --convs 6 --par 4 &&
python3 -m mem.beam judge --par 6 &&
python3 -m mem.beam report --convs 6
