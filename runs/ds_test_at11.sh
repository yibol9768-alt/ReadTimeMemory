#!/bin/bash
# DeepSeek-V4.1-Flash (H20 v9) reader on the three test sets, gen 1, 2 concurrent; starts in the H20 window (11:05 ET).
cd "$(dirname "$0")/.."
T=$(TZ=America/New_York date -d "tomorrow 11:05" +%s); [ $(TZ=America/New_York date +%H) -lt 11 ] && T=$(TZ=America/New_York date -d "today 11:05" +%s)
while [ $(date +%s) -lt $T ]; do sleep 300; done
for s in 100k 1m lme; do NOJUDGE=1 R=deepseek-v4.1-flash V3_DS_VER=v9 GENS=1 P=2 bash runs/test_reader.sh $s > runs/test_logs/ds_$s.log 2>&1; echo "ds $s exit $? $(TZ=America/New_York date +%H:%M)"; done
