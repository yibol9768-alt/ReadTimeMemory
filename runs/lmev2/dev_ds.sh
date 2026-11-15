#!/bin/bash
# LME-V2 Small dev pass with DeepSeek reader (no thinking), after the MEME test; hard stop at 08:20 ET.
cd "$(dirname "$0")/../.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
while kill -0 ${WAIT_PID:-0} 2>/dev/null; do sleep 30; done
END=$(TZ=America/New_York date -d 08:20 +%s)
( while [ $(date +%s) -lt $END ]; do sleep 30; kill -0 $$ 2>/dev/null || exit 0; done; pkill -P $$; echo "deadline stop" ) &
for T in abs dynamic-environment; do for m in none uistore bm25 hybrid; do
  [ $(date +%s) -lt $END ] || break 2
  python3 runs/lmev2/run_lme.py answer --method $m --tag ds_dev --types $T --effort low --par 2
done; done
python3 runs/lmev2/run_lme.py report
