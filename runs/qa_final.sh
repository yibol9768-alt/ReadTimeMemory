#!/bin/bash
# LoCoMo questions (50 per conversation) on base / merge5 / atomic / atomic2 final memories; waits for atomic2.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
until ! pgrep -f "mem.atomic build" >/dev/null; do sleep 30; done
V3_MODEL=gpt-6-luna python3 -m mem.qa_final run --par 5 &
V3_MODEL=gpt-6.1-sol python3 -m mem.qa_final run --par 5
wait
