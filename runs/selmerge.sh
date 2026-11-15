#!/bin/bash
# H29: merge the H28 candidates into one memory (same budget); evaluated on the same 80 facts per conversation.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.selrebuild merge --par 3 &
V3_MODEL=gpt-6.1-sol python3 -m mem.selrebuild merge --par 3
wait
