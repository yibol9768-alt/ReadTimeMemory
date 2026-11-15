#!/bin/bash
# H29 controls: refine1, refine_ledger, same5, merge5_s2.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.selrebuild controls --par 3 &
V3_MODEL=gpt-6.1-sol python3 -m mem.selrebuild controls --par 3
wait
