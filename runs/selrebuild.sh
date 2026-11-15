#!/bin/bash
# Test-selected rebuild pilot: 5 candidate memories per rebuild, picked on 40 ledger facts, evaluated on 80 others.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.selrebuild build --par 1 &
V3_MODEL=gpt-6.1-sol python3 -m mem.selrebuild build --par 1
wait
for m in gpt-6-luna gpt-6.1-sol; do echo "== $m"; V3_MODEL=$m python3 -m mem.selrebuild report; done
