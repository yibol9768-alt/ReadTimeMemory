#!/bin/bash
# Test-triggered rebuild (tests as a drift monitor, rebuild only when the miss rate > 0.4), luna and sol at 8000,
# then the ledger-probe metric for every arm of both models.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.rehearse build --arms trigger --par 3 &
V3_MODEL=gpt-6.1-sol python3 -m mem.rehearse build --arms trigger --par 3 &
wait
V3_MODEL=gpt-6-luna python3 -m mem.rehearse lprobe --par 3 &
V3_MODEL=gpt-6.1-sol python3 -m mem.rehearse lprobe --par 3 &
wait
