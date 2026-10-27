#!/bin/bash
# Main metric with more power: 80 ledger facts per conversation asked of every arm's memory (luna, 6 conversations).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse lprobe --par 3
