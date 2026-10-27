#!/bin/bash
# Test-driven memory, scheduled version: every ledger record gets a probe; Leitner scheduling, K=12 per check.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse build --arms leitner --par 3 && python3 -m mem.rehearse answer --arms leitner --par 4
