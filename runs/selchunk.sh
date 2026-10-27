#!/bin/bash
# H30 static test: chunked compression of the ledger (k = 2, 4, 8) at the final checkpoint, same 80 eval facts.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.selrebuild chunk --par 3 &
until ! pgrep -f "selrebuild controls" >/dev/null; do sleep 30; done
V3_MODEL=gpt-6.1-sol python3 -m mem.selrebuild chunk --par 3
wait
