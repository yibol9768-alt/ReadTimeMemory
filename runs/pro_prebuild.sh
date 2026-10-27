#!/bin/bash
# Same-cost baseline: roll with a full rebuild from the ledger every 3, 6 or 12 sessions (luna, 6 conversations).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
for e in 3 6 12; do
  python3 -m mem.rehearse build --arms prebuild --every $e --par 2 &&
  python3 -m mem.rehearse answer --arms prebuild --every $e --par 3
done
python3 -m mem.rehearse report
