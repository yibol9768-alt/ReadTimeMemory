#!/bin/bash
# CCR step 1 (docs/refine/FINAL_PROPOSAL.md): slot-keyed, non-adjudicating ledger on BEAM-100K dev; then chain audit.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.ccr build --par 6 && python3 -m mem.ccr audit && python3 -m mem.ccr audit2
