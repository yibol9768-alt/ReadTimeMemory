#!/bin/bash
# H32: compiled memory over time (append notes, level drops recompress from the ledger), time-averaged retention.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.compiled build --par 2
