#!/bin/bash
# Time-averaged retention (memory replayed after every session from the cache; probed every 2 sessions), luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse tavg --par 4
