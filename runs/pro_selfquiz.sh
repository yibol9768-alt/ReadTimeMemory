#!/bin/bash
# WP5 extra arm: MemDream-style self-probing (probes written from the memory itself), same 6 conversations.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse build --arms selfquiz --par 2 && python3 -m mem.rehearse answer --arms selfquiz --par 3
