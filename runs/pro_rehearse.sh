#!/bin/bash
# WP5 self-quiz rehearsal vs restudy, 6 conversations, Pro gpt-6-luna. roll/rebuild come from runs/drift/pro6.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse build --par 2 && python3 -m mem.rehearse answer --par 4 && python3 -m mem.rehearse report
