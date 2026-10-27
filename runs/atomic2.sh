#!/bin/bash
# H31 v2: notes grouped under a date header per session, overlong notes rewritten instead of cut.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.atomic build --par 3 --v 2 &
V3_MODEL=gpt-6.1-sol python3 -m mem.atomic build --par 3 --v 2
wait
