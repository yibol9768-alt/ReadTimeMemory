#!/bin/bash
# H31 static test: compiled memory (every record compressed to budget/n characters, all packed), luna and sol.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6-luna python3 -m mem.atomic build --par 3 &
V3_MODEL=gpt-6.1-sol python3 -m mem.atomic build --par 3
wait
