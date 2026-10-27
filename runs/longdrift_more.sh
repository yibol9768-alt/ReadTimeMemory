#!/bin/bash
# Speed-up for the horizon axis: sol convs 2,3,4 in a separate process (longdrift.sh does 0,1 first, then skips done files).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
V3_MODEL=gpt-6.1-sol python3 -m mem.longdrift build --split 1M --convs 2,3,4 --par 3
