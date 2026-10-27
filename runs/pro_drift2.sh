#!/bin/bash
# WP4 drift, 3 more conversations (0,5,8) to double the sample. Pro gpt-6-luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.drift build --convs 0,5,8 --par 3 && python3 -m mem.drift answer --convs 0,5,8 --par 3 &&
python3 -m mem.drift report --convs 0,2,4,5,7,8
