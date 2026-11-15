#!/bin/bash
# wait for >= 3 GB available memory (daily-ai-news ffmpeg jobs), then run asp_full~m0 on 1M. luna.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=. V3_BEAM_SPLIT=1m
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 3 ]; do sleep 30; done
python3 -m mem.beam answer --convs 5 --conds asp_full~m0 --par 8 && python3 -m mem.beam judge --convs 5 --par 8
