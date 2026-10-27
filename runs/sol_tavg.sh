#!/bin/bash
# Time-averaged retention for sol (8000): roll, quiz, trigger, periodic rebuild every 6/12 (prebuild generated live).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6.1-sol V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse tavg --par 4 --tavg-arms quiz:3,trigger:3,prebuild:6,prebuild:12
