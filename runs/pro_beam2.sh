#!/bin/bash
# BEAM part of pro_beam.sh, relaunched after the source-id fix in lychee.parse (10-15 01:50). Encodings come from the cache.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.beam build --convs 6 --par 3 &&
python3 -m mem.beam answer --convs 6 --par 6 &&
python3 -m mem.beam judge --par 6 &&
python3 -m mem.beam report --convs 6
