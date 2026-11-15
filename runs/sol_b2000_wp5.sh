#!/bin/bash
# WP5 inside the drift zone of the strong model: gpt-6.1-sol at 2000 characters (roll/rebuild from drift _b2000).
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6.1-sol V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse build --budget 2000 --par 2 &&
python3 -m mem.rehearse build --arms leitner --budget 2000 --par 2 &&
python3 -m mem.rehearse answer --budget 2000 --par 3 &&
python3 -m mem.rehearse answer --arms leitner --budget 2000 --par 3 &&
python3 -m mem.rehearse report --budget 2000
