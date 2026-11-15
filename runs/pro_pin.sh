#!/bin/bash
# Protected-store variant (pin): restored facts kept in a 2000-char section that is never rewritten; notes get 6000.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
python3 -m mem.rehearse build --arms pin --par 2 && python3 -m mem.rehearse answer --arms pin --par 3
