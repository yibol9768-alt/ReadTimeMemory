#!/bin/bash
# After dev10.sh: Hindsight contexts (AMB, exported to ext/hindsight) read by our reader on the 4 new dev convs, 3 gens.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_MODEL=gpt-6-luna V3_PRO_STOP=101 PYTHONPATH=.
while kill -0 2377325 2>/dev/null; do sleep 60; done
for g in 1 2 3; do
  V3_GEN=$g python3 -m mem.beam answer --convs 10 --conds hindsight --par 6 && V3_GEN=$g python3 -m mem.beam judge --convs 10 --par 8
done
