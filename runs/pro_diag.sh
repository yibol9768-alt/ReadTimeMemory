#!/bin/bash
# G1 diagnosis (10-15): H19 durability, H18 K sweep (luna), H20 sol replication of WP4+WP5.
cd "$(dirname "$0")/.." && export V3_PROXY_PORT=18318 V3_PRO_STOP=101 PYTHONPATH=.
# H19: replay quiz arm from cache, re-ask restored facts in final memories
V3_MODEL=gpt-6-luna python3 -m mem.rehearse durability > runs/pro6_durability.log 2>&1 &
# H18: more probes per check
( V3_MODEL=gpt-6-luna python3 -m mem.rehearse build --arms quiz --k 36 --par 2 &&
  V3_MODEL=gpt-6-luna python3 -m mem.rehearse answer --arms quiz --k 36 --par 3 &&
  V3_MODEL=gpt-6-luna python3 -m mem.rehearse build --arms quiz --k 72 --par 2 &&
  V3_MODEL=gpt-6-luna python3 -m mem.rehearse answer --arms quiz --k 72 --par 3 ) > runs/pro6_ksweep.log 2>&1 &
# H20: gpt-6.1-sol, same 6 conversations: roll/rebuild (drift), then restudy/quiz
( V3_MODEL=gpt-6.1-sol python3 -m mem.drift build --convs 0,2,4,5,7,8 --par 2 &&
  V3_MODEL=gpt-6.1-sol python3 -m mem.drift answer --convs 0,2,4,5,7,8 --par 3 &&
  V3_MODEL=gpt-6.1-sol python3 -m mem.rehearse build --par 2 &&
  V3_MODEL=gpt-6.1-sol python3 -m mem.rehearse answer --par 3 &&
  V3_MODEL=gpt-6.1-sol python3 -m mem.rehearse report ) > runs/sol_wp45.log 2>&1 &
wait
