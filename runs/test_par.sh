#!/bin/bash
# All remaining held-out runs in parallel (48GB WSL), staggered 20 s so the cache loads do not peak together.
cd "$(dirname "$0")/.."; L=runs/test_logs; J=()
st() { nohup bash -c "$2" > $L/$1.log 2>&1 & J+=($!); echo "$1 $!"; sleep 20; }
st sol_100k_judge3 "export PYTHONPATH=. V3_TEST=1 V3_PROXY_PORT=18318 V3_BEAM_SPLIT=100k V3_GEN=3 V3_MODEL=gpt-6-luna V3_OUT_ARM=pro_gpt-6.1-sol V3_PRO_STOP=101 H39_AB=abstention,contradiction_resolution,event_ordering,information_extraction,instruction_following,knowledge_update,multi_session_reasoning,preference_following,summarization,temporal_reasoning; python3 -m mem.beam judge --convs 10 --par 16"
for g in 1 2 3; do st sol_1m_g$g "R=gpt-6.1-sol GENS=$g P=12 bash runs/test_reader.sh 1m"; done
for s in 100k 1m lme; do st astra_$s "R=gpt-6-astra GENS=1 P=12 STOP=99 bash runs/test_reader.sh $s"; done
for s in 100k 1m; do st rr_$s "for g in 1 2 3; do bash runs/test_rr.sh $s \$g || exit 1; done"; done
wait
echo all done $(TZ=America/New_York date +%H:%M)
