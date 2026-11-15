#!/bin/bash
# D3 stage 2a: W20 on the feedback worlds and the localization control world; signal E; frozen mapping. Containers removed at exit.
while pgrep -f "^bash poolx.sh" >/dev/null; do sleep 15; done   # extra rerankers on GPUs 0-4 must be gone first
cd ~/data/rtm/sel; log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-sel0 rtm-sel1 rtm-sel2 rtm-sel3 >/dev/null 2>&1' EXIT
C=/home/centuser/data/rtm/sel/ckpt/w20; for g in 0 1 2 3; do bash serve_sel.sh rtm-sel$g $g $((18341+g)) $C; done
for k in $(seq 80); do ok=1; for p in 18341 18342 18343 18344; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done
for t in ws6f ws6p0; do rm -f ctx_W20_$t.jsonl; K1_BUDGETS=1000,2000,4000 python3 infer.py env_$t.json --pooldir pool_$t --port 18341,18342,18343,18344 --tag W20_$t --cond 0 --par 16 > inf_W20_$t.log 2>&1 || { log "FAIL infer $t"; exit 1; }; done
python3 ecell.py env_ws6f.json pool_ws6f 1000,2000,4000 W20_ws6f --json d3_signal.json > d3_signal.txt || { log FAIL signal; exit 1; }
python3 d3map.py d3_signal.json d3_map.json > d3_map.txt || { log FAIL map; exit 1; }
python3 ecell.py env_ws6p0.json pool_ws6p0 1000,2000,4000 W20_ws6p0 --min-n 5 --json d3_pc_before.json > d3_pc_before.txt && python3 ecell.py env_ws6p0.json pool_ws6p0 1000,2000,4000 W20_ws6p0:demote --min-n 5 --json d3_pc_after.json > d3_pc_after.txt || { log FAIL pc; exit 1; }
python3 pcgate.py loc d3_pc_before.json d3_pc_after.json > d3_pc_loc.txt; r=$?; cat d3_pc_loc.txt; [ $r = 0 ] || { log "PROCESS INVALID: localization control failed"; exit 3; }
log D3ADONE
