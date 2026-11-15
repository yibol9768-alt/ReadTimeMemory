#!/bin/bash
# D3 v8 worlds: 980-997 full (M0); 998-1001 one plan, paired fixed/E renders (spec v8). Pools on all reranker GPUs. Containers removed at exit.
cd ~/data/rtm/sel; O=ws7; mkdir -p ws7 ws7x ws7e
log(){ echo "[$(TZ=America/New_York date +%H:%M)] $*"; }
trap 'docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o rtm-env-rr rtm-env-rr2 rtm-rrx0 rtm-rrx1 rtm-rrx2 rtm-rrx3 rtm-rrx4 >/dev/null 2>&1; docker stop rtm-env-emb >/dev/null 2>&1' EXIT
bash wup.sh | tail -1 | grep -q UPOK || { log FAIL up; exit 1; }; log up
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302,Qwen3.8-27B-FP8@18303,Qwen3.8-27B-FP8@18304,Qwen3.8-27B-FP8@18305 --gem gemma-4-31b-it@18321 --oss gpt-oss-120b@18322'
M='{"p_group": 0.3, "p_long": 0.3, "p_dated": 0.15, "p_greet": 0.1, "p_sib": 0.3}'; E='{"p_group": 0.3, "p_long": 0.6, "p_dated": 0.15, "p_greet": 0.1, "p_sib": 0.3}'
st(){ d=$1; i=$2; k=$3; shift 3; for s in "$@"; do python3 world.py $s --out $d --start $i --n 1 --rounds 240 --par 24 --knobs "$k" $A > $d/$i.$s.log 2>&1 || { log "FAIL $d $i $s"; return 1; }; done; log "$d $i done"; }
pr(){ i=$1; st ws7 $i "$M" plan || return 1; for a in x e; do [ $a = x ] && k=$M || k=$E; python3 pairknob.py ws7/w0$(printf %04d $i).json ws7$a/w0$(printf %04d $i).json "$k" >> pairknob.log 2>&1 || { log FAIL pair; return 1; }; done; }
for i in 998 999 1000 1001; do pr $i || exit 1; done; log plans paired; cat pairknob.log
J=(); for i in $(seq 980 997); do J+=("ws7 $i M"); done; for i in 998 999 1000 1001; do J+=("ws7x $i M" "ws7e $i E"); done
P=""; n=0; for j in "${J[@]}"; do set -- $j; if [ $3 = M ]; then k=$M; else k=$E; fi; if [ $1 = ws7 ]; then st ws7 $2 "$k" plan render verify build gates & else st $1 $2 "$k" render verify build gates & fi; P="$P $!"; n=$((n+1)); if [ $((n%6)) = 0 ]; then for p in $P; do wait $p || exit 1; done; P=""; log "batch $((n/6))"; fi; done; for p in $P; do wait $p || exit 1; done; log generated
docker rm -f rtm-env-q0 rtm-env-q1 rtm-env-q2 rtm-env-q3 rtm-env-q4 rtm-env-g rtm-env-o >/dev/null 2>&1
python3 wfilter.py ws7 980,981 0 ws7ct && python3 wfilter.py ws7 $(seq -s, 982 989) 0 ws7ce && python3 wfilter.py ws7 $(seq -s, 990 997) 0 ws7v && python3 wfilter.py ws7x 998,999,1000,1001 0 ws7x && python3 wfilter.py ws7e 998,999,1000,1001 0 ws7e || { log FAIL filter; exit 1; }
docker start rtm-env-emb >/dev/null; bash ~/data/rtm/env/up_rr.sh; bash up_rrx.sh
R=18312,18313,18370,18371,18372,18373,18374
for k in $(seq 80); do ok=1; for p in 18311 ${R//,/ }; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && break; sleep 15; done; [ $ok = 1 ] || { log FAIL rr; exit 1; }
for t in ws7ct ws7ce ws7v ws7x ws7e; do python3 pool.py env_$t.json --rr-port $R --T 600 --par 32 --out pool_$t > pool_$t.log 2>&1 || { log "FAIL pool $t"; exit 1; }; log "pool $t"; done
log D3V8WDONE
