#!/bin/bash
# Step 1 (PLAN_NEXT v2, 11-02): one pointwise ranking per model on the five dev sets with the same pools, packing and
# infer.py, so every model has a full 'rank' field (R@10, MRR) and recall at 1k/2k/4k/16k. New tags P1<model>_<bench>.
# usage: setsid nohup bash p1.sh qwen|plain [LIMIT] > p1_<mode>.log 2>&1 &
#   qwen  = Qwen3-reranker-format models (A0, T1, CruxRank stage-1 seeds 0-2, MemReranker-4B, Qwen3-Reranker-8B), GPUs 0-7
#   plain = BGE-large, BGE-v2-m3, gte-modernbert, jina-v2 (generic cross-encoders, no Qwen template), GPUs 0-3
#   LIMIT>0 = smoke on the first LIMIT questions of each set (tags get suffix S). Modes never run concurrently (flock).
# Codex audit 11-02 fixes: lock + own containers only; deadline kills the whole process group and exits non-zero;
# server start and identity (port free before start, mounted model dir) are verified; LIMIT validated; outputs are written under a temporary
# tag and published only after p1check.py accepts them.
set -u
cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
MODE=${1:-}; LIM=${2:-0}
[[ "$MODE" == qwen || "$MODE" == plain || "$MODE" == official ]] || { echo "usage: p1.sh qwen|plain|official [LIMIT]"; exit 2; }
[[ "$LIM" =~ ^[0-9]{1,6}$ ]] || { echo "LIMIT must be an integer 0..999999"; exit 2; }
SFX=""; [ "$LIM" -gt 0 ] && SFX=S
exec 9>/tmp/rtm_p1_$MODE.lock; flock -n 9 || { log "another p1.sh $MODE is running"; exit 2; }   # modes use disjoint containers and ports (qwen 18400-18407 with prefix rtm-p1q-, plain 18400-18403 only while qwen GPUs 0-3 are free; free() refuses busy ports)
unset MASK_DATE
export K1_BUDGETS=1000,2000,4000,16000
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
M=/home/centuser/data/models; CK=/home/centuser/data/rtm/sel/ckpt; RRD=$M/rerankers
PFX=rtm-p1${MODE:0:1}-          # container names owned by this run
END=$(TZ=America/New_York date -d "today 18:30" +%s); [ $(( END - $(date +%s) )) -gt 900 ] || { log "too close to the 18:30 deadline"; exit 2; }
PG=$(ps -o pgid= $$ | tr -d ' ')
cleanup(){ timeout 120 bash -c "docker ps -a --format '{{.Names}}' | grep '^$PFX' | xargs -r docker rm -f" >/dev/null 2>&1; }
# watchdog: closes the lock fd first; at the deadline it stops containers (bounded by timeout) and then the whole group
( exec 9>&-; sleep $(( END - $(date +%s) )); log "WATCHDOG: deadline reached, stopping"; cleanup; kill -TERM -- -$PG ) & W=$!
# stop the watchdog shell first (else its sleep ending would fire the deadline branch), then its orphaned sleep
trap 'S=$(pgrep -P $W); kill $W 2>/dev/null; [ -n "$S" ] && kill $S 2>/dev/null; cleanup' EXIT

free(){ ! ss -ltn 2>/dev/null | grep -q ":$((18400+$1)) " || { log "port $((18400+$1)) already in use, refusing to start"; return 1; }; }
qwen(){ docker rm -f $PFX$1 >/dev/null 2>&1; free $1 || return 1
  bash serve_sel.sh $PFX$1 $1 $((18400+$1)) $2 | grep -q served; }                     # Qwen3-reranker format
plain(){ docker rm -f $PFX$1 >/dev/null 2>&1; free $1 || return 1
  docker run -d --name $PFX$1 --gpus "\"device=$1\"" --ipc=host --network host -v $2:/model:ro $IMG /model --served-model-name sel --host 127.0.0.1 --port $((18400+$1)) --runner pooling --max-model-len $3 --max-num-seqs 256 --gpu-memory-utilization 0.6 >/dev/null; }
up(){ local g=$1 dir=$2
  local ready=0
  for k in $(seq 90); do
    [ "$(docker inspect -f '{{.State.Running}}' $PFX$g 2>/dev/null)" = true ] || { log "container $PFX$g not running"; docker logs --tail 20 $PFX$g 2>&1 | sed 's/^/    /'; return 1; }
    curl -sf -m 3 localhost:$((18400+g))/v1/models | grep -q '"id":"sel"' && { ready=1; break; }; sleep 10; done
  [ $ready = 1 ] || { log "NOTREADY $PFX$g"; return 1; }
  src=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/model"}}{{.Source}}{{end}}{{end}}' $PFX$g)
  [ "$(realpath $src)" = "$(realpath $dir)" ] || { log "IDENTITY mismatch $PFX$g: $src vs $dir"; return 1; }
  ss -ltnp 2>/dev/null | grep -q ":$((18400+g)) " || { log "port $((18400+g)) not listening"; return 1; }; }
down(){ docker rm -f $PFX$1 >/dev/null 2>&1; }
inf(){ local ports=$1 tag=$2 extra=$3
  for b in lmes150:q_lmes150.json:pool_lmes150 ep150:q_ep150.json:pool_ep150 ev150:q_ev150.json:pool_ev150 loc150:q_loc150.json:pool_loc150 H:beam_ex.json:pool; do
    IFS=: read n f pd <<< "$b"; t=P1${tag}${SFX}_$n; tmp=${t}_tmp; rm -f ctx_$tmp.jsonl
    python3 infer.py $f --pooldir $pd --port $ports --tag $tmp --cond 0 --par 16 --limit $LIM $extra > inf_$t.log 2>&1 || { log "FAIL $t (see inf_$t.log)"; return 1; }
    python3 p1check.py ctx_$tmp.jsonl $f $pd $LIM >> inf_$t.log 2>&1 || { log "CHECK FAIL $t"; tail -1 inf_$t.log; return 1; }
    mv -f ctx_$tmp.jsonl ctx_$t.jsonl || { log "PUBLISH FAIL $t"; return 1; }; log "done $t"; done; }
one(){ local g=$1 kind=$2 dir=$3 tag=$4 extra=$5 len=${6:-0} c
  ls $dir/*.safetensors >/dev/null 2>&1 && [ -e $dir/config.json ] || { log "SKIP $tag: weights not present in $dir"; return 0; }
  if [ $kind = q ]; then qwen $g $dir; else plain $g $dir $len; fi || { log "START FAIL $tag"; down $g; return 1; }
  up $g $dir && inf $((18400+g)) $tag "$extra"; c=$?; down $g; return $c; }
two(){ local g1=$1 g2=$2 dir=$3 tag=$4 extra=${5:-} c   # one model on two GPUs (4B, 8B)
  qwen $g1 $dir && qwen $g2 $dir || { log "START FAIL $tag"; down $g1; down $g2; return 1; }
  up $g1 $dir && up $g2 $dir && inf $((18400+g1)),$((18400+g2)) $tag "$extra"; c=$?; down $g1; down $g2; return $c; }

log "start $MODE limit $LIM"
if [ $MODE = qwen ]; then
  ( one 0 q $M/qwen/Qwen3-Reranker-0.6B A0 "" && one 0 q $CK/t1 T1 "" ) & p0=$!
  ( one 1 q $CK/d3_e E0 "" ) & p1=$!
  ( one 2 q $CK/d3_e_s1 E1 "" ) & p2=$!
  ( one 3 q $CK/d3_e_s2 E2 "" ) & p3=$!
  ( two 4 5 $M/qwen/MemReranker-4B MR4 ) & p4=$!
  ( two 6 7 $M/qwen/Qwen3-Reranker-8B RR8 ) & p5=$!
  PIDS="$p0 $p1 $p2 $p3 $p4 $p5"
elif [ $MODE = official ]; then   # audit 11-02: each Qwen-format baseline also with its own official-style instruction (A0O etc.)
  ( one 0 q $M/qwen/Qwen3-Reranker-0.6B A0O "--official" ) & p0=$!
  ( two 1 2 $M/qwen/MemReranker-4B MR4O "--official" ) & p1=$!
  ( one 5 q $M/qwen/Qwen3-Reranker-8B RR8O "--official" ) & p2=$!
  PIDS="$p0 $p1 $p2"
else
  ( one 0 p $RRD/bge-reranker-large BGEL "--plain --trunc 512" 512 ) & p0=$!
  ( one 1 p $RRD/bge-reranker-v2-m3 BGEM3 "--plain" 8192 ) & p1=$!
  ( one 2 p $RRD/gte-reranker-modernbert-base GTE "--plain" 8192 ) & p2=$!
  ( one 3 p $RRD/jina-reranker-v2-base-multilingual JINA "--plain --trunc 1024" 1024 ) & p3=$!
  PIDS="$p0 $p1 $p2 $p3"
fi
X=0; for p in $PIDS; do wait $p || X=1; done; log "P1DONE $MODE code $X"; exit $X
