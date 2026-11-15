#!/bin/bash
# Final latency (audit 11-02 item 10): run only when no other job uses the H20 GPUs. One model per GPU, identical vLLM
# serving flags for every model (pooling runner, max-num-seqs 256, max-num-batched-tokens 65536, prefix caching, util 0.85);
# all candidates of a query in one request; 5 warm-up, 50 timed sequential queries, then the same 50 at 16 concurrent.
# BEAM dev pools (top 600). Results in out/LATFINAL.jsonl with GPU memory used by each server.
set -u; cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
busy=$(docker ps --format '{{.Names}}' | grep '^rtm-' | grep -v '^rtm-latf-' | wc -l); [ "$busy" = 0 ] || { log "other rtm containers running: refuse"; exit 2; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; M=/home/centuser/data/models; CK=/home/centuser/data/rtm/sel/ckpt; RRD=$M/rerankers; PFX=rtm-latf-
trap 'docker ps -a --format "{{.Names}}" | grep "^$PFX" | xargs -r docker rm -f >/dev/null 2>&1' EXIT
COMMON="--served-model-name sel --host 127.0.0.1 --runner pooling --max-num-seqs 256 --max-num-batched-tokens 65536 --enable-prefix-caching --gpu-memory-utilization 0.85"
QW='--convert classify --hf_overrides {"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
one(){ local g=$1 dir=$2 tag=$3 len=$4 kind=$5 extra=$6 port=$((18460+$1))
  if [ $kind = q ]; then docker run -d --name $PFX$g --gpus "\"device=$g\"" --ipc=host --network host -v $dir:/model:ro $IMG /model $COMMON --port $port --max-model-len $len --convert classify --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' >/dev/null
  else docker run -d --name $PFX$g --gpus "\"device=$g\"" --ipc=host --network host -v $dir:/model:ro $IMG /model $COMMON --port $port --max-model-len $len >/dev/null; fi
  for k in $(seq 90); do curl -sf -m3 localhost:$port/v1/models | grep -q '"id":"sel"' && break; sleep 10; done
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  python3 lat.py beam_ex.json --pooldir pool --port $port --tag $tag --out out/LATFINAL.jsonl $extra >> latf_$tag.log 2>&1 && log "done $tag mem_MiB $mem $(tail -1 latf_$tag.log)" || log "FAIL $tag"
  docker rm -f $PFX$g >/dev/null 2>&1; }
log start
one 0 $M/qwen/Qwen3-Reranker-0.6B A0 8192 q "" & one 1 $CK/d3_e CX 8192 q "" & one 2 $M/qwen/MemReranker-4B MR4 8192 q "" & one 3 $M/qwen/Qwen3-Reranker-8B RR8 8192 q "" &
one 4 $RRD/bge-reranker-large BGEL 512 p "--plain --trunc 512" & one 5 $RRD/bge-reranker-v2-m3 BGEM3 8192 p "--plain" & one 6 $RRD/gte-reranker-modernbert-base GTE 8192 p "--plain" &
wait; log LATFINALDONE
