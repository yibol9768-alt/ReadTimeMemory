#!/bin/bash
# MEME pilot 2 (10-26): hybrid method + robustness (implicit-paraphrase dependencies, 32k filler), DeepSeek reader
# (2 concurrent), then Qwen3.8-27B judge on H20 GPUs 0-3 (started here, removed at the end).
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
IDS=$(cat runs/meme/pilot_ids.txt); M=runs/meme/data
log() { echo "[$(TZ=America/New_York date '+%m-%d %H:%M')] $*"; }
curl -s -m 10 -o /dev/null http://127.0.0.1:18003/v1/models || ssh -fN -o ExitOnForwardFailure=yes -L 18003:127.0.0.1:18003 h20
[ -s $M/para_p1.json ] || python3 -m mem.meme_perturb --data $M/nofiller.json --ids $IDS --out $M/para_p1.json
log "perturb ready"
for c in state hybrid; do python3 -m mem.stateq run --data $M/nofiller.json --ids $IDS --cond $c --suffix _p1b --par 2; done   # _p1b: extractor also records the source value named in a rule
for c in full_fair state hybrid; do python3 -m mem.stateq run --data $M/para_p1.json --ids $IDS --cond $c --suffix _para --par 2; done
log "para done"
for c in full_fair state hybrid; do python3 -m mem.stateq run --data $M/filler32k.json --ids $IDS --cond $c --suffix _f32 --par 2; done
log "reader done"
ssh h20 'docker rm -f rtm-judge >/dev/null 2>&1; docker run -d --name rtm-judge --gpus "\"device=0,1,2,3\"" --ipc=host --network host -v /home/centuser/data/models/Qwen/Qwen3___8-27B-FP8:/model:ro -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909 /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port 18011 --tensor-parallel-size 4 --gpu-memory-utilization 0.12 --max-model-len 65536 --max-num-seqs 32 --enforce-eager >/dev/null'
curl -s -m 5 -o /dev/null http://127.0.0.1:18011/health || ssh -fN -o ExitOnForwardFailure=yes -L 18011:127.0.0.1:18011 h20
until curl -sf -m 5 http://127.0.0.1:18011/health >/dev/null; do sleep 15; done
log "judge up"
J=Qwen3.8-27B-FP8
for c in state hybrid; do python3 -m mem.stateq judge --cond $c --suffix _p1b --judge $J; done
for s in _para _f32; do for c in full_fair state hybrid; do python3 -m mem.stateq judge --cond $c --suffix $s --judge $J; done; done
ssh h20 'docker rm -f rtm-judge >/dev/null'
log "all done (judge removed)"
