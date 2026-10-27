#!/bin/bash
# MEME final test (method frozen 10-26 04:40: hybrid + extractor v3) on the 78 untouched episodes.
# DeepSeek reader 2 concurrent; Qwen judge on H20 GPUs 0-3 judges each condition as soon as it is answered.
# Hard stop at DEADLINE (default 08:15 ET, before the H20 training window): readers killed, judge container removed.
# Resumable: re-running continues unanswered questions (stateq run skips done rows).
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
IDS=$(cat runs/meme/test_ids.txt); M=runs/meme/data; J=Qwen3.8-27B-FP8
DEADLINE=$(TZ=America/New_York date -d "${STOP_AT:-08:15}" +%s)
log() { echo "[$(TZ=America/New_York date '+%m-%d %H:%M')] $*"; }
curl -s -m 10 -o /dev/null http://127.0.0.1:18003/v1/models || ssh -fN -o ExitOnForwardFailure=yes -L 18003:127.0.0.1:18003 h20
ssh h20 'docker ps -q -f name=rtm-judge | grep -q . || docker run -d --name rtm-judge --gpus "\"device=0,1,2,3\"" --ipc=host --network host -v /home/centuser/data/models/Qwen/Qwen3___8-27B-FP8:/model:ro -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909 /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port 18011 --tensor-parallel-size 4 --gpu-memory-utilization 0.12 --max-model-len 65536 --max-num-seqs 32 --enforce-eager >/dev/null'
curl -s -m 5 -o /dev/null http://127.0.0.1:18011/health || ssh -fN -o ExitOnForwardFailure=yes -L 18011:127.0.0.1:18011 h20
( # watchdog: at the deadline stop everything this script started and free the GPUs
  while [ $(date +%s) -lt $DEADLINE ]; do sleep 30; kill -0 $$ 2>/dev/null || exit 0; done
  log "deadline: stopping"; pkill -P $$; for p in $(pgrep -f "mem.stateq (run|judge) .*_t(nf|32)"); do kill $p; done
  ssh h20 'docker rm -f rtm-judge >/dev/null'; log "judge removed" ) &
WD=$!
JOBS=()
run() {  # $1 data file, $2 cond, $3 suffix
  python3 -m mem.stateq run --data $M/$1 --ids $IDS --cond $2 --suffix $3 --par 2 > /dev/null || return
  log "answered $2$3"
  ( until curl -sf -m 5 http://127.0.0.1:18011/health >/dev/null; do sleep 15; done
    python3 -m mem.stateq judge --cond $2 --suffix $3 --judge $J 2>&1 | grep "judge Qwen" ) &
  JOBS+=($!)
}
run nofiller.json hybrid _tnf
run nofiller.json full_fair _tnf
run nofiller.json full _tnf
run filler32k.json hybrid _t32
run filler32k.json full_fair _t32
wait "${JOBS[@]}"
kill $WD 2>/dev/null; ssh h20 'docker rm -f rtm-judge >/dev/null'
log "all done (judge removed)"
