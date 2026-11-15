#!/bin/bash
# Re-judge the three MEME pilot-2 conditions whose judge run hit a truncated Qwen reply; judge removed at the end.
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
ssh h20 'docker rm -f rtm-judge >/dev/null 2>&1; docker run -d --name rtm-judge --gpus "\"device=0,1,2,3\"" --ipc=host --network host -v /home/centuser/data/models/Qwen/Qwen3___8-27B-FP8:/model:ro -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909 /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port 18011 --tensor-parallel-size 4 --gpu-memory-utilization 0.12 --max-model-len 65536 --max-num-seqs 32 --enforce-eager >/dev/null'
curl -s -m 5 -o /dev/null http://127.0.0.1:18011/health || ssh -fN -o ExitOnForwardFailure=yes -L 18011:127.0.0.1:18011 h20
until curl -sf -m 5 http://127.0.0.1:18011/health >/dev/null; do sleep 15; done
J=Qwen3.8-27B-FP8
python3 -m mem.stateq judge --cond state --suffix _p1b --judge $J
python3 -m mem.stateq judge --cond hybrid --suffix _para --judge $J
python3 -m mem.stateq judge --cond state --suffix _f32 --judge $J
ssh h20 'docker rm -f rtm-judge >/dev/null'
echo "rejudge done $(TZ=America/New_York date +%H:%M)"
