#!/bin/bash
# A3 on one GPU; the container is removed at the end whatever happens
cd ~/data/rtm/sel; C=rtm-qa-0; G=${1:-0}; P=18301
trap 'docker rm -f $C >/dev/null 2>&1' EXIT
docker run -d --name $C --gpus "\"device=$G\"" --network host --ipc host -v /home/centuser/data/models/Qwen/Qwen3.8-27B-FP8:/model:ro docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909 /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port $P --enable-prefix-caching --max-model-len 131072 --gpu-memory-utilization 0.9 --max-num-seqs 128 >/dev/null || exit 1
for i in $(seq 1 90); do curl -s -m 3 localhost:$P/v1/models | grep -q Qwen && break; sleep 10; done
curl -s -m 3 localhost:$P/v1/models | grep -q Qwen || { echo NOTREADY; exit 1; }
echo READY $(TZ=America/New_York date +%H:%M); python3 a3.py --port $P; python3 a3.py --sum; echo END $(TZ=America/New_York date +%H:%M)
