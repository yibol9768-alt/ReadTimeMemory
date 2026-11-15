#!/bin/bash
# Model servers for world generation: 27B on GPUs 0-2, gemma on 6, gpt-oss on 7.
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; M=/home/centuser/data/models
S="--max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 256"
run(){ name=$1 g=$2 model=$3 served=$4 port=$5; docker rm -f $name >/dev/null 2>&1; docker run -d --name $name --gpus "\"device=$g\"" --ipc=host --network host -v $model:/model:ro $IMG /model --served-model-name $served --host 127.0.0.1 --port $port --enable-prefix-caching $S >/dev/null; }
for g in 0 1 2; do run rtm-env-q$g $g $M/Qwen/Qwen3.8-27B-FP8 Qwen3.8-27B-FP8 $((18301+g)); done
run rtm-env-g 6 $M/google/gemma-4-31b-it gemma-4-31b-it 18321
run rtm-env-o 7 $M/openai-mirror/gpt-oss-120b gpt-oss-120b 18322
for k in $(seq 120); do ok=1; for p in 18301 18302 18303 18321 18322; do curl -s -m 3 localhost:$p/v1/models >/dev/null || ok=0; done; [ $ok = 1 ] && { echo UPOK; exit 0; }; sleep 15; done; echo UPFAIL; exit 1
