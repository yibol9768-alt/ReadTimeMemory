# serve_sel.sh NAME GPU PORT MODELDIR
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
docker rm -f $1 >/dev/null 2>&1
docker run -d --name $1 --gpus "\"device=$2\"" --ipc=host --network host -v $4:/model:ro $IMG /model --served-model-name sel --host 127.0.0.1 --port $3 --runner pooling --convert classify --max-model-len 8192 --max-num-batched-tokens 65536 --max-num-seqs 256 --enable-prefix-caching --gpu-memory-utilization 0.85 --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' >/dev/null && echo served $1
