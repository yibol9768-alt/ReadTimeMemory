#!/bin/bash
# T2q: same synthetic data as T2, 0.25 epoch (overfitting check); <30 min on GPUs 2,3
set -eu
cd ~/data/rtm/sel
export K1_BUDGETS=4000,16000
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
timeout 3600 docker run --rm --name rtm-train-t2q --gpus '"device=2,3"' --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node 2 --master_port 29777 /w/train.py --model /models/qwen/Qwen3-Reranker-0.6B --data /w/tr_env.jsonl --dev /w/dv_env.jsonl --out /w/ckpt/t2q --lr 1e-5 --epochs 0.25 --eval-every 100 --max-d 1100
bash serve_sel.sh rtm-sel2 2 18343 /home/centuser/data/rtm/sel/ckpt/t2q
bash serve_sel.sh rtm-sel3 3 18344 /home/centuser/data/rtm/sel/ckpt/t2q
sleep 20; for k in $(seq 80); do curl -s -m 3 localhost:18343/v1/models >/dev/null && curl -s -m 3 localhost:18344/v1/models >/dev/null && break; sleep 15; done
python3 infer.py beam_ex.json --port 18343,18344 --tag T2Q_H --cond 0 --par 8
docker rm -f rtm-sel2 rtm-sel3 >/dev/null 2>&1 || true
python3 txsum.py T1_H,T2_H,T2Q_H
echo T2QDONE
