#!/bin/bash
# Experiment X (10-31, plan audit): old synthetic data with fu/bg rounds ignored and shortcut-predictable groups dropped
set -eu
cd ~/data/rtm/sel
export K1_BUDGETS=4000,16000
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909
docker rm -f rtm-qa-0 rtm-qa-1 rtm-qa-4 >/dev/null 2>&1 || true; sleep 10
timeout 7200 docker run --rm --name rtm-train-t2x --gpus '"device=0,1,4"' --ipc=host --network host -v /home/centuser/data/rtm/sel:/w -v /home/centuser/data/models:/models:ro --entrypoint torchrun $IMG --nproc_per_node 3 --master_port 29788 /w/train.py --model /models/qwen/Qwen3-Reranker-0.6B --data /w/trx.jsonl --dev /w/dvx.jsonl --out /w/ckpt/t2x --lr 1e-5 --epochs 2 --eval-every 100 --max-d 1100
bash serve_sel.sh rtm-sel0 0 18341 /home/centuser/data/rtm/sel/ckpt/t2x
bash serve_sel.sh rtm-sel1 1 18342 /home/centuser/data/rtm/sel/ckpt/t2x
sleep 20; for k in $(seq 80); do curl -s -m 3 localhost:18341/v1/models >/dev/null && curl -s -m 3 localhost:18342/v1/models >/dev/null && break; sleep 15; done
python3 infer.py beam_ex.json --port 18341,18342 --tag T2X_H --cond 0 --par 8
docker rm -f rtm-sel0 rtm-sel1 >/dev/null 2>&1 || true
python3 txsum.py T1_H,T2_H,T2X_H
echo T2XDONE
