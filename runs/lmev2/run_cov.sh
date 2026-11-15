#!/bin/bash
# 10-29 (DS time, spare VRAM): replay_bm25 + exhaustive absence line (replay_cov), all 451 questions.
# Readers: Qwen3.5-9B TP2 on H20 GPUs 0+1 (18021), 2+5 (18022), 6+7 (18025), util 0.13 next to DeepSeek
# (GPUs 3/4 hold another 1.3 GB process -> skipped). Then judge Qwen3.8-27B-FP8 TP4 on GPUs 0,1,2,5 (18011).
# All rtm containers are removed at the end.
cd "$(dirname "$0")/../.."; export PYTHONPATH=. V3_MODEL=Qwen3.5-9B V3_NO_WINDOW=1 V3_LME_EXPAND_MODEL=Qwen3.5-9B
R=runs/lmev2/run_lme.py; M=replay_cov
go() { V3_READER9_URL=http://127.0.0.1:$1/v1/chat/completions python3 $R answer --method $M --tag $3 --types $2 --effort medium --par 12; }
go 18021 abs,errors-gotchas q9a & go 18022 static-environment q9s & go 18025 dynamic-environment,procedure q9d & wait
cat runs/lmev2/out/q9{a,s,d}_$M.jsonl > runs/lmev2/out/q9_$M.jsonl
n=$(wc -l < runs/lmev2/out/q9_$M.jsonl); echo "[$(TZ=America/New_York date +%H:%M)] answered $n"
[ "$n" -ge 451 ] || { echo "incomplete, stop (readers kept)"; exit 1; }
rm runs/lmev2/out/q9{a,s,d}_$M.jsonl
ssh h20 'docker rm -f rtm-reader-a rtm-reader-b rtm-reader-c >/dev/null; sleep 10; docker run -d --name rtm-judge --gpus "\"device=0,1,2,5\"" --ipc=host --network host -v /home/centuser/data/models/Qwen/Qwen3___8-27B-FP8:/model:ro -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909 /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port 18011 --tensor-parallel-size 4 --gpu-memory-utilization 0.13 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 4096 --enforce-eager >/dev/null'
for i in $(seq 1 60); do curl -sf -m 5 127.0.0.1:18011/health >/dev/null && break; sleep 20; done
python3 $R judge --method $M --tag q9
ssh h20 'docker rm -f rtm-judge >/dev/null'
python3 $R report | grep -E "q9_(bm25|hybrid|replay)"
echo "[$(TZ=America/New_York date +%H:%M)] done, containers removed"
