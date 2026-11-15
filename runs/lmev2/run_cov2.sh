#!/bin/bash
# Rebalance of run_cov.sh (04:16): q9a / q9s answer processes keep running; dynamic resumes on 18025, procedure on 18022.
cd "$(dirname "$0")/../.."; export PYTHONPATH=. V3_MODEL=Qwen3.5-9B V3_NO_WINDOW=1 V3_LME_EXPAND_MODEL=Qwen3.5-9B
R=runs/lmev2/run_lme.py; M=replay_cov
go() { V3_READER9_URL=http://127.0.0.1:$1/v1/chat/completions python3 $R answer --method $M --tag $3 --types $2 --effort medium --par $4; }
go 18025 dynamic-environment q9d 12 & go 18022 procedure q9p 8 & go 18021 procedure q9p2 4 &
wait
while kill -0 3188842 2>/dev/null || kill -0 3188844 2>/dev/null; do sleep 30; done
python3 - <<'PY'
import json
seen, rows = set(), []
for t in "asdp":
    for t2 in (t, t + "2"):
        try:
            for l in open(f"runs/lmev2/out/q9{t2}_replay_cov.jsonl"):
                r = json.loads(l)
                if r["id"] not in seen: seen.add(r["id"]); rows.append(r)
        except FileNotFoundError: pass
open("runs/lmev2/out/q9_replay_cov.jsonl", "w").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
print("merged", len(rows))
PY
n=$(wc -l < runs/lmev2/out/q9_$M.jsonl); echo "[$(TZ=America/New_York date +%H:%M)] answered $n"
[ "$n" -ge 451 ] || { echo "incomplete, stop (readers kept)"; exit 1; }
ssh h20 'docker rm -f rtm-reader-a rtm-reader-b rtm-reader-c >/dev/null; sleep 10; docker run -d --name rtm-judge --gpus "\"device=0,1,2,5\"" --ipc=host --network host -v /home/centuser/data/models/Qwen/Qwen3___8-27B-FP8:/model:ro -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909 /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port 18011 --tensor-parallel-size 4 --gpu-memory-utilization 0.13 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 4096 --enforce-eager >/dev/null'
for i in $(seq 1 60); do curl -sf -m 5 127.0.0.1:18011/health >/dev/null && break; sleep 20; done
python3 $R judge --method $M --tag q9
ssh h20 'docker rm -f rtm-judge >/dev/null'
python3 $R report | grep -E "q9_(bm25|hybrid|replay)"
echo "[$(TZ=America/New_York date +%H:%M)] done, containers removed"
