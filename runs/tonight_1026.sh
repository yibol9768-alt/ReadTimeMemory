#!/bin/bash
# 10-26 night (DeepSeek time 19:35 -> 08:15 next day): finish MEME test (32k full_fair), then LME-V2 Small dev pass
# with DeepSeek reader on abstention + dynamic questions (4 methods), then Qwen judges LME's LLM-judged questions.
cd "$(dirname "$0")/.."; export PYTHONPATH=. V3_MODEL=deepseek-v4.1-flash V3_DS_VER=v9 V3_NO_WINDOW=1
log() { echo "[$(TZ=America/New_York date '+%m-%d %H:%M')] $*"; }
T=$(TZ=America/New_York date -d "today 19:35" +%s); while [ $(date +%s) -lt $T ]; do sleep 60; done
curl -s -m 10 -o /dev/null http://127.0.0.1:18003/v1/models || ssh -fN -o ExitOnForwardFailure=yes -L 18003:127.0.0.1:18003 h20
until curl -s -m 20 -H "Authorization: Bearer $(grep -oP 'DEEPSEEK_KEY=\K\S+' ~/auto-llm-driving/deepseek-key.env)" http://127.0.0.1:18003/v1/models | grep -q deepseek; do sleep 60; done
log "DeepSeek up"
STOP_AT="tomorrow 08:15" runs/meme_test.sh
log "meme test resumed run finished"
END=$(TZ=America/New_York date -d "tomorrow 08:15" +%s)
for T in abs dynamic-environment; do for m in none uistore bm25 hybrid; do
  [ $(date +%s) -lt $END ] || break 2
  python3 runs/lmev2/run_lme.py answer --method $m --tag ds_dev --types $T --effort low --par 2
done; done
log "lme answered"
ssh h20 'docker rm -f rtm-judge >/dev/null 2>&1; docker run -d --name rtm-judge --gpus "\"device=0,1,2,3\"" --ipc=host --network host -v /home/centuser/data/models/Qwen/Qwen3___8-27B-FP8:/model:ro -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909 /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port 18011 --tensor-parallel-size 4 --gpu-memory-utilization 0.12 --max-model-len 65536 --max-num-seqs 32 --enforce-eager >/dev/null'
curl -s -m 5 -o /dev/null http://127.0.0.1:18011/health || ssh -fN -o ExitOnForwardFailure=yes -L 18011:127.0.0.1:18011 h20
until curl -sf -m 5 http://127.0.0.1:18011/health >/dev/null; do sleep 15; done
for m in none uistore bm25 hybrid; do [ $(date +%s) -lt $END ] && python3 runs/lmev2/run_lme.py judge --method $m --tag ds_dev; done
ssh h20 'docker rm -f rtm-judge >/dev/null'
python3 runs/lmev2/run_lme.py report
log "all done (judge removed)"
