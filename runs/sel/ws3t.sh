#!/bin/bash
cd ~/data/rtm/sel; IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; M=/home/centuser/data/models
S="--max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 256"
trap 'docker rm -f rtm-env-q0 rtm-env-q1 >/dev/null 2>&1' EXIT
for g in 0 1; do docker rm -f rtm-env-q$g >/dev/null 2>&1; docker run -d --name rtm-env-q$g --gpus "\"device=$g\"" --ipc=host --network host -v $M/Qwen/Qwen3.8-27B-FP8:/model:ro $IMG /model --served-model-name Qwen3.8-27B-FP8 --host 127.0.0.1 --port $((18301+g)) --enable-prefix-caching $S >/dev/null; done
for k in $(seq 60); do curl -s -m 3 localhost:18301/v1/models >/dev/null && curl -s -m 3 localhost:18302/v1/models >/dev/null && break; sleep 10; done
mkdir -p ws3t; cp ws1/w00012.json ws3t/
A='--q27 Qwen3.8-27B-FP8@18301,Qwen3.8-27B-FP8@18302 --gem x@1 --oss x@1'
python3 world.py build --out ws3t --start 12 --n 1 --rounds 240 $A || { echo FAIL build; exit 1; }
python3 world.py gates --out ws3t --start 12 --n 1 --rounds 240 --par 24 $A || { echo FAIL gates; exit 1; }
python3 -c "
import json,collections
q=json.load(open('ws3t/q00012.json')); print(collections.Counter((x['question_type'],x['scope']) for x in q).most_common())
for x in [y for y in q if y['scope']=='none'][:4]: print(x['question'][:120], len(x['gold_stages']))
g={r['qid']:r for r in json.load(open('ws3t/g00012.json'))}
import statistics as st
for sc in ('date','none'):
    v=[g[x['question_id']] for x in q if x['scope']==sc]; print(sc,len(v),'k',round(st.mean(r['k'] for r in v),1),'blind',round(st.mean(r['blind'] for r in v),2),'suff',round(st.mean(r['suff'] for r in v),2),'del',round(st.mean(r.get('del',0) for r in v),2))
"
echo WS3TDONE
