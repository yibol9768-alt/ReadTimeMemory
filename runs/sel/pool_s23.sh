#!/bin/bash
# SPEC_STAGE23 v3.2 8 step 0: candidate pools + 8B teacher scores for every stage-2/3 training and inner-validation question.
# Same first stage as all pools (Qwen3-Embedding-0.6B + BM25, RRF k=60, top 600) and teacher = Qwen3-Reranker-8B official
# format (k1._rr_scores) on the top 300, i.e. pool.py --P 600 --T 300, unchanged code. Existing pools of the same questions
# (same pool.py recipe) are linked, not rebuilt. Output dir pool_s23/.
# usage: setsid nohup bash pool_s23.sh "1,2,4,5,6,7" > pool_s23.log 2>&1 &   (first GPU = embedding, the rest = 8B teacher)
set -u
cd ~/data/rtm/sel || exit 2
log(){ echo "[$(TZ=America/New_York date +%H:%M:%S)] $*"; }
GPUS=(${1//,/ }); [ ${#GPUS[@]} -ge 2 ] || { echo "need >= 2 GPUs"; exit 2; }
exec 9>/tmp/rtm_pool_s23.lock; flock -n 9 || { log "already running"; exit 2; }
IMG=docker.1ms.run/vllm/vllm-openai:deepseekv41-flash-0909; M=/home/centuser/data/models; PFX=rtm-pool23-
END=$(TZ=America/New_York date -d "today 18:30" +%s); [ $(( END - $(date +%s) )) -gt 1800 ] || { log "too close to deadline"; exit 2; }
PG=$(ps -o pgid= $$ | tr -d ' ')
cleanup(){ timeout 120 bash -c "docker ps -a --format '{{.Names}}' | grep '^$PFX' | xargs -r docker rm -f" >/dev/null 2>&1; }
( exec 9>&-; sleep $(( END - $(date +%s) )); log WATCHDOG; cleanup; kill -TERM -- -$PG ) & W=$!
trap 'S=$(pgrep -P $W); kill $W 2>/dev/null; [ -n "$S" ] && kill $S 2>/dev/null; cleanup' EXIT
free(){ ! ss -ltn | grep -q ":$1 " || { log "port $1 in use"; return 1; }; }
up(){ local name=$1 port=$2 ok=0; for k in $(seq 90); do [ "$(docker inspect -f '{{.State.Running}}' $name 2>/dev/null)" = true ] || { log "$name died"; return 1; }
  curl -sf -m3 localhost:$port/v1/models >/dev/null && { ok=1; break; }; sleep 10; done; [ $ok = 1 ]; }

mkdir -p pool_s23
# link existing pools built with the same recipe (pool.py --P 600 --T 300)
python3 - <<'EOF'
import json, os, sys
sys.path.insert(0, ".")
srcs = ["pool", "pool_lmes150", "pool_loc150", "pool_ev150", "pool_ep150"]; n = 0
for b in ["beam", "lme", "loc", "ev", "ep"]:
    for part in ("train", "inner"):
        for q in json.load(open(f"q_{part}_{b}.json")):
            dst = f"pool_s23/{q['question_id']}.json"
            if os.path.exists(dst): continue
            for s in srcs:
                p = f"{s}/{q['question_id']}.json"
                if os.path.exists(p):
                    d = json.load(open(p))
                    rr = {int(k): v for k, v in d.get("rr8", {}).items()}; top = d["fused"][:300]
                    from k1 import rounds_of as _ro; nR = len(_ro(q))
                    if (len(d["fused"]) == min(600, nR) and len(set(d["fused"])) == len(d["fused"])
                            and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < nR for i in d["fused"])) and all(i in rr and isinstance(rr[i], (int, float)) and rr[i] == rr[i] and 0 <= rr[i] <= 1 for i in top):
                        os.symlink(os.path.abspath(p), dst); n += 1
                    break
print("linked existing pools:", n)
EOF
E=${GPUS[0]}; free 18311 || exit 1
docker run -d --name ${PFX}emb --gpus "\"device=$E\"" --ipc=host --network host -v $M/qwen/Qwen3-Embedding-0.6B:/model:ro $IMG /model --served-model-name Qwen3-Embedding-0.6B --host 127.0.0.1 --port 18311 --runner pooling --convert embed --max-model-len 8192 --gpu-memory-utilization 0.9 >/dev/null || exit 1
RP=""
for g in "${GPUS[@]:1}"; do p=$((18330+g)); free $p || exit 1; bash serve_sel.sh ${PFX}$g $g $p $M/qwen/Qwen3-Reranker-8B | grep -q served || { log "START FAIL $g"; exit 1; }; RP="$RP,$p"; done; RP=${RP#,}
up ${PFX}emb 18311 || exit 1; for p in ${RP//,/ }; do g=$((p-18330)); up ${PFX}$g $p || exit 1; done; log "ready emb 18311 teacher $RP"
for b in beam lme loc ev ep; do for part in train inner; do
  python3 pool.py q_${part}_$b.json --emb-port 18311 --rr-port $RP --rr-model sel --P 600 --T 300 --par 24 --out pool_s23 > pool_s23_${part}_$b.log 2>&1 || { log "FAIL $part $b"; exit 1; }
  log "done $part $b ($(tail -1 pool_s23_${part}_$b.log))"; done; done
python3 - <<'EOF'
import json, os, sys
sys.path.insert(0, ".")
bad = []
for b in ["beam", "lme", "loc", "ev", "ep"]:
    for part in ("train", "inner"):
        for q in json.load(open(f"q_{part}_{b}.json")):
            p = f"pool_s23/{q['question_id']}.json"
            if not os.path.exists(p): bad.append(p); continue
            d = json.load(open(p))
            from k1 import rounds_of as _ro
            nR = len(_ro(q)); rr = {int(k): v for k, v in d["rr8"].items()}; top = d["fused"][:300]
            if (len(d["fused"]) != min(600, nR) or len(set(d["fused"])) != len(d["fused"])
                    or any(isinstance(i, bool) or not isinstance(i, int) or not (0 <= i < nR) for i in d["fused"])
                    or any(not (i in rr and isinstance(rr[i], (int, float)) and rr[i] == rr[i] and 0 <= rr[i] <= 1) for i in top)): bad.append(p)
print("POOLCHECK missing or incomplete:", len(bad), bad[:5])
if not bad:   # freeze: md5 of every pool file's content (symlinks resolved), keyed by question id (audit 11-02 item 4)
    import hashlib; H = {}
    for b_ in ["beam", "lme", "loc", "ev", "ep"]:
        for part in ("train", "inner"):
            for q in json.load(open(f"q_{part}_{b_}.json")):
                H[q["question_id"]] = hashlib.md5(open(f"pool_s23/{q['question_id']}.json", "rb").read()).hexdigest()
    if os.path.exists("POOL_S23_MD5.json"):   # never overwrite a frozen list: verify it
        old = json.load(open("POOL_S23_MD5.json")); diff = [k for k in set(old) | set(H) if old.get(k) != H.get(k)]
        print("POOL_S23_MD5 verify: mismatches", len(diff), diff[:5]); bad += diff
    else: json.dump(H, open("POOL_S23_MD5.json", "w")); print("wrote POOL_S23_MD5.json", len(H))
sys.exit(1 if bad else 0)
EOF
c=$?; log "POOLS23DONE code $c"; exit $c
