#!/bin/bash
# Hindsight 1.0.0 (self-hosted, luna extraction) on the 62 LME-S dev questions, one AMB unit (= one bank) per question,
# 3 in parallel (8 ran the 15 GB host out of memory); then collect the retrieved contexts into runs/lme/pro6/ext/hindsight/ctx.jsonl (qid = b<i>-<ability>-0).
cd "$(dirname "$0")/.." && R=$PWD
source runs/amb_env.sh; export LONGMEMEVAL_DATA_PATH=$R/data/longmemeval/longmemeval_s.json
one() {
  [ -s ~/v3venvs/amb_out/longmemeval/hs_lme_$1/retrieval/s.json ] && return 0
  cd ~/v3venvs/amb && timeout 7200 amb run --dataset longmemeval --split s --memory hindsight-http --mode retrieval --llm openai \
    --unit $1 --output-dir ~/v3venvs/amb_out --name hs_lme_$1 > $R/runs/hs_lme/$1.log 2>&1
}
export -f one; export R; mkdir -p runs/hs_lme
xargs -a runs/lme_dev_ids.txt -P ${HP:-3} -I{} bash -c 'one {}'
cd $R && PYTHONPATH=. V3_BEAM_SPLIT=lme python3 - <<'PY'
import json, os
from mem.beam import load_lme
convs = load_lme(); out = "runs/lme/pro6/ext/hindsight"; os.makedirs(out, exist_ok=True); n = 0
with open(out + "/ctx.jsonl", "w") as f:
    for i, c in enumerate(convs):
        p = os.path.expanduser(f"~/v3venvs/amb_out/longmemeval/hs_lme_{c['id']}/retrieval/s.json")
        if not os.path.exists(p): print("missing", c["id"]); continue
        ctx = json.load(open(p))["results"][0]["context"]
        if not ctx: print("empty", c["id"]); continue
        ab = list(c["probing"])[0]
        f.write(json.dumps({"qid": f"b{i}-{ab}-0", "context": ctx}) + "\n"); n += 1
print("wrote", n)
PY
