"""Question files for stage 2/3 (SPEC_STAGE23 v3.2): q_train_<bench>.json and q_inner_<bench>.json with exactly the ids of
DATA_FREEZE_STAGE23.json (checked against its md5 list first). Runs no model. usage: python3 stage_qfiles.py"""
import hashlib, json, sys
fr = json.load(open("DATA_FREEZE_STAGE23.json"))
for k, p in fr["files"].items():
    if hashlib.md5(open(p, "rb").read()).hexdigest() != fr["md5"][k]: sys.exit(f"INVALID md5 changed: {k} {p}")
SEL = {}
SRC = {"beam": ["beam500k", "beam100k", "beam1m_b6_15"], "lme": ["lme"], "loc": ["loc"], "ev": ["ev_t01", "ev_t02"], "ep": ["ep200"]}
for b, keys in SRC.items():
    Q = {}
    for k in keys:
        for q in json.load(open(fr["files"][k])): Q[q["question_id"]] = q
    for part in ("train", "inner"):
        ids = fr["bench"][b][part]; miss = [i for i in ids if i not in Q]
        if miss: sys.exit(f"INVALID {b} {part}: {len(miss)} ids not found")
        json.dump([Q[i] for i in ids], open(f"q_{part}_{b}.json", "w"))
        print(b, part, len(ids))
    # selection subset (SPEC v3.3): at most 100 frozen inner questions per benchmark, seed 20261102
    import random as _r
    inner = list(fr["bench"][b]["inner"]); _r.Random(f"20261102-{b}").shuffle(inner); sel = sorted(inner[:100])
    json.dump([Q[i] for i in sel], open(f"q_innersel_{b}.json", "w")); SEL[b] = sel; print(b, "innersel", len(sel))
json.dump(SEL, open("INNERSEL_STAGE23.json", "w"))
