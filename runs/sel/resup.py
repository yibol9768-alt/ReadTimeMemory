"""Recompute support after the over-broad value rule (11-01), no model calls. A support round is dropped when it was there only because a gold value's
tokens appeared in it: value without the entity, and not an update predecessor, and not sharing a gold category. usage: python3 resup.py DIR ENVFILE"""
import glob, json, os, re, sys
d, env = sys.argv[1:3]
def norm(t): return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()   # same as world.py
idx = lambda st: int(st.split("#")[1])
def fix(q, w, T, F, st):
    R = w["rounds"]; gold = [idx(x) for x in q["gold_stages"]]; gs = set(gold); keep = []
    com = set.intersection(*[set(R[k].get("cats", [])) | {F[R[k]["fid"]]["cat"]} for k in gold])
    slots = {(F[R[k]["fid"]]["entity"], norm(F[R[k]["fid"]]["attr"])) for k in gold}
    for s in q["support_stages"]:
        j = idx(s); r = R[j]; f = F[r["fid"]]; carrier = False; strict = False
        for k in gold:
            g = F[R[k]["fid"]]; v = set(norm(g["value"]).split()); e = set(norm(g["entity"]).split())
            if (len(v) >= 2 or any(c.isdigit() for c in "".join(v))) and v <= T[j]:
                carrier = True; strict |= e <= T[j]
        base = (q["question_type"] == "knowledge_update" and (f["entity"], norm(f["attr"])) in slots) or (q["question_type"] in ("aggregation", "ordering") and r["kind"] == "ev" and (com & (set(r.get("cats_any", [])) | {f["cat"]})))
        if not carrier or strict or base: keep.append(s)
    return keep
new = {}
for f in sorted(glob.glob(f"{d}/q*.json")):
    i = f[-10:-5]; w = json.load(open(f"{d}/w{i}.json")); Fx = {x["id"]: x for x in w["facts"]}
    T = [set(norm(r["user"] + " " + r["assistant"]).split()) for r in w["rounds"]]
    Q = json.load(open(f))
    for q in Q: q["support_stages"] = new[q["question_id"]] = fix(q, w, T, Fx, None)
    json.dump(Q, open(f, "w"))
E = json.load(open(env))
for q in E: q["support_stages"] = new[q["question_id"]]
json.dump(E, open(env, "w")); print("rewrote", len(new), len(E))
