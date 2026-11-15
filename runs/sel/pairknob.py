"""Paired structural render (D3 v8). Same plan; speaker and long marks come from uniforms keyed by (world, fact id) only, so arms differ only where one threshold is higher (common random numbers).
usage: python3 pairknob.py PLAN_WORLD.json OUT_WORLD.json KNOBS_JSON"""
import json, random, sys
sys.path.insert(0, "."); from world import GROUP_KINDS, LEGACY
w = json.load(open(sys.argv[1])); K = {**LEGACY, **json.loads(sys.argv[3])}; i = w["world"]
if K["p_sib"] != w["knobs"]["p_sib"] or K["p_dated"] != w["knobs"]["p_dated"]: raise SystemExit("INVALID: only p_group / p_long may change")
for k in ("rounds", "verified"): w.pop(k, None)
ids = [f["id"] for f in w["facts"]]
if len(set(ids)) != len(ids): raise SystemExit("INVALID: duplicate fact ids")
for f in w["facts"]:
    f.pop("speaker", None); f.pop("long", None); r = random.Random(f"pair{i}|{f['id']}"); ug, ul, who = r.random(), r.random(), r.choice(w["team"])
    if f["kind"] in GROUP_KINDS and ug < K["p_group"]: f["speaker"] = who
    if ul < K["p_long"]: f["long"] = True
w["knobs"] = K; json.dump(w, open(sys.argv[2], "w"))
print("pairknob", i, "speakers", sum("speaker" in f for f in w["facts"]), "long", sum("long" in f for f in w["facts"]))
