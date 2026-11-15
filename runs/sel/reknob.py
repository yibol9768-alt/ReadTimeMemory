"""E-arm world from a fixed-mix world (D3): same plan (project, entities, facts, siblings), only speaker and long marks redrawn with the E knobs.
usage: python3 reknob.py SRC_WORLD.json OUT_WORLD.json KNOBS_JSON"""
import json, random, sys
sys.path.insert(0, "."); from world import GROUP_KINDS, LEGACY
w = json.load(open(sys.argv[1])); K = {**LEGACY, **json.loads(sys.argv[3])}; i = w["world"]; r = random.Random(f"knobE{i}")
for k in ("rounds", "verified"): w.pop(k, None)
if K["p_sib"] != w["knobs"]["p_sib"] or K["p_dated"] != w["knobs"]["p_dated"]: raise SystemExit("INVALID: E knobs may only change p_group / p_long")
for f in w["facts"]:
    f.pop("speaker", None); f.pop("long", None)
    if K["p_group"] and f["kind"] in GROUP_KINDS and r.random() < K["p_group"]: f["speaker"] = r.choice(w["team"])
    if K["p_long"] and r.random() < K["p_long"]: f["long"] = True
w["knobs"] = K; json.dump(w, open(sys.argv[2], "w"))
print("reknob", i, "speakers", sum("speaker" in f for f in w["facts"]), "long", sum("long" in f for f in w["facts"]))
