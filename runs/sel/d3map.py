"""D3 mapping (spec v2, frozen): ecell JSON -> E-arm knobs and question-type weights. usage: python3 d3map.py ECELL.json OUT.json"""
import json, sys
M0 = {"p_group": 0.3, "p_long": 0.3, "p_dated": 0.15, "p_greet": 0.1, "p_sib": 0.3}
e = json.load(open(sys.argv[1])); t = e["target"]
if t is None: raise SystemExit("NOT EXECUTABLE: no eligible cell")
qt, car = t; K = dict(M0)
if car == "group": K["p_group"] = min(0.6, 2 * M0["p_group"])
if car == "long": K["p_long"] = min(0.6, 2 * M0["p_long"])
out = {"target": t, "knobs": K, "weights": f"{qt}=2"}
json.dump(out, open(sys.argv[2], "w")); print(out)
