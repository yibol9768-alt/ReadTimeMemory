"""D4 SPEC §12.7 margin check: on H4j (odd worlds) group-chat questions, area(oracle) - area(E0) >= 10 and n_group >= 80.
area = mean recall over budgets 1k,2k,4k,8k,16k; oracle = gold rounds first (chronological), then fused pool order, then the rest."""
import json, statistics as st, sys, collections
sys.path.insert(0, ".")
from k1 import rounds_of, pack
BUD = [1000, 2000, 4000, 8000, 16000]
Q = {q["question_id"]: q for q in json.load(open("env_h4.json"))}
E = {}
for r in map(json.loads, open("ctx_E0_h4.jsonl")):
    if r["gold_total"]: E[(r["qid"], r["budget"])] = r["gold_in"] / r["gold_total"]
def world(qid): return int(qid.split("_")[0][1:])
def grp(q): return any("group" in c.lower() for c in q.get("carriers", []))
rows = collections.defaultdict(list); miss = 0
for qid, q in Q.items():
    R = rounds_of(q); tot = sum(r[3] for r in R)
    if not tot: continue
    F = json.load(open(f"pool_h4/{qid}.json"))["fused"]
    o = [i for i, r in enumerate(R) if r[3]]; s = set(o); o += [i for i in F if i not in s]; s = set(o); o += [i for i in range(len(R)) if i not in s]
    orc = []; e0 = []
    for B in BUD:
        p = pack(q, R, o, B); orc.append(p["gold_in"] / p["gold_total"])
        if (qid, B) not in E: miss += 1; e0.append(float("nan"))
        else: e0.append(E[(qid, B)])
    half = "H4j" if world(qid) % 2 else "H4s"
    rows[(half, "group" if grp(q) else "other")].append((st.mean(orc), st.mean(e0)))
print("missing E0 cells", miss)
for k in sorted(rows):
    v = rows[k]; o = st.mean(x for x, _ in v); e = st.mean(y for _, y in v)
    print(f"{k[0]} {k[1]:6s} n={len(v):3d}  oracle {100*o:5.1f}  E0 {100*e:5.1f}  margin {100*(o-e):+5.1f}")
j = rows[("H4j", "group")]; m = 100 * (st.mean(x for x, _ in j) - st.mean(y for _, y in j))
print("VERDICT", "PASS" if m >= 10 and len(j) >= 80 else "FAIL (add worlds)", f"margin {m:.1f}, n_group {len(j)}")
