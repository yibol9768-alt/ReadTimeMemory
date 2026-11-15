import json, math, random, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of
B = [1000, 2000, 4000, 16000]
SETS = {"lmes150": "q_lmes150.json", "ep150": "q_ep150.json", "ev150": "q_ev150.json", "loc150": "q_loc150.json", "H": "beam_ex.json"}
def area(tag, n):
    gs = {}
    if n == "ev150":
        for q in json.load(open(SETS[n])):
            R = rounds_of(q); gs[q["question_id"]] = ({r[0] for r in R if r[3]}, R)
    o = {}
    for r in map(json.loads, open(f"ctx_{tag}_{n}.jsonl")):
        if not r["gold_total"]: continue
        v = (len({gs[r["qid"]][1][i][0] for i in r["keep"]} & gs[r["qid"]][0]) / len(gs[r["qid"]][0])) if n == "ev150" else r["gold_in"] / r["gold_total"]
        o.setdefault(r["qid"], {})[r["budget"]] = v
    return {q: st.mean(d[b] for b in B) for q, d in o.items() if all(b in d for b in B)}
allc = []
for n in SETS:
    x = {q: st.mean(v) for q, v in zip(area("P1E0", n), zip(*(area(f"P1E{s}", n).values() for s in (0, 1, 2))))} if False else None
    e = [area(f"P1E{s}", n) for s in (0, 1, 2)]; a = area("P1A0", n)
    K = sorted(set(a) & set(e[0]) & set(e[1]) & set(e[2]))
    d = {q: st.mean(ei[q] for ei in e) - a[q] for q in K}
    clus = {}
    for q in K: clus.setdefault(q.split("_")[0] if n in ("H", "loc150") else q, []).append(d[q])
    cl = list(clus); rng = random.Random(0)
    bs = sorted(st.mean([v for c in rng.choices(cl, k=len(cl)) for v in clus[c]]) for _ in range(4000))
    hw = 100 * (bs[3899] - bs[100]) / 2
    print(f"{n:8s} n={len(K)} clusters={len(cl)} mean diff {100*st.mean(d.values()):+.1f}  95% half-width {hw:.1f}")
    allc.append(([v for c in cl for v in clus[c]], clus))
# 5-fold pooled (fold-stratified, equal weight per fold)
rng = random.Random(0); bs = []
for _ in range(4000):
    m = []
    for vals, clus in allc:
        cl = list(clus); m.append(st.mean([v for c in rng.choices(cl, k=len(cl)) for v in clus[c]]))
    bs.append(st.mean(m))
bs.sort(); print(f"pooled 5 folds: mean {100*st.mean(st.mean(v) for v, _ in allc):+.1f}, 95% half-width {100*(bs[3899]-bs[100])/2:.1f}")
