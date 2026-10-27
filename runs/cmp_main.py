# Main comparison for one reader arm on BEAM: note3_deep~m0 vs Hindsight (+ conflict check on point questions), mean over gens,
# bootstrap clustered by conversation. usage: cmp_main.py ARM SCALE(beam|beam1m) GENS(1,2,3)
import json, collections, random, sys, os
arm, S, gens = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
P = f"runs/{S}/{arm}/"; route = json.load(open(P + "route.json")); acc = collections.defaultdict(lambda: collections.defaultdict(list))
for g in gens:
    f = P + f"judged{'' if g == '1' else '_g' + g}.jsonl"; seen = set()
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if (r["cond"], r["qid"]) in seen: continue
        seen.add((r["cond"], r["qid"])); acc[r["cond"]][r["qid"]].append(r["score"])
d = {c: {q: sum(v) / len(v) for q, v in x.items()} for c, x in acc.items()}
qs = sorted(q for q in route if q in d["note3_deep~m0"] and q in d["hindsight~m0"])
D, H = d["note3_deep~m0"], d["hindsight~m0"]; HC = {q: (d["cf_hindsight~m0"][q] if route[q] == "POINT" else H[q]) for q in qs}
m = lambda x, Q: round(100 * sum(x[q] for q in Q) / len(Q), 1)
pts = [q for q in qs if route[q] == "POINT"]; ar = [q for q in qs if route[q] == "AREA"]
print(f"{arm} {S} gens={min(len(acc['note3_deep~m0'][q]) for q in qs)} n={len(qs)} ours {m(D,qs)} hs_cf {m(HC,qs)} hs {m(H,qs)} | point {m(D,pts)}/{m(HC,pts)}/{m(H,pts)} area {m(D,ar)}/{m(H,ar)}")
conv = collections.defaultdict(list)
for q in qs: conv[q.split("-")[0]].append(q)
random.seed(0)
for nm, X in (("hs_cf", HC), ("hs", H)):
    dd = {q: D[q] - X[q] for q in qs}; bs = []
    for _ in range(5000):
        Q = [q for c in random.choices(list(conv), k=len(conv)) for q in conv[c]]; bs.append(100 * sum(dd[q] for q in Q) / len(Q))
    bs.sort(); print(f"   ours-{nm} {100*sum(dd.values())/len(qs):+.1f} [{bs[125]:+.1f},{bs[4874]:+.1f}]")
