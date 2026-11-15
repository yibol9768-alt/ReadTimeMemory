"""3-generation means and conversation-clustered paired bootstrap on the BEAM-100K dev set (PROTOCOL 10-16)."""
import json, math, os, random, statistics as st, sys
DEV = set(os.environ.get("DEV", "b0,b2,b3,b9,b10,b11,b14,b16,b17,b18").split(","))
ROOT = os.environ.get("ROOT", "runs/beam/pro6/")
G = {}
for g, f in [(1, "judged.jsonl"), (2, "judged_g2.jsonl"), (3, "judged_g3.jsonl")]:
    G[g] = {(r["qid"], r["cond"]): r["score"] for r in map(json.loads, open(ROOT + f))}
conds = sys.argv[1].split(",")
qs = sorted({q for q, c in G[1] if c == conds[0] and q.split("-")[0] in DEV})
qs = [q for q in qs if all(any((q, c) in G[g] for g in G) for c in conds)]
def m(c):
    out = {}
    for q in qs:
        v = [G[g][(q, c)] for g in G if (q, c) in G[g]]
        out[q] = sum(v) / len(v)
    return out
M = {c: m(c) for c in conds}
ngen = {c: min(sum((q, c) in G[g] for g in G) for q in qs) for c in conds}
print(f"{len(qs)} questions, {len({q.split('-')[0] for q in qs})} conversations")
for c in conds:
    print(f"  {c:12s} {100 * st.mean(M[c].values()):5.1f}  (gens >= {ngen[c]})")
cv = sorted({q.split("-")[0] for q in qs})
for i, a in enumerate(conds):
    for b in conds[i + 1:]:
        d = {q: M[a][q] - M[b][q] for q in qs}
        w = sum(x > 0 for x in d.values()); l = sum(x < 0 for x in d.values()); n = w + l; k = min(w, l)
        p = min(1, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1
        byc = {c: [d[q] for q in qs if q.split("-")[0] == c] for c in cv}
        rng = random.Random(0); bs = []
        for _ in range(4000):
            s = [x for c in (rng.choice(cv) for _ in cv) for x in byc[c]]; bs.append(100 * sum(s) / len(s))
        bs.sort()
        print(f"  {a} - {b}: {100 * st.mean(d.values()):+.1f}  {w}:{l} p={p:.3f}  95% [{bs[100]:+.1f}, {bs[3899]:+.1f}]")
