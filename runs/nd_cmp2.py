# Note-part ablation vs full note3_deep~m0 for one reader arm, over scales and gens; per question, average over the gens
# where both the variant and the full notes are judged; bootstrap clustered by conversation.
# usage: nd_cmp2.py ARM SCALES(beam,beam1m) GENS(1,2,3)
import json, os, random, sys, collections
arm, scales, gens = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
C = ["nd_p1234~m0", "nd_p1345~m0", "nd_p1245~m0", "nd_p1235~m0", "nd_p1~m0"]
names = {"nd_p1234~m0": "no standing", "nd_p1345~m0": "no changes", "nd_p1245~m0": "no conflicts", "nd_p1235~m0": "no not-found", "nd_p1~m0": "evidence only"}
S = collections.defaultdict(dict)       # (gen, cond) -> {qid: score}
for s in scales:
    for g in gens:
        f = f"runs/{s}/{arm}/judged{'' if g == '1' else '_g' + g}.jsonl"
        if not os.path.exists(f): continue
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            S[(g, r["cond"])][s + ":" + r["qid"]] = r["score"]
for c in C:
    d = collections.defaultdict(list); full = collections.defaultdict(list)
    for g in gens:
        a, b = S.get((g, c), {}), S.get((g, "note3_deep~m0"), {})
        for q in a:
            if q in b: d[q].append(a[q] - b[q]); full[q].append(b[q])
    if not d: continue
    q = sorted(d); dd = {k: sum(v) / len(v) for k, v in d.items()}
    conv = collections.defaultdict(list)
    for k in q: conv[k.split("-")[0]].append(k)
    rng = random.Random(0); bs = []
    for _ in range(4000):
        Q = [k for cc in rng.choices(list(conv), k=len(conv)) for k in conv[cc]]; bs.append(100 * sum(dd[k] for k in Q) / len(Q))
    bs.sort(); ng = collections.Counter(len(v) for v in d.values())
    print(f"{names[c]:14s} n={len(q)} gens/q={dict(ng)} full {100*sum(sum(v)/len(v) for v in full.values())/len(q):.1f} "
          f"diff {100*sum(dd.values())/len(q):+.1f} [{bs[100]:+.1f},{bs[3900]:+.1f}]")
