# Note-part ablation vs full note3_deep~m0 (gen1, luna), on qids judged for both; paired bootstrap over questions.
import json, random, sys
s = sys.argv[1]; S = {}
for l in open(f"runs/{s}/pro6/judged.jsonl"):
    r = json.loads(l); S.setdefault(r["cond"], {})[r["qid"]] = (r["score"], r["ability"])
base = S["note3_deep~m0"]
for c in ["nd_p1234~m0", "nd_p1345~m0", "nd_p1245~m0", "nd_p1235~m0", "nd_p1~m0"]:
    if c not in S: continue
    q = [k for k in S[c] if k in base]; d = [S[c][k][0] - base[k][0] for k in q]
    rng = random.Random(0); bs = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
    ab = {}
    for k, x in zip(q, d): ab.setdefault(base[k][1], []).append(x)
    worst = sorted(((100 * sum(v) / len(v), a) for a, v in ab.items()))[:2]
    print(f"{c:12s} n={len(q)} {100*sum(S[c][k][0] for k in q)/len(q):.1f} vs full {100*sum(base[k][0] for k in q)/len(q):.1f} "
          f"diff {100*sum(d)/len(d):+.1f} [{bs[100]:+.1f},{bs[3900]:+.1f}] worst: " + ", ".join(f"{a[:12]} {v:+.0f}" for v, a in worst))
