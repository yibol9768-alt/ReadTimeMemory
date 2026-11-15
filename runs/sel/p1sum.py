"""Step 1 summary: per model and dev set, R@10 and MRR (ranking over the 600-round pool, gold rounds) and evidence
recall at 1k/2k/4k/16k (EverMem: gold sessions, as ssum.py). Paired bootstrap 95% CI of each model minus CruxRank
(E0), 4000 draws, seed 0; BEAM resampled by conversation, others by question.
usage: python3 p1sum.py [PREFIX=P1] [MODELS=A0,T1,MR4,RR8,E0,E1,E2,BGEL,BGEM3,GTE,JINA]"""
import json, os, random, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of
PFX = sys.argv[1] if len(sys.argv) > 1 else "P1"
MODELS = (sys.argv[2] if len(sys.argv) > 2 else "A0,T1,MR4,RR8,E0,E1,E2,BGEL,BGEM3,GTE,JINA").split(",")
BENCH = {"lmes150": "q_lmes150.json", "ep150": "q_ep150.json", "ev150": "q_ev150.json", "loc150": "q_loc150.json", "H": "beam_ex.json"}
BUD = [1000, 2000, 4000, 16000]
G = {}
for b, f in BENCH.items():
    G[b] = {}
    for q in json.load(open(f)):
        R = rounds_of(q); G[b][q["question_id"]] = ({i for i, r in enumerate(R) if r[3]}, {r[0] for r in R if r[3]}, R)


def load(m, b):
    p = f"ctx_{PFX}{m}_{b}.jsonl"
    if not os.path.exists(p): return None
    out = {}
    for r in map(json.loads, open(p)):
        gr, gs, R = G[b][r["qid"]]
        if not gr: continue
        d = out.setdefault(r["qid"], {})
        rk = r["rank"]
        d["r10"] = len(set(rk[:10]) & gr) / len(gr)
        d["mrr"] = next((1 / (k + 1) for k, i in enumerate(rk) if i in gr), 0.0)
        d[r["budget"]] = (len({R[i][0] for i in r["keep"]} & gs) / len(gs)) if b == "ev150" else r["gold_in"] / r["gold_total"]
    return out


def ci(x, y):
    K = sorted(set(x) & set(y)); beam = all(k.startswith("h") for k in K)
    by = {}
    for k in K: by.setdefault(k.split("_")[0] if beam else k, []).append(k)
    cl = list(by); rng = random.Random(0); bs = []
    for _ in range(4000):
        ks = [k for c in rng.choices(cl, k=len(cl)) for k in by[c]]
        bs.append(st.mean(x[k] - y[k] for k in ks))
    bs.sort(); return 100 * st.mean(x[k] - y[k] for k in K), 100 * bs[99], 100 * bs[3899]


cols = ["r10", "mrr"] + BUD
for b in BENCH:
    D = {m: load(m, b) for m in MODELS}; D = {m: v for m, v in D.items() if v}
    if not D: continue
    print(f"\n== {b}   (n = {len(next(iter(D.values())))})")
    print(f"{'model':8s}" + "".join(f"{str(c):>8s}" for c in cols) + "   minus E0 at 2k (95% CI)")
    for m, d in D.items():
        row = "".join(f"{100 * st.mean(v[c] for v in d.values() if c in v):8.1f}" for c in cols)
        extra = ""
        if m != "E0" and "E0" in D:
            k = [q for q in d if q in D["E0"]]
            dx, lo, hi = ci({q: d[q][2000] for q in k}, {q: D["E0"][q][2000] for q in k}); extra = f"   {dx:+.1f} [{lo:+.1f},{hi:+.1f}]"
        print(f"{m:8s}" + row + extra)
