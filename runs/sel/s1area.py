"""Stage-1 summary metric (CRITERIA_FROZEN §4–5 on DEV data, descriptive only): recall area = mean recall over budgets
1k,2k,4k,8k,16k (log-equal). 8k is re-packed from each model's stored full rank with k1.pack (same packing as infer.py);
the stored 1k/2k/4k/16k keep sets are re-packed too and must match exactly, otherwise INVALID.
EverMem: session recall. BEAM: also "mixed" = hard@16k and other@4k, equal weight per question.
CI: paired bootstrap 8000 draws, seed 0, 95%; BEAM and LoCoMo resampled by conversation, others by question.
usage: python3 s1area.py [MODELS]   (baseline per bench = better of A0/A0O on area, as CRITERIA §5)"""
import json, random, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of, pack
MODELS = (sys.argv[1] if len(sys.argv) > 1 else "E0,E1,E2,A0,A0O,RR8,RR8O,MR4,MR4O,T1,BGEL,BGEM3,GTE").split(",")
BENCH = {"lmes150": "q_lmes150.json", "ep150": "q_ep150.json", "ev150": "q_ev150.json", "loc150": "q_loc150.json", "H": "beam_ex.json"}
BUD = [1000, 2000, 4000, 8000, 16000]
PD = {"lmes150": "pool_lmes150", "ep150": "pool_ep150", "ev150": "pool_ev150", "loc150": "pool_loc150", "H": "pool"}
hard = set(open("hard_ids.txt").read().split())
Q = {b: {q["question_id"]: q for q in json.load(open(f))} for b, f in BENCH.items()}
RR = {b: {i: rounds_of(q) for i, q in Q[b].items()} for b in BENCH}


def rec(b, R, keep):
    if b == "ev150":
        gs = {r[0] for r in R if r[3]}; return len({R[i][0] for i in keep} & gs) / len(gs)
    return sum(R[i][3] for i in keep) / sum(r[3] for r in R)


def order(b, qid, rk):
    """infer.py packing order: ranked pool, then the rest of the fused pool, then every other round (backfill)"""
    F = json.load(open(f"{PD[b]}/{qid}.json"))["fused"]; o = list(rk); s = set(o)
    o += [i for i in F if i not in s]; s = set(o); o += [i for i in range(len(RR[b][qid])) if i not in s]; return o


def load(m, b):
    try: rows = [json.loads(l) for l in open(f"ctx_P1{m}_{b}.jsonl")]
    except FileNotFoundError: return None
    out = {}; rank = {}
    for r in rows:
        R = RR[b][r["qid"]]
        if not any(x[3] for x in R): continue
        if r["budget"] == 2000 or r["qid"] not in rank: rank[r["qid"]] = r["rank"]
        k = pack(Q[b][r["qid"]], R, order(b, r["qid"], r["rank"]), r["budget"])["keep"]
        if sorted(k) != sorted(r["keep"]): sys.exit(f"INVALID repack {m} {b} {r['qid']} {r['budget']}")
        out.setdefault(r["qid"], {})[r["budget"]] = rec(b, R, r["keep"])
    for qid, rk in rank.items():
        out[qid][8000] = rec(b, RR[b][qid], pack(Q[b][qid], RR[b][qid], order(b, qid, rk), 8000)["keep"])
    for qid, d in out.items():
        if set(d) != set(BUD): sys.exit(f"INVALID budgets {m} {b} {qid} {sorted(d)}")
        d["area"] = st.mean(d[x] for x in BUD)
        if b == "H": d["mixed"] = d[16000] if qid in hard else d[4000]
    return out


def clus(b, k): return k.split("_")[0] if b == "H" else (Q[b][k].get("conv") or k.rsplit("_", 1)[0]) if b == "loc150" else k


def ci(b, x, y):
    K = sorted(set(x) & set(y)); by = {}
    for k in K: by.setdefault(clus(b, k), []).append(k)
    cl = list(by); rng = random.Random(0); bs = []
    for _ in range(8000):
        ks = [k for c in rng.choices(cl, k=len(cl)) for k in by[c]]; bs.append(st.mean(x[k] - y[k] for k in ks))
    bs.sort(); return 100 * st.mean(x[k] - y[k] for k in K), 100 * bs[199], 100 * bs[7799], len(cl)



if "--crit" in sys.argv:
    # CRITERIA_FROZEN §5 applied to DEV data (descriptive): two-sided 97.5% (Bonferroni over two recipes), paired bootstrap 8000.
    # Baseline = better of A0/A0O on the metric used. BEAM: hard@16k, other@4k, mixed; LME, LoCoMo: area (and LME 2k, key budget);
    # EpBench, EverMem: largest budget in {1k,2k,4k,16k} where the baseline is <= 80% (decided from baseline only).
    def ci2(b, x, y):
        K = sorted(set(x) & set(y)); by = {}
        for k in K: by.setdefault(clus(b, k), []).append(k)
        cl = list(by); rng = random.Random(0); bs = []
        for _ in range(8000):
            ks = [k for c in rng.choices(cl, k=len(cl)) for k in by[c]]; bs.append(st.mean(x[k] - y[k] for k in ks))
        bs.sort(); return 100 * st.mean(x[k] - y[k] for k in K), 100 * bs[99], 100 * bs[7899]
    for b in BENCH:
        D = {m: load(m, b) for m in ("E0", "A0", "A0O")}
        mets = {"H": ["mixed", 16000, 4000], "lmes150": ["area", 2000], "loc150": ["area"]}.get(b)
        if mets is None:
            for B in (16000, 4000, 2000, 1000):
                if max(100 * st.mean(v[B] for v in D[m].values()) for m in ("A0", "A0O")) <= 80: mets = [B]; break
            mets = mets or [1000]
        for met in mets:
            sub = (lambda q: q in hard) if (b == "H" and met == 16000) else (lambda q: q not in hard) if (b == "H" and met == 4000) else (lambda q: True)
            mean = {m: st.mean(v[met] for q, v in D[m].items() if sub(q)) for m in ("A0", "A0O")}; base = max(mean, key=mean.get)
            k = [q for q in D["E0"] if sub(q)]
            dx, lo, hi = ci2(b, {q: D["E0"][q][met] for q in k}, {q: D[base][q][met] for q in k})
            print(f"CRIT {b:8s} {str(met):6s} n={len(k):3d} base={base:4s} E0 {100*st.mean(D['E0'][q][met] for q in k):5.1f} vs {100*mean[base]:5.1f}  {dx:+.1f} [{lo:+.1f},{hi:+.1f}] (97.5%)")
    sys.exit(0)

for b in BENCH:
    D = {m: load(m, b) for m in MODELS}; D = {m: v for m, v in D.items() if v}
    cols = ["area"] + BUD + (["mixed"] if b == "H" else [])
    print(f"\n== {b} (n={len(D['E0'])})"); print("model   " + "".join(f"{str(c):>8s}" for c in cols) + "   E0 minus model, area (95% CI) [clusters]" + ("   mixed" if b == "H" else ""))
    for m, d in D.items():
        line = f"{m:8s}" + "".join(f"{100 * st.mean(v[c] for v in d.values()):8.1f}" for c in cols)
        if m != "E0":
            k = set(d) & set(D["E0"]); dx, lo, hi, nc = ci(b, {q: D["E0"][q]["area"] for q in k}, {q: d[q]["area"] for q in k})
            line += f"   {dx:+.1f} [{lo:+.1f},{hi:+.1f}] [{nc}]"
            if b == "H":
                dx, lo, hi, _ = ci(b, {q: D["E0"][q]["mixed"] for q in k}, {q: d[q]["mixed"] for q in k}); line += f"   {dx:+.1f} [{lo:+.1f},{hi:+.1f}]"
        print(line)
