"""Frozen rank fusion (11-01 15:00): equal RRF k=60 of A0 and morning fixed, comparator A0 + 8B, and the single models, EpBench 1k and EverMem 4k, same packing. Ties by round index. No weights searched.
Keep only if the mean over both benchmarks beats the stronger single model with paired 95% lower bound > 0 and neither benchmark loses more than 2. usage: python3 fus.py"""
import json, random, statistics as st, sys
sys.path.insert(0, "."); from k1 import rounds_of, pack
RK = 60
def rrf(*L):
    s = {}
    for l in L:
        for k, i in enumerate(l): s[i] = s.get(i, 0) + 1 / (RK + k + 1)
    return sorted(s, key=lambda i: (-s[i], i))
def ctx(t, b): return {r["qid"]: r for r in map(json.loads, open(f"ctx_{t}.jsonl")) if r["budget"] == b}
SET = {"ep": ("q_ep150.json", "pool_ep150", 1000, "A0X_ep150", "D3RFR_ep150", False), "ev": ("q_ev150.json", "pool_ev150", 4000, "A0X_ev150", "D3RFR_ev150", True)}
T = {}
for n, (qf, pd, b, a0, fx, sess) in SET.items():
    A, F = ctx(a0, b), ctx(fx, b); tab = {k: {} for k in ("A0", "FIX", "A0+FIX", "A0+8B", "8B")}
    for q in json.load(open(qf)):
        i = q["question_id"]; R = rounds_of(q); g = {r[0] for r in R if r[3]}
        if not any(r[3] for r in R): continue
        P = json.load(open(f"{pd}/{i}.json")); Fu = P["fused"]; rr = {int(k): v for k, v in P["rr8"].items()}; o8 = sorted(rr, key=lambda x: (-rr[x], x)); a, f = A[i]["rank"], F[i]["rank"]
        for nm, o in (("A0", a), ("FIX", f), ("A0+FIX", rrf(a, f)), ("A0+8B", rrf(a, o8)), ("8B", o8)):
            seen = set(o); p = pack(q, R, list(o) + [x for x in Fu if x not in seen], b)
            tab[nm][i] = len({R[k][0] for k in p["keep"]} & g) / len(g) if sess else p["gold_in"] / p["gold_total"]
    T[n] = tab; print(n, {k: round(100 * st.mean(v.values()), 1) for k, v in tab.items()})
def score(nm): return {n: T[n][nm] for n in T}
m = lambda nm: st.mean(st.mean(T[n][nm].values()) for n in T)
best = max(("A0", "FIX"), key=m); print("stronger single", best)
for nm in ("A0+FIX", "A0+8B"):
    r = random.Random(0); bs = []
    for _ in range(4000):
        v = []
        for n in T:
            K = list(T[n][nm]); s = r.choices(K, k=len(K)); v.append(st.mean(T[n][nm][q] - T[n][best][q] for q in s))
        bs.append(st.mean(v))
    bs.sort(); per = {n: round(100 * st.mean(T[n][nm][q] - T[n][best][q] for q in T[n][nm]), 1) for n in T}; lo, hi = bs[100], bs[3899]; d = m(nm) - m(best)
    print(nm, f"mean diff {100*d:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] per benchmark {per}", "KEEP" if lo > 0 and min(per.values()) >= -2 else "DROP")
