"""D3 v9 primary table on real dev sets (spec v9). usage: python3 v9sum.py
Four equal-weight benchmarks: BEAM (hard 16k, others 4k), LME 2k, EpBench 1k, EverMem session recall 4k. Tags are <TAG>_<set>; a missing file shows as absent, and a seed-averaged arm is never reported unless all seeds are present."""
import json, os, random, statistics as st, sys
sys.path.insert(0, "."); from k1 import rounds_of
hard = set(open("hard_ids.txt").read().split()); GS = {}
def gold(n):
    if n not in GS:
        GS[n] = {}
        for q in json.load(open(f"q_{n}.json")):
            R = rounds_of(q); GS[n][q["question_id"]] = ({r[0] for r in R if r[3]}, R)
    return GS[n]
def rows(t):
    p = f"ctx_{t}.jsonl"
    if not os.path.exists(p): return None
    o = {}
    for r in map(json.loads, open(p)):
        if not r["gold_total"]: continue
        k = (r["qid"], r["budget"])
        if k in o: raise SystemExit(f"INVALID duplicate {k} in {p}")
        o[k] = r
    return o
def sc(tag, b):
    if b == "beam":
        o = rows(f"{tag}_H")
        if not o: return None
        ids = {k[0] for k in o}; return {q: o[(q, 16000 if q in hard else 4000)]["gold_in"] / o[(q, 16000 if q in hard else 4000)]["gold_total"] for q in ids if (q, 16000 if q in hard else 4000) in o}
    if b == "lme":
        o = rows(f"{tag}_lmes150"); return {k[0]: r["gold_in"] / r["gold_total"] for k, r in o.items() if k[1] == 2000} if o else None
    if b == "ep":
        o = rows(f"{tag}_ep150"); return {k[0]: r["gold_in"] / r["gold_total"] for k, r in o.items() if k[1] == 1000} if o else None
    o = rows(f"{tag}_ev150")
    if not o: return None
    G = gold("ev150"); return {k[0]: len({G[k[0]][1][i][0] for i in r["keep"]} & G[k[0]][0]) / len(G[k[0]][0]) for k, r in o.items() if k[1] == 4000}
SETS = ["beam", "lme", "ep", "ev"]
NEED = {"beam": 280, "lme": 150, "ep": 150, "ev": 150}
def arm(tags):
    out = {}
    for b in SETS:
        v = [sc(t, b) for t in tags]
        if any(x is None for x in v): out[b] = None; continue
        if any(set(x) != set(v[0]) or len(x) != NEED[b] for x in v): print(f"INVALID {tags} {b}: incomplete or unequal sets {[len(x) for x in v]}"); out[b] = None; continue
        out[b] = {q: st.mean(x[q] for x in v) for q in v[0]}
    return out
def show(name, A):
    m = {b: (100 * st.mean(A[b].values()) if A[b] else None) for b in SETS}; tot = st.mean(m.values()) if all(v is not None for v in m.values()) else None
    print(f"{name:14s}", " ".join(f"{b} {m[b]:5.1f}" if m[b] is not None else f"{b}   -  " for b in SETS), "| mean", f"{tot:.1f}" if tot is not None else "-"); return m, tot
def cmp(a, b, name):
    for s in SETS:
        if a[s] and b[s]:
            K = sorted(set(a[s]) & set(b[s])); d = [a[s][k] - b[s][k] for k in K]; r = random.Random(0); bs = sorted(st.mean(r.choices(d, k=len(d))) for _ in range(4000))
            print(f"  {name} {s:5s} {100*st.mean(d):+.1f} [{100*bs[100]:+.1f},{100*bs[3899]:+.1f}] n={len(K)}")
T = {"A0O": ["A0O"], "A0": ["A0X"], "W20": ["W20X"], "F-am": ["D3RFX"], "E-am": ["D3REX"], "H-am": ["D3RHX"], "FIX": ["V9F0", "V9F1", "V9F2"], "E": ["V9E0", "V9E1", "V9E2"], "HS0": ["V9H0"], "SHORT": ["V9S0"], "F3": ["D3RFX", "D3F1", "D3F2"], "E3": ["D3REX", "D3E1", "D3E2"]}
R = {k: arm(v) for k, v in T.items()}
for k, v in R.items(): show(k, v)
for a, b in (("E", "FIX"), ("E", "W20"), ("FIX", "W20"), ("F3", "W20"), ("E3", "W20"), ("E3", "F3"), ("F3", "A0O"), ("E3", "A0O"), ("HS0", "F-am"), ("SHORT", "F-am"), ("F-am", "W20"), ("E-am", "W20"), ("H-am", "W20"), ("W20", "A0O"), ("F-am", "A0O"), ("E-am", "A0O")): cmp(R[a], R[b], f"{a}-{b}")
mE = {b: 100 * st.mean(R["E"][b].values()) for b in SETS if R["E"][b]}; mF = {b: 100 * st.mean(R["FIX"][b].values()) for b in SETS if R["FIX"][b]}; mW = {b: 100 * st.mean(R["W20"][b].values()) for b in SETS if R["W20"][b]}
if len(mE) == len(mF) == len(mW) == 4:
    e, f, w = st.mean(mE.values()), st.mean(mF.values()), st.mean(mW.values()); worst = min(mE[b] - mF[b] for b in SETS); ok = e - f >= 2 and e > w and worst >= -2
    print(f"SCREEN {'PASS' if ok else 'FAIL'} | E {e:.1f} fixed {f:.1f} W20 {w:.1f} | E-fixed {e-f:+.1f} need >=+2 | worst benchmark E-fixed {worst:+.1f} need >=-2")
else: print("SCREEN INCOMPLETE")
for nm, tags in (("F", ["D3RFX", "D3F1", "D3F2"]), ("E", ["D3REX", "D3E1", "D3E2"])):
    for t in tags: show(f"{nm} seed {t}", arm([t]))
