"""D3 evaluation (spec v2, frozen). usage: python3 d3eval.py QFILE BUDGETS TARGET_TYPE,TARGET_CARRIER W20TAG E=TAG FIX=TAG HARD=TAG
Cells with >= 30 questions on the eval worlds; macro = equal weight per cell; each budget separately, then the mean. Intervals: 95%, bootstrap over questions."""
import collections, json, random, statistics as st, sys
qf, Bs, tgt, w20 = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], tuple(sys.argv[3].split(",")), sys.argv[4]
arms = dict(x.split("=") for x in sys.argv[5:])
Q = {q["question_id"]: q for q in json.load(open(qf))}
def cell(q): c = q.get("carriers", ["?"]); return (q["question_type"], c[0] if len(c) == 1 else "mixed")
def load(tag):
    o = {}
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if r["budget"] in Bs and r["gold_total"]: o[(r["qid"], r["budget"])] = r["gold_in"] / r["gold_total"]
    miss = [(q, b) for q in Q for b in Bs if (q, b) not in o and any(t.get("has_answer") for s in Q[q]["haystack_sessions"] for t in s)]
    if miss: raise SystemExit(f"INVALID {tag}: {len(miss)} missing, e.g. {miss[:2]}")
    return o
W = load(w20); A = {k: load(v) for k, v in arms.items()}
qs = sorted({q for q, _ in W}); cells = collections.Counter(cell(Q[q]) for q in qs); E = [c for c, n in cells.items() if n >= 30]
print("eligible cells", {str(c): cells[c] for c in E}, "| target", tgt, "eligible" if tgt in E else "NOT ELIGIBLE")
if tgt not in E: print("VERDICT INVALID: target cell has fewer than 30 questions on the eval worlds"); sys.exit(2)
def per_q(o): return {q: st.mean(o[(q, b)] for b in Bs) for q in qs}
Wq = per_q(W); Aq = {k: per_q(v) for k, v in A.items()}
def macro(d, sel=None):
    cs = [c for c in E if sel is None or sel(c)]; return st.mean(st.mean(d[q] for q in qs if cell(Q[q]) == c) for c in cs) if cs else float("nan")
def boot(f, n=2000):
    r = random.Random(0); by = collections.defaultdict(list)
    for q in qs: by[cell(Q[q])].append(q)
    v = []
    for _ in range(n):
        s = [q for c in E for q in (r.choices(by[c], k=len(by[c])))]   # resample questions within each cell
        v.append(f(s))
    v.sort(); return v[int(.025 * n)], v[int(.975 * n)]
for k in A:
    print(f"{k:5s} macro {100*macro(Aq[k]):.1f} (W20 {100*macro(Wq):.1f})")
def ci(xs):
    r = random.Random(2); v = sorted(st.mean(r.choices(xs, k=len(xs))) for _ in range(2000)); return v[50], v[1949]
for c in sorted(E, key=str):
    cq = [q for q in qs if cell(Q[q]) == c]
    print(f"  {str(c):30s} n={cells[c]:3d} W20 {100*st.mean(Wq[q] for q in cq):5.1f} " + " ".join(f"{k} {100*st.mean(Aq[k][q] for q in cq):5.1f} (vs W20 {100*st.mean(Aq[k][q]-Wq[q] for q in cq):+.1f} [{100*ci([Aq[k][q]-Wq[q] for q in cq])[0]:+.1f},{100*ci([Aq[k][q]-Wq[q] for q in cq])[1]:+.1f}])" for k in A))
if "E" in A and tgt in E:
    tq = [q for q in qs if cell(Q[q]) == tgt]; d_t = st.mean(Aq["E"][q] - Wq[q] for q in tq)
    r = random.Random(1); bs = sorted(st.mean(Aq["E"][q] - Wq[q] for q in r.choices(tq, k=len(tq))) for _ in range(2000)); lo = bs[50]
    best = max((k for k in A if k != "E"), key=lambda k: macro(Aq[k])); d_m = macro(Aq["E"]) - macro(Aq[best])
    nt = [c for c in E if c != tgt]; drop = macro(Wq, lambda c: c != tgt) - macro(Aq["E"], lambda c: c != tgt)
    print(f"GATE target E-W20 {100*d_t:+.1f} (lower {100*lo:+.1f}) need >=+5 and lower>0 | macro E-{best} {100*d_m:+.1f} need >=+2 | non-target drop {100*drop:+.1f} need <=2")
    pts = d_t >= .05 and d_m >= .02 and drop <= .02
    print("VERDICT", "PASS" if pts and lo > 0 else ("INSUFFICIENT" if pts else "FAIL"))
for k in A:   # per-world paired difference vs W20
    byw = collections.defaultdict(list)
    for q in qs: byw[q.split("_")[0]].append(Aq[k][q] - Wq[q])
    print(f"{k} per world vs W20:", {w: round(100 * st.mean(v), 1) for w, v in sorted(byw.items())})
