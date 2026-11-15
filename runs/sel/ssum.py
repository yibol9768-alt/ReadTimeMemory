"""S2 main criterion (11-01). usage: python3 ssum.py W20 W20C. Baseline per benchmark = the better of A0 and A0O. Intervals: two-sided 97.5%, paired bootstrap."""
import json, os, random, re, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of
hard = set(open("hard_ids.txt").read().split())
def gold_sessions(n):
    o = {}
    for q in json.load(open(f"q_{n}.json")):
        R = rounds_of(q); o[q["question_id"]] = ({r[0] for r in R if r[3]}, R)
    return o
GS = {}
def score(tag, n=None):
    """{(qid, budget): recall}; EverMem at session level."""
    o = {}
    if n == "ev150" and n not in GS: GS[n] = gold_sessions(n)
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if not r["gold_total"]: continue
        k = (r["qid"], r["budget"])
        if k in o: raise SystemExit(f"INVALID duplicate {k} in ctx_{tag}")
        if n == "ev150":
            g, R = GS[n][r["qid"]]; o[k] = len({R[i][0] for i in r["keep"]} & g) / len(g)
        else: o[k] = r["gold_in"] / r["gold_total"]
    return o
def conv(k):
    q = k[0] if isinstance(k, tuple) else k; return q.split("_")[0]
def diff(xs, ys, keys):
    """conversation-level bootstrap; benchmarks without shared conversations fall back to question level"""
    rng = random.Random(0); by = {}
    beam = all(re.match(r"h\d+_", (k[0] if isinstance(k, tuple) else k)) for k in keys)   # only BEAM questions share conversations here
    for k in keys: by.setdefault(conv(k) if beam else k, []).append(xs[k] - ys[k])
    cs = list(by); bs = sorted(st.mean(x for c in rng.choices(cs, k=len(cs)) for x in by[c]) for _ in range(8000))
    return st.mean(xs[k] - ys[k] for k in keys), bs[100], bs[7899], len(keys)
def need(lbl, keys, x, y, b, n):
    """Key sets must match the baseline at this budget and reach the frozen count, else the result is invalid."""
    a_ = {k for k in x if k[1] == b}; c = {k for k in y if k[1] == b}
    if len(keys) != n or not c <= a_: print(f"INVALID {lbl}: {len(keys)} of {n}, missing {len(c - a_)}"); return False
    return True
def line(lbl, x, y, keys):
    if not keys: print(f"{lbl:26s} no shared questions"); return float("nan")
    m, lo, hi, n = diff(x, y, keys); print(f"{lbl:26s} {100 * st.mean(x[k] for k in keys):.1f} vs {100 * st.mean(y[k] for k in keys):.1f} diff {100 * m:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}] n={n}"); return lo
def base(n, b, sel=lambda q: True):
    A = {t: score(f"{t}_{n}", "ev150" if n == "ev150" else None) for t in ("A0", "A0O")}
    return max(A.values(), key=lambda v: st.mean(x for (q, c), x in v.items() if c == b and sel(q)))
if __name__ == "__main__":
    B = json.load(open("budgets.json"))
    for tag in sys.argv[1:]:
        print("==", tag); res = {}
        x = score(f"{tag}_H"); h = base("H", 16000, lambda q: q in hard); o = base("H", 4000, lambda q: q not in hard)
        mix = lambda v, w: {q: (v[(q, 16000)] if q in hard else w[(q, 4000)]) for (q, c) in v if c == 16000 and (q, 4000) in w}
        xm, bm = mix(x, x), mix(h, o); ks = [q for q in xm if q in bm]
        res["beam"] = line("BEAM mixed", xm, bm, ks)
        kh = [k for k in x if k[1] == 16000 and k[0] in hard and k in h]; ko = [k for k in x if k[1] == 4000 and k[0] not in hard and k in o]
        valid = need("BEAM hard", kh, x, {k: 0 for k in h if k[1] == 16000 and k[0] in hard}, 16000, 140) and need("BEAM other", ko, x, {k: 0 for k in o if k[1] == 4000 and k[0] not in hard}, 4000, 140) and len(ks) == 280
        res["beam_h"] = line("  hard 16k", x, h, kh); res["beam_o"] = line("  other 4k", x, o, ko)
        x = score(f"{tag}_lmes150"); y = base("lmes150", 2000); kl = [k for k in x if k[1] == 2000 and k in y]; valid = need("LME", kl, x, y, 2000, 150) and valid; res["lme"] = line("LongMemEval 2k", x, y, kl)
        for n, lbl in (("ep150", "EpBench"), ("ev150", "EverMem")):
            b = B[n]
            if b is None: print(f"INVALID {lbl}: every budget above 80% for the baseline"); res[n] = float("nan"); continue
            x = score(f"{tag}_{n}", n); y = base(n, b); kk = [k for k in x if k[1] == b and k in y]; valid = need(lbl, kk, x, y, b, 150) and valid; res[n] = line(f"{lbl} {b // 1000}k", x, y, kk)
        x, y = score(f"{tag}_w14"), score("A0_w14"); line("held-out world 4k", x, y, [k for k in x if k[1] == 4000 and k in y])
        for sc in ("_date_", "_none_", "_single_"): line(f"  world scope {sc.strip('_')}", x, y, [k for k in x if k[1] == 4000 and k in y and sc in k[0]])
        x = score(f"{tag}_H"); t = score("T2X_H"); line("BEAM hard 16k vs T2X", x, t, [k for k in x if k[1] == 16000 and k[0] in hard and k in t]); line("BEAM other 4k vs T2X", x, t, [k for k in x if k[1] == 4000 and k[0] not in hard and k in t])
        x = score(f"{tag}_lmes150"); t = score("T2X_lmes150"); line("LME 2k vs T2X", x, t, [k for k in x if k[1] == 2000 and k in t])
        Q = {q["question_id"]: q["question_type"] for q in json.load(open("q_lmes150.json"))}; y = base("lmes150", 2000)
        for ty in sorted(set(Q.values())): line(f"  LME {ty[:18]}", x, y, [k for k in x if k[1] == 2000 and k in y and Q[k[0]] == ty])
        if os.path.exists(f"ctx_{tag}M_H.jsonl"):
            xm = score(f"{tag}M_H"); a0 = score("A0M_H"); line("BEAM dates hidden hard16k", xm, a0, [k for k in xm if k[1] == 16000 and k[0] in hard and k in a0])
            xm = score(f"{tag}M_lmes150"); y = score("A0M_lmes150"); line("LME dates hidden 2k", xm, y, [k for k in xm if k[1] == 2000 and k in y])
        bad = [k for k, v in res.items() if v != v]   # a missing or empty comparison can never pass
        if bad: print("MISSING RESULTS", bad)
        ok = valid and not bad and res["beam"] > 0 and res["lme"] > 0 and sum(res[k] > 0 for k in ("beam", "lme", "ep150", "ev150")) >= 3 and min(res.values()) >= -.02
        x = score(f"{tag}_H"); t = score("T2X_H"); xl = score(f"{tag}_lmes150"); tl = score("T2X_lmes150")
        lo = [diff(x, t, [k for k in x if k[1] == 16000 and k[0] in hard and k in t])[1], diff(xl, tl, [k for k in xl if k[1] == 2000 and k in tl])[1]]
        print("BEATS OLD SYNTHETIC T2X (BEAM hard 16k and LME 2k lower bounds > 0)", "YES" if min(lo) > 0 else "NO", [round(100 * v, 1) for v in lo])
        print("MAIN CRITERION", "PASS" if ok else "NOT PASSED", {k: round(100 * v, 1) for k, v in res.items()})
