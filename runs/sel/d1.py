"""D1 (11-01, PLAN §11): retro test of signal R = selection-behaviour shift on unlabelled real histories. CPU only, S2 dev results.
Features are frozen here before any result is looked at. Reference = 8B teacher order (A0O ctx). Student = W20; A0 reported too.
usage: python3 d1.py > d1_report.txt"""
import collections, hashlib, json, random, re, statistics as st, sys
sys.path.insert(0, "."); from k1 import rounds_of, pack
MN = "January|February|March|April|May|June|July|August|September|October|November|December"
FULLD = re.compile(r"\b(?:" + MN + r") \d{1,2}, \d{4}\b|\b\d{4}-\d{2}-\d{2}\b")
def user(t): m = re.search(r"User: (.*?)(?:\nAssistant: |$)", t, re.S); return m.group(1) if m else t
def asst(t): m = re.search(r"\nAssistant: (.*)$", t, re.S); return m.group(1) if m else ""
def asker(q):
    m = re.search(r"\bI \(([A-Z][a-z]+ [A-Z][a-z]+)\)|\bI'?m ([A-Z][a-z]+ [A-Z][a-z]+)\b", q["question"]); return (m.group(1) or m.group(2)) if m else None
FEATS = {   # frozen 11-01 08:25 ET
    "date_lead":   lambda q, R, i, a: bool(FULLD.search(user(R[i][2])[:120])),
    "any_date":    lambda q, R, i, a: bool(FULLD.search(R[i][2])),
    "long":        lambda q, R, i, a: len(R[i][2]) > 3000,
    "short":       lambda q, R, i, a: len(R[i][2]) < 400,
    "asker_self":  lambda q, R, i, a: bool(a) and bool(re.match(r"\[Group \d+\] " + re.escape(a) + ":", user(R[i][2]))),
    "multi_mention": lambda q, R, i, a: user(R[i][2]).count("@") >= 2,
    "user_question": lambda q, R, i, a: "?" in user(R[i][2]),
    "first_person": lambda q, R, i, a: bool(re.search(r"\b(I|my|I'm|I've)\b", user(R[i][2]))),
    "late":        lambda q, R, i, a: i >= 0.8 * len(R),
    "early":       lambda q, R, i, a: i < 0.2 * len(R),
    "numbers":     lambda q, R, i, a: len(re.findall(r"\d+", R[i][2])) >= 3,
    "no_assistant": lambda q, R, i, a: not asst(R[i][2]).strip(),
}
SETS = {"BEAM": ("beam_ex.json", "H", "pool"), "LME": ("q_lmes150.json", "lmes150", "pool_lmes150"), "EverMem": ("q_ev150.json", "ev150", "pool_ev150"), "EpBench": ("q_ep150.json", "ep150", "pool_ep150")}
B = {"LME": 2000, "EverMem": 4000, "EpBench": 1000}; hard = set(open("hard_ids.txt").read().split())
budget = lambda n, qid: B.get(n) or (16000 if qid in hard else 4000)
def ctx(tag): return {(r["qid"], r["budget"]): r for r in map(json.loads, open(f"ctx_{tag}.jsonl"))}
def half(n, qid):
    k = qid.split("_")[0] if n == "BEAM" else qid
    return int(hashlib.md5(k.encode()).hexdigest(), 16) % 2   # 0 discovery, 1 test
def share(F, keep): return sum(F[i] for i in keep) / len(keep) if keep else 0.0
def boot(xs, f, n=2000):
    r = random.Random(0); v = sorted(f([xs[r.randrange(len(xs))] for _ in xs]) for _ in range(n)); return v[int(.025 * n)], v[int(.975 * n)]
def spear(p):
    if len(p) < 3: return float("nan")
    def rk(v):
        o = sorted(range(len(v)), key=v.__getitem__); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    x, y = rk([a for a, _ in p]), rk([b for _, b in p]); mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5; sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")
out = {}
for n, (qf, tag, pd) in SETS.items():
    Q = {q["question_id"]: q for q in json.load(open(qf))}; W, A, O = ctx(f"W20_{tag}"), ctx(f"A0_{tag}"), ctx(f"A0O_{tag}")
    rows = []
    for qid, q in Q.items():
        b = budget(n, qid); k = (qid, b)
        if k not in W or k not in O or k not in A or not W[k]["gold_total"]: continue
        R = rounds_of(q); a = asker(q); F = {f: [fn(q, R, i, a) for i in range(len(R))] for f, fn in FEATS.items()}
        g = [i for i, r in enumerate(R) if r[3]]
        rows.append(dict(qid=qid, h=half(n, qid), F=F, R=R, q=q, b=b, w=W[k]["keep"], a0=A[k]["keep"], o=O[k]["keep"], g=g,
                         rw=W[k]["gold_in"] / W[k]["gold_total"], ro=O[k]["gold_in"] / O[k]["gold_total"], ra=A[k]["gold_in"] / A[k]["gold_total"]))
    out[n] = rows
    print(f"\n===== {n}: {len(rows)} questions (discovery {sum(r['h']==0 for r in rows)}, test {sum(r['h']==1 for r in rows)}); recall W20 {100*st.mean(r['rw'] for r in rows):.1f} 8B {100*st.mean(r['ro'] for r in rows):.1f} A0 {100*st.mean(r['ra'] for r in rows):.1f}")
    D = {}
    for h in (0, 1):
        rr = [r for r in rows if r["h"] == h]
        D[h] = {f: st.mean(share(r["F"][f], r["w"]) - share(r["F"][f], r["o"]) for r in rr) for f in FEATS}
    print(" feature        shift W20-8B (disc / test)  A0-8B   gold-8B   base rate")
    for f in sorted(FEATS, key=lambda f: -abs(D[0][f])):
        da = st.mean(share(r["F"][f], r["a0"]) - share(r["F"][f], r["o"]) for r in rows)
        dg = st.mean(share(r["F"][f], r["g"]) - share(r["F"][f], r["o"]) for r in rows)
        base = st.mean(sum(r["F"][f]) / len(r["R"]) for r in rows)
        print(f" {f:14s} {100*D[0][f]:+6.1f} / {100*D[1][f]:+6.1f}        {100*da:+6.1f}   {100*dg:+6.1f}   {100*base:5.1f}")
    top = sorted(FEATS, key=lambda f: -abs(D[0][f]))[:3]
    print(" top 3 on discovery:", top)
    for f in top:
        tr = [r for r in rows if r["h"] == 1]
        same = (D[0][f] > 0) == (D[1][f] > 0)
        away = st.mean((abs(share(r["F"][f], r["w"]) - share(r["F"][f], r["g"])) - abs(share(r["F"][f], r["o"]) - share(r["F"][f], r["g"]))) for r in tr)
        p = [(abs(share(r["F"][f], r["w"]) - share(r["F"][f], r["g"])) - abs(share(r["F"][f], r["o"]) - share(r["F"][f], r["g"])), r["rw"] - r["ro"]) for r in tr]
        rho = spear(p); lo, hi = boot(p, spear, 1000)
        print(f"   {f}: test same sign {same}; test away-from-gold {100*away:+.1f} (+ = W20 further from gold than 8B); rho(away, recall W20-8B) {rho:+.2f} [{lo:+.2f},{hi:+.2f}] n={len(p)}")
# controls
print("\n===== controls")
ok_pos = 0
for n, rows in out.items():
    pd = SETS[n][2]; rng = random.Random(1); D = []; Dn = []
    for r in rows[:120]:
        P = json.load(open(f"{pd}/{r['qid']}.json")); rr8 = {int(k): v for k, v in P["rr8"].items()}
        F = r["F"]["user_question"]; base = sorted(rr8, key=lambda i: -rr8[i])
        boosted = sorted(rr8, key=lambda i: -(rr8[i] + 0.5 * F[i]))   # injected bias toward rounds whose user turn asks a question
        kb = pack(r["q"], r["R"], boosted, r["b"])["keep"]; k0 = pack(r["q"], r["R"], base, r["b"])["keep"]
        D.append({f: share(r["F"][f], kb) - share(r["F"][f], k0) for f in FEATS})
        perm = {f: rng.sample(r["F"][f], len(r["F"][f])) for f in FEATS}
        Dn.append({f: share(perm[f], r["w"]) - share(perm[f], r["o"]) for f in FEATS})
    m = {f: st.mean(d[f] for d in D) for f in FEATS}; rank = sorted(FEATS, key=lambda f: -abs(m[f]))
    mn = max(abs(st.mean(d[f] for d in Dn)) for f in FEATS)
    print(f" {n}: injected user_question rank {rank.index('user_question')+1} (shift {100*m['user_question']:+.1f}); top3 {rank[:3]}; permuted-label max |shift| {100*mn:.1f}")
for n, f in (("EverMem", "asker_self"), ("EverMem", "multi_mention"), ("EpBench", "date_lead")):
    rows = out[n]; d = {g: abs(st.mean(share(r["F"][g], r["w"]) - share(r["F"][g], r["o"]) for r in rows)) for g in FEATS}
    print(f" known: {n} {f} rank {sorted(FEATS, key=lambda g: -d[g]).index(f)+1} of {len(FEATS)} (shift vs 8B)")
