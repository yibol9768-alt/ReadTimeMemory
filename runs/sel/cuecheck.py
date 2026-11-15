"""Every number in docs/CUES_2026-11-01.md comes from this script (11-01). usage: python3 cuecheck.py > cue_report.txt. CPU only, dev sets only."""
import collections, datetime, json, random, re, interv as V
S = {"BEAM": "q_dv_beam.json", "LME": "q_dv_lme_raw.json", "EverMem": "q_dv_ev.json", "EpBench": "q_dv_ep.json", "World": "env_ws4.json"}
has = lambda q: any(t.get("has_answer") for t in V.turns(q))
def boot(rows):
    r = random.Random(0); B = sorted(sum(rows[r.randrange(len(rows))] for _ in rows) / len(rows) for _ in range(2000))
    return f"{100 * sum(rows) / len(rows):+.1f} [{100 * B[50]:+.1f},{100 * B[1949]:+.1f}] n={len(rows)}"
print("## 1 phrase rates, user rounds, strict rules")
for nm, f in S.items():
    if nm == "EpBench": continue
    Q = [q for q in json.load(open(f)) if has(q)]
    if len(Q) > 150: Q = random.Random(1).sample(Q, 150)
    for lab, P in (("by the way", V.TPL[1]), ("opener", V.OPEN), ("any", None)):
        c = collections.Counter(); rows = []; same = []
        for q in Q:
            a = b = d = e = f2 = g2 = 0
            for s in q["haystack_sessions"]:
                U = [t for t in s if t["role"] == "user"]; gs = any(t.get("has_answer") for t in U)
                for t in U:
                    h = any(p.search(t["content"]) for p in V.TPL) if P is None else bool(P.search(t["content"]))
                    if t.get("has_answer"): a += h; b += 1
                    elif gs: d += h; e += 1
                    else: f2 += h; g2 += 1
            c.update(gold=a, goldn=b, same=d, samen=e, other=f2, othern=g2)
            if b and e + g2: rows.append(a / b - (d + f2) / (e + g2))
            if b and e: same.append(a / b - d / e)
        p = lambda x, y: f"{c[x]}/{c[y]}={100 * c[x] / max(c[y], 1):.1f}%"
        print(f"{nm} {lab}: gold {p('gold', 'goldn')}; same answer sessions {p('same', 'samen')}; other sessions {p('other', 'othern')}; diff vs all other {boot(rows)}; diff vs same session {boot(same) if same else 'n/a'}")
print("## 2 coverage: kept questions, gold rounds touched, reasons")
for nm, f in S.items():
    A = [q for q in json.load(open(f)) if has(q)]
    if len(A) > 200: A = random.Random(1).sample(A, 200)
    O = {q["question_id"]: q for q in A}
    for fn in V.FN:
        out, log = V.run(fn, A); gt = gn = bad = 0; hist = {}; inc = 0
        for e in out:
            o = O[e["orig_question_id"]]
            a = [(t["role"], bool(t.get("has_answer"))) for t in V.turns(o)]; b = [(t["role"], bool(t.get("has_answer"))) for t in V.turns(e)]
            bad += (sorted(a) != sorted(b)) if fn == "shuffle" else (a != b)
            if fn != "shuffle":
                for x, y in zip(V.turns(o), V.turns(e)):
                    if x.get("has_answer"): gn += 1; gt += x["content"] != y["content"]
            k = V.hkey(o); v = V.hkey(e)
            if hist.setdefault(k, v) != v: inc += 1
        print(f"{nm} {fn}: kept {len(out)}/{len(A)}; gold touched {gt}/{gn}; structure bad {bad}; same-history mismatch {inc}; rejected {collections.Counter(v.split(':')[0][:40] for v in log['rejected'].values()).most_common(4)}")
print("## 3 dates move by exactly the same days")
FULL = re.compile(r"\b(" + V.MN + r") (\d{1,2}), (\d{4})\b")
for nm, f in S.items():
    A = [q for q in json.load(open(f)) if has(q)]; A = random.Random(1).sample(A, min(40, len(A))); good = tot = stamp = stot = 0
    def p(x):
        x = str(x); m = re.match(r"(\d{4})[-/](\d{2})[-/](\d{2})", x)
        if m: return datetime.date(*map(int, m.groups()))
        m = re.match(r"(" + V.MN + r")-(\d{2})-(\d{4})", x)
        return datetime.date(int(m[3]), V.MONTHS.index(m[1]) + 1, int(m[2])) if m else None
    for q in A:
        try: e, info = V.shift(q)
        except V.Reject: continue
        d = info["days"]
        for a, b in zip(q["haystack_dates"], e["haystack_dates"]):
            u, v = p(a), p(b)
            if u and v: stot += 1; stamp += (v - u).days == d
        for a, b in zip(V.turns(q), V.turns(e)):
            x, y = FULL.findall(a["content"]), FULL.findall(b["content"])
            if len(x) == len(y):
                for m, n in zip(x, y):
                    tot += 1; good += (datetime.date(int(n[2]), V.MONTHS.index(n[0]) + 1, int(n[1])) - datetime.date(int(m[2]), V.MONTHS.index(m[0]) + 1, int(m[1]))).days == d
    print(f"{nm}: session stamps {stamp}/{stot}; full text dates {good}/{tot}")
