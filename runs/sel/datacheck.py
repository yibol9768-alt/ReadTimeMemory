"""Training-data checks (10-31). For one or more datasets in the LME-style format (haystack_sessions with has_answer),
reports the statistics a synthetic set must match before it is used for training. Reference = BEAM-100K.
 style AUC   : logistic regression on 7 writing features only (no content); fit on the set itself (higher = shortcut)
 cross AUC   : style model fit on the reference and applied to the set, and the set's model applied to the reference
 lengths     : user-message chars of gold / non-gold rounds (median), gold/non ratio
 gold        : gold rounds per question (median), share of history that is gold
 overlap     : query-word overlap of gold vs non-gold user messages (lexical shortcut)
 templates   : distinct question openings (first 5 words) / questions
usage: alfenv python datacheck.py REF.json SET1.json [SET2.json ...] [--maxq 150]"""
import json, re, sys, statistics as st, collections, random
import numpy as np
SW = set("the a an of to in and or my i me we our is was were be been for on with about how did what when which that this it as at by from order mention only items have has do does any all project discussions progress over time give summary can you".split())
def toks(s): return {w for w in re.findall(r"[a-z][a-z0-9\-]+", s.lower()) if w not in SW and len(w) > 2}
def feat(u):
    l = u.lower()
    return [len(u) / 1000, int("?" in u), int(bool(re.search(r"\b(quick update|update on|we just|i just|finally|it.s been|so far|managed to|turned out|ended up)\b", l))),
            int(bool(re.search(r"\b(how (do|can|should) i|what.s the best way|can you help)\b", l))), 10 * sum(c.isdigit() for c in u) / max(1, len(u)),
            int(l.startswith(("hey", "hi ", "hmm", "so ", "ok"))), u.count("\n") / 10]
def auc(y, s):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); p = y == 1
    return float((r[p].sum() - p.sum() * (p.sum() - 1) / 2) / (p.sum() * (~p).sum()))
def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = np.c_[(X - mu) / sd, np.ones(len(X))]; w = np.zeros(Z.shape[1])
    cw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    for _ in range(2000): p = 1 / (1 + np.exp(-Z @ w)); w -= 0.5 * Z.T @ ((p - y) * cw) / len(y)
    return lambda A: np.c_[(A - mu) / sd, np.ones(len(A))] @ w
def load(fn, maxq):
    qs = json.load(open(fn)); random.Random(0).shuffle(qs); qs = [q for q in qs if any(t.get("has_answer") in (True, 1, "True") for s in q["haystack_sessions"] for t in s)][:maxq]
    X, y, L, G, OV, op, seen = [], [], collections.defaultdict(list), [], collections.defaultdict(list), collections.Counter(), set()
    for q in qs:
        op[" ".join(q["question"].lower().split()[:5])] += 1; qt = toks(q["question"]); n = g = 0
        for s in q["haystack_sessions"]:
            for t in s:
                if t["role"] != "user": continue
                a = int(t.get("has_answer") in (True, 1, "True")); u = re.sub(r"\s*->->\s*\d+\s*,\s*\d+", "", t["content"])
                n += 1; g += a; L[a].append(len(u)); OV[a].append(len(qt & toks(u)) / max(1, len(qt)))
                h = (q["question_id"].split("_")[0], u[:200])           # one row per distinct message per user
                if h in seen and not a: continue
                seen.add(h); X.append(feat(u)); y.append(a)
        G.append((g, g / max(1, n)))
    return np.array(X), np.array(y), L, G, OV, op, len(qs)
maxq = int(sys.argv[sys.argv.index("--maxq") + 1]) if "--maxq" in sys.argv else 150
files = [a for a in sys.argv[1:] if a.endswith(".json")]
D = {f: load(f, maxq) for f in files}; ref = files[0]; mref = fit(D[ref][0], D[ref][1])
print(f"{'set':22s} {'nq':>4s} {'styleAUC':>8s} {'refModel':>8s} {'onRef':>6s} {'len g/n':>11s} {'ratio':>5s} {'gold/q':>6s} {'gold%':>5s} {'ovl g/n':>11s} {'tmpl':>5s}")
for f, (X, y, L, G, OV, op, nq) in D.items():
    m = fit(X, y); own = auc(y, m(X)); rm = auc(y, mref(X)); onref = auc(D[ref][1], m(D[ref][0]))
    lg, ln = st.median(L[1]), st.median(L[0])
    print(f"{f.split('/')[-1][:22]:22s} {nq:4d} {own:8.3f} {rm:8.3f} {onref:6.3f} {lg:5.0f}/{ln:<5.0f} {lg / ln:5.2f} {st.median(g for g, _ in G):6.1f} {100 * st.mean(r for _, r in G):5.1f} {st.mean(OV[1]):.2f}/{st.mean(OV[0]):.2f}    {len(op) / nq:5.2f}")
