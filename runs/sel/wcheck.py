"""Group-level shortcut check (10-31), question-free. Positive = gold rounds, negative = non-gold rounds, repeats and follow-ups left out of both.
Features: style (user), assistant length, position in history; bag-of-words hashed. Fit on all worlds but one, test on the left-out conversation.
Also BEAM-100K with the same procedure. usage: alfenv python wcheck.py"""
import json, re, sys, collections, hashlib
import numpy as np
exec(open("datacheck.py").read().split("def load(")[0])   # feat, fit; datacheck runs on import
fit_np = fit
try:
    import torch
    if torch.cuda.is_available():
        def fit(X, y):
            """same full-batch updates as datacheck.fit, on the GPU (11-01: the CPU version took over 13 minutes on 21 worlds)"""
            d = dict(dtype=torch.float64, device="cuda"); X = torch.as_tensor(X, **d); y = torch.as_tensor(y, **d); mu, sd = X.mean(0), X.std(0, unbiased=False) + 1e-6
            Z = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, **d)], 1); w = torch.zeros(Z.shape[1], **d); m = y.mean(); cw = torch.where(y == 1, 0.5 / m, 0.5 / (1 - m))
            for _ in range(2000): p = 1 / (1 + torch.exp(-Z @ w)); w -= 0.5 * Z.T @ ((p - y) * cw) / len(y)
            return lambda A: ((torch.as_tensor(A, **d) - mu) / sd @ w[:-1] + w[-1]).cpu().numpy()
except ImportError: pass
def auc(y, s):
    """rank AUC with average ranks for ties"""
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); end = np.cumsum(cnt); r = (end - (cnt - 1) / 2 - 1)[inv]; p = y == 1
    return float((r[p].sum() - p.sum() * (p.sum() - 1) / 2) / (p.sum() * (~p).sum()))
def rows(fn, pooldir, key):
    Q = json.load(open(fn)); by = collections.defaultdict(lambda: {"g": collections.Counter(), "q": None})
    for q in Q:
        p = json.load(open(f"{pooldir}/{q['question_id']}.json")); b = by[key(q)]; b["q"] = b["q"] or q
        for i, x in enumerate(p["gold"]):
            if x: b["g"][i] += 1
    out = []
    for c, b in by.items():
        q = b["q"]; R = []
        for s in q["haystack_sessions"]:
            i = 0
            while i < len(s):
                # Match k1.rounds_of and the pool's gold indices: only a user
                # immediately followed by an assistant forms a two-message round.
                paired = s[i]["role"] == "user" and i + 1 < len(s) and s[i + 1]["role"] == "assistant"
                a = s[i + 1]["content"] if paired else ""
                R.append((s[i]["content"], a, s[i].get("kind"))); i += 2 if paired else 1
        for i, (u, a, k) in enumerate(R):
            if k in ("repeat", "fu"): continue
            out.append((c, u, a, i / len(R), int(b["g"][i] > 0)))
    return out
def bow(t, D=4096):
    v = np.zeros(D)
    for w in re.findall(r"[a-z][a-z0-9\-]+", t.lower()): v[int(hashlib.md5(w.encode()).hexdigest(), 16) % D] += 1
    return np.log1p(v)
FAIL = []
def report(name, R):
    y = np.array([r[4] for r in R]); gr = np.array([r[0] for r in R]); uniq = sorted(set(gr))
    F = {"style": np.array([feat(r[1]) for r in R]), "asst": np.array([[len(r[2]) / 1000] for r in R]), "pos": np.array([[r[3]] for r in R]), "style+asst+pos": None}
    F["style+asst+pos"] = np.c_[F["style"], F["asst"], F["pos"]]; B = np.array([bow(r[1] + " " + r[2]) for r in R]); res = {}
    for n, X in list(F.items()) + [("bow", B)]:
        s = np.zeros(len(y))
        folds = [uniq[i::5] for i in range(min(5, len(uniq)))]
        for f in folds:
            te = np.isin(gr, f); tr = ~te
            if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0: continue
            s[te] = fit(X[tr], y[tr])(X[te])
        res[n] = round(auc(y, s), 3)
    if name != "BEAM100K":
        fam = np.array([int(g[1:]) % 3 for g in gr]); P = np.array([r[3] for r in R])
        byf = {int(f): round(auc(y[fam == f], P[fam == f]), 3) for f in set(fam)}; print(" position AUC per model family", byf)
        if max(abs(v - .5) for v in byf.values()) > .1: FAIL.append(name + " family-position")
    if name != "BEAM100K" and (res["style"] > .56 or res["bow"] > .58 or res["pos"] > .55): FAIL.append(name)
    print(name, "rounds", len(y), "gold share", round(y.mean(), 3), res, "asst len gold/non", round(np.median([len(r[2]) for r in R if r[4]])), round(np.median([len(r[2]) for r in R if not r[4]])), "user len", round(np.median([len(r[1]) for r in R if r[4]])), round(np.median([len(r[1]) for r in R if not r[4]])), "pos gold/non", round(np.mean([r[3] for r in R if r[4]]), 3), round(np.mean([r[3] for r in R if not r[4]]), 3))
report("BEAM100K", rows("beam100k_g.json", "pool_c", lambda q: q["question_id"].split("_")[0]))
for tag in sys.argv[1:]:
    report(tag, rows(f"env_{tag}.json", f"pool_{tag}", lambda q: q["question_id"].split("_")[0]))
    Q = json.load(open(f"env_{tag}.json")); seen = {}
    for q in Q: seen[q["question_id"].split("_")[0]] = q
    k = collections.Counter(t.get("kind") for q in seen.values() for s in q["haystack_sessions"] for t in s if t["role"] == "user"); print(" kinds", dict(k), "fu share", round(k["fu"] / sum(k.values()), 3))
sys.exit(1 if FAIL else 0)
