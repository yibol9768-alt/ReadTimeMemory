"""AFLite-style group filter for experiment X (10-31): drop training groups whose positive is ranked first by a
content-free shortcut model (style + length features of the whole document, user and assistant). The model is fit
with 2-fold cross-fitting over worlds (users) so a group is never scored by a model trained on it.
usage: python3 afl.py IN.jsonl OUT.jsonl [TOP=1]  (drop groups whose positive is within the shortcut model's top TOP)"""
import json, re, sys, collections
import numpy as np
def feat(d):
    body = d.split("<Document>:")[-1]; u, _, a = body.partition("Assistant:"); l = u.lower()
    return [len(u) / 1000, len(a) / 1000, int("?" in u), int(bool(re.search(r"\\b(quick update|update on|we just|i just|finally|it.s been|so far|managed to|turned out|ended up)\\b", l))),
            int(bool(re.search(r"\\b(how (do|can|should) i|what.s the best way|can you help)\\b", l))), 10 * sum(c.isdigit() for c in u) / max(1, len(u)),
            int(l.strip().startswith(("user: hey", "user: hi", "user: hmm", "user: so ", "user: ok", "hey", "hmm"))), u.count("\n") / 10, a.count("\n") / 50, int("```" in a)]
def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = np.c_[(X - mu) / sd, np.ones(len(X))]; w = np.zeros(Z.shape[1])
    cw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    for _ in range(1500): p = 1 / (1 + np.exp(-Z @ w)); w -= 0.5 * Z.T @ ((p - y) * cw) / len(y)
    return lambda A: np.c_[(A - mu) / sd, np.ones(len(A))] @ w
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rows = [json.loads(l) for l in open(sys.argv[1])]
users = sorted({r["qid"].split("_")[0] for r in rows}); fold = {u: i % 2 for i, u in enumerate(users)}
F = [np.array([feat(d) for d in r["docs"]]) for r in rows]
keep = []; top1 = collections.Counter()
for k in (0, 1):
    tr = [i for i, r in enumerate(rows) if fold[r["qid"].split("_")[0]] != k]
    X = np.concatenate([F[i] for i in tr]); y = np.concatenate([[1] + [0] * (len(F[i]) - 1) for i in tr]); m = fit(X, y)
    for i, r in enumerate(rows):
        if fold[r["qid"].split("_")[0]] != k: continue
        s = m(F[i]); first = int((s > s[0]).sum()) < TOP; top1[first] += 1
        if not first: keep.append(i)
open(sys.argv[2], "w").writelines(json.dumps(rows[i]) + "\n" for i in sorted(keep))
print(f"groups {len(rows)}  positive ranked in top {TOP} by shortcut model: {top1[True]} ({100 * top1[True] / len(rows):.0f}%, chance {100 * TOP / len(F[0]):.0f}%)  kept {len(keep)}")
