"""Step 0: question-agnostic trivial rankers on Crucible gold vs the Stage-1 model (W20).
Same round parsing and packing as k1. Recall = gold_in / gold_total, mean over questions.
usage: python3 -I triv.py K1DIR TRAIN_ENV TEST_ENV CTX_W20 [TEST_ENV CTX_W20 ...]"""
import sys, json, random, collections, re
sys.path.insert(0, sys.argv[1]); import k1
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BUD = [1000, 2000, 4000]
NB = 10
MARK = ["by the way", "btw", "actually", "remember", "update", "changed", "decided", "now", "instead", "anyway"]


def feats(R, i):
    t = R[i][2]; lo = t.lower(); n = len(R)
    u = t.split("\nAssistant:")[0]
    return [i / max(1, n - 1), len(t) / 1000, len(u) / 500, sum(c.isdigit() for c in t) / 20,
            t.count("$") > 0, "?" in u, t.count("\n"), lo.count(" i "), "[group" in lo,
            *[m in lo for m in MARK]]


def gold_rounds(R): return [i for i, r in enumerate(R) if r[3] > 0]


def load(p): return json.load(open(p))


train = load(sys.argv[2])
# position prior and style classifier, fit on training worlds only
hist = np.ones(NB); X, y = [], []
rng = random.Random(0)
for q in train:
    R = k1.rounds_of(q); g = set(gold_rounds(R)); n = len(R)
    for i in g: hist[min(NB - 1, int(NB * i / n))] += 1
    neg = rng.sample([i for i in range(n) if i not in g], min(len(g) * 5, n - len(g)))
    for i in list(g) + neg: X.append(feats(R, i)); y.append(int(i in g))
pos_prior = hist / hist.sum()
clf = LogisticRegression(max_iter=2000).fit(np.array(X, float), y)
print("train questions", len(train), "pos prior by decile", np.round(pos_prior, 3).tolist())
print("style clf train AUC", round(roc_auc_score(y, clf.predict_proba(np.array(X, float))[:, 1]), 3))
del train

for k in range(3, len(sys.argv), 2):
    test = load(sys.argv[k]); ctx = collections.defaultdict(dict)
    for l in open(sys.argv[k + 1]):
        d = json.loads(l); ctx[d["qid"]][d["budget"]] = d
    test = [q for q in test if q["question_id"] in ctx]
    res = collections.defaultdict(lambda: collections.defaultdict(list)); aucs = []
    mism = 0
    for q in test:
        R = k1.rounds_of(q); n = len(R); tp = q["question_type"]
        g = set(gold_rounds(R))
        Xq = np.array([feats(R, i) for i in range(n)], float); ps = clf.predict_proba(Xq)[:, 1]
        if 0 < len(g) < n: aucs.append(roc_auc_score([int(i in g) for i in range(n)], ps))
        bm = k1.BM25([k1.toks(r[2]) for r in R]).scores(k1.toks(q["question"]))
        orders = {
            "recency": list(range(n - 1, -1, -1)),
            "earliest": list(range(n)),
            "position prior": sorted(range(n), key=lambda i: (-pos_prior[min(NB - 1, int(NB * i / n))], -i)),
            "style clf": sorted(range(n), key=lambda i: -ps[i]),
            "BM25 (uses question)": sorted(range(n), key=lambda i: -bm[i]),
        }
        for B in BUD:
            for name, o in orders.items():
                p = k1.pack(q, R, o, B); res[name][B].append((tp, p["gold_in"] / max(1, p["gold_total"])))
            rr = []
            for s in range(5):
                o = list(range(n)); random.Random(s).shuffle(o); p = k1.pack(q, R, o, B); rr.append(p["gold_in"] / max(1, p["gold_total"]))
            res["random (5 seeds)"][B].append((tp, sum(rr) / 5))
            w = ctx[q["question_id"]].get(B)
            if w:
                if w["gold_total"] != sum(r[3] for r in R): mism += 1
                res["Stage 1 (W20)"][B].append((tp, w["gold_in"] / max(1, w["gold_total"])))
    print(f"\n== {sys.argv[k]}  questions {len(test)}  gold_total mismatches {mism}  style per-question AUC {np.mean(aucs):.3f}")
    print(f"{'ranker':24s}" + "".join(f"{B:>8d}" for B in BUD))
    for name in res:
        print(f"{name:24s}" + "".join(f"{100*np.mean([v for _, v in res[name][B]]):8.1f}" for B in BUD))
    types = sorted({t for t, _ in res["recency"][2000]})
    print("\nper type at 2k:", f"{'':24s}" + "".join(f"{t[:10]:>11s}" for t in types))
    for name in res:
        row = [np.mean([v for t, v in res[name][2000] if t == tt] or [float('nan')]) for tt in types]
        print(f"{name:24s}" + "".join(f"{100*x:11.1f}" for x in row))
