"""Gold-round carrier signal (D3 v8 draft). For each question type x gold-round carrier: per question, mean over budgets of (oracle gold recall - selector gold recall), counted only on gold rounds of that carrier.
Oracle = shared-budget shortest-first fill over all gold in the fused top-600, as ecell.py. Also prints plain selector miss rate and oracle miss rate, because shortest-first understates long gold.
usage: python3 gcell.py QFILE POOLDIR BUDGETS TAG [--min-n 20]"""
import argparse, collections, json, statistics as st, sys
sys.path.insert(0, "."); from k1 import rounds_of, ntok
ap = argparse.ArgumentParser(); ap.add_argument("qfile"); ap.add_argument("pooldir"); ap.add_argument("budgets"); ap.add_argument("tag"); ap.add_argument("--min-n", type=int, default=20); a = ap.parse_args()
Bs = [int(x) for x in a.budgets.split(",")]; Q = json.load(open(a.qfile))
C = {(r["qid"], r["budget"]): r for r in map(json.loads, open(f"ctx_{a.tag}.jsonl"))}
def fill(R, order, B):
    keep, used = [], 0
    for i in order:
        n_ = ntok(R[i][2])
        if used + n_ <= B: keep.append(i); used += n_
    return set(keep)
def carriers(q):
    out, i = {}, 0
    for s in q["haystack_sessions"]:
        j = 0
        while j < len(s):
            n = 2 if j + 1 < len(s) and s[j]["role"] == "user" and s[j + 1]["role"] == "assistant" else 1
            out[i] = "group" if s[j].get("carrier") == "group" else "long" if s[j].get("carrier") == "long" else "dialog"; i += 1; j += n
    return out
V = collections.defaultdict(dict)
for q in Q:
    R = rounds_of(q); g = {i for i, r in enumerate(R) if r[3]}
    if not g: continue
    if any((q["question_id"], b) not in C for b in Bs): raise SystemExit(f"INVALID missing {q['question_id']}")
    cr = carriers(q); assert len(cr) == len(R), (len(cr), len(R))
    F = json.load(open(f"{a.pooldir}/{q['question_id']}.json"))["fused"][:600]
    orc = sorted([i for i in F if i in g], key=lambda i: (ntok(R[i][2]), i))
    for c in ("dialog", "group", "long"):
        gc = {i for i in g if cr[i] == c}
        if not gc: continue
        o, x = [], []
        for b in Bs:
            o.append(len(fill(R, orc, b) & gc) / len(gc)); x.append(len(set(C[(q["question_id"], b)]["keep"]) & gc) / len(gc))
        V[(q["question_type"], c)][q["question_id"]] = (st.mean(o), st.mean(x))
rows = []
for k, d in V.items():
    w = {q.split("_")[0] for q in d}; O = st.mean(v[0] for v in d.values()); X = st.mean(v[1] for v in d.values())
    rows.append((k, len(d), len(w), O, X, O - X))
rows.sort(key=lambda r: (not (r[1] >= a.min_n and r[2] >= 2), -r[5], -r[1]))
print(f"{'cell':32s}  n worlds oracle selector gap")
for k, n, w, O, X, G in rows:
    print(f"{str(k):32s} {n:3d} {w:3d} {100*O:6.1f} {100*X:6.1f} {100*G:6.1f}" + ("" if n >= a.min_n and w >= 2 else "  (too few)"))
