"""Summarise runs/lme_sm/{S,M}.jsonl: evidence recall@k (all gold sessions in top-k) per question type, S vs M,
and what kind of sessions crowd the top-20 in M."""
import json, collections
R = {t: {json.loads(l)["qid"]: json.loads(l) for l in open(f"runs/lme_sm/{t}.jsonl")} for t in "SM"}
qs = sorted(set(R["S"]) & set(R["M"]))
K = [5, 10, 20, 50]


def hit(r, k):
    g = r["gold_ranks"]
    return bool(g) and max(g) <= k


by = collections.defaultdict(list)
for q in qs:
    t = R["S"][q]["type"] + ("_abs" if R["S"][q]["abs"] else "")
    by[t].append(q); by["ALL"].append(q)
print(f"{'type':34s} {'n':>4s} " + " ".join(f"S@{k:<3d} M@{k:<3d}" for k in K) + "  medrank S / M")
for t, Q in sorted(by.items(), key=lambda x: (x[0] == "ALL", x[0])):
    Q = [q for q in Q if R["S"][q]["gold_ranks"]]
    if not Q:
        continue
    cells = []
    for k in K:
        s = sum(hit(R["S"][q], k) for q in Q) / len(Q); m = sum(hit(R["M"][q], k) for q in Q) / len(Q)
        cells.append(f"{s:5.2f} {m:5.2f}")
    mr = lambda T: sorted(max(R[T][q]["gold_ranks"]) for q in Q)[len(Q) // 2]
    print(f"{t:34s} {len(Q):4d} " + " ".join(cells) + f"   {mr('S')} / {mr('M')}")
ns = lambda T: sum(R[T][q]["n_sess"] for q in qs) / len(qs)
nt = lambda T: sum(R[T][q]["n_tok"] for q in qs) / len(qs)
print(f"\nsessions/question S {ns('S'):.0f} M {ns('M'):.0f}; content words/question S {nt('S'):.0f} M {nt('M'):.0f}")
for T in "SM":
    kinds = collections.Counter(); top = collections.Counter()
    for q in qs:
        kinds.update(R[T][q]["kinds"]); top.update(R[T][q]["top20_kinds"])
    print(T, "haystack mix", {k: round(v / sum(kinds.values()), 3) for k, v in kinds.items()},
          "| BM25 top-20 mix", {k: round(v / sum(top.values()), 3) for k, v in top.items()})
