"""Evidence recall of each retriever under the 8000-character budget, no LLM calls: does the context contain
every gold evidence turn (via the turns each record cites)?"""
import json, sys, collections
sys.argv = ["x"]
from mem import locomo_exp as L, retrieve as R
d = json.load(open("data/locomo10.json"))
qs = L.questions(d, 50)
recs = {ci: json.load(open(f"runs/locomo/mem_{ci}.json"))["rec_sess"] for ci in range(10)}
docs = {ci: [r["text"] for r in recs[ci]] for ci in recs}
R.precompute_ranks(list(docs.values()), [(docs[q["conv"]], q["question"]) for q in qs])
J = {(r["qid"], r["cond"]): r["correct"] for r in map(json.loads, open("runs/locomo/judged.jsonl"))}
def shown(order, ci):
    n, src = 0, set()
    for i in order:
        x = len(recs[ci][i]["text"]) + 3
        if n + x > L.BUDGET: break
        n += x; src |= set(recs[ci][i]["sources"])
    return src
res = collections.defaultdict(lambda: collections.Counter())
for q in qs:
    ev = {e.strip() for e in q.get("evidence", []) for e in e.split(";") if e.strip()}
    if not ev: continue
    ci = q["conv"]; bm = L.BM25(docs[ci]).rank(q["question"]); de = R.dense_rank(q["question"], docs[ci])
    for name, order in [("bm25", bm), ("dense", de), ("hybrid", R.rrf([bm, de]))]:
        s = shown(order, ci)
        k = "all" if ev <= s else ("part" if ev & s else "none")
        res[(name, 0)][k] += 1; res[(name, q["category"])][k] += 1
    if not (ev <= shown(bm, ci)) and not J[(q["qid"], "rec_sess")]:
        for name, order in [("dense", de), ("hybrid", R.rrf([bm, de]))]:
            res[("rescue_" + name, 0)]["all" if ev <= shown(order, ci) else "not"] += 1
for (name, c), v in sorted(res.items()):
    tot = sum(v.values())
    print(f"{name:14s} {L.CAT.get(c,'all'):12s} n={tot:3d} " + "  ".join(f"{k}={v[k]}" for k in ["all","part","none","not"] if k in v) + f"   all%={100*v['all']/tot:.0f}")
