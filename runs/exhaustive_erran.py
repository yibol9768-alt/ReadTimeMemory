# Offline (no LLM): share of lost rubric points by where they were lost, split by whether the question needs
# exhaustive evidence (count / list-all / latest value / order / absence), from the existing erran.jsonl (dev, luna).
import json, re, collections, sys
EXH = re.compile(r"\b(how many|how often|number of|count|total|all (the|of)|every|list|each time|times|sequence|order|"
                 r"first|last|latest|most recent|current(ly)?|now|still|ever|never|any (time|point)|chronolog|summar|"
                 r"overall|progress|evolv|changed?|over time)\b", re.I)
Q_ALL = {}
for s in ["beam", "beam1m"]:
    Q = Q_ALL[s] = {}
    for l in open(f"runs/{s}/pro6/answers.jsonl", errors="replace"):
        r = json.loads(l); Q[r["qid"]] = r["q"]
    E = [json.loads(l) for l in open(f"runs/{s}/pro6/erran.jsonl")]
    nq = collections.Counter()
    for q, t in Q.items(): nq[bool(EXH.search(t))] += 1
    print(f"\n== {s}: questions exhaustive {nq[True]} / other {nq[False]}")
    for cond in ["lychee", "hindsight", "rag"]:
        T = collections.defaultdict(collections.Counter)
        for r in E:
            if r["cond"] != cond: continue
            k = "exh" if EXH.search(Q.get(r["qid"], "")) else "oth"
            T[k][r["where"]] += 1 - r["score"]
        for k in ["exh", "oth"]:
            tot = sum(T[k].values())
            if tot: print(f"  {cond:9s} {k}: lost {tot:6.1f} items | " + "  ".join(f"{w} {100*v/tot:4.0f}%" for w, v in sorted(T[k].items())))
    # by ability for lychee (records on disk -> write / retrieval / reader split)
    A = collections.defaultdict(collections.Counter)
    for r in E:
        if r["cond"] == "lychee": A[r["ability"]][r["where"]] += 1 - r["score"]
    print("  lychee by ability:")
    for a, c in sorted(A.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(c.values()); print(f"    {a:26s} {tot:5.1f} | " + "  ".join(f"{w} {100*v/tot:3.0f}%" for w, v in sorted(c.items())))

# Upper bound: lychee 1M/100K dev score if every item lost at retrieval were recovered (item credit -> 1).
import ast
for s in ["beam", "beam1m"]:
    ans = {}; rub = {}
    for l in open(f"runs/{s}/pro6/answers.jsonl", errors="replace"):
        r = json.loads(l)
        if r["cond"] == "lychee":
            x = r["rubric"]
            rub[r["qid"]] = len(x) if isinstance(x, list) else len(ast.literal_eval(x))
    sc = {}
    for l in open(f"runs/{s}/pro6/judged.jsonl", errors="replace"):
        r = json.loads(l)
        if r["cond"] == "lychee": sc[r["qid"]] = r["score"]
    gain = collections.Counter(); seen = set()
    for l in open(f"runs/{s}/pro6/erran.jsonl"):
        r = json.loads(l)
        if r["cond"] != "lychee" or r["where"] != "retrieval" or (r["qid"], r["item"]) in seen: continue
        seen.add((r["qid"], r["item"])); gain[r["qid"]] += (1 - r["score"]) / rub.get(r["qid"], 1)
    gain = collections.Counter({q: min(v, 1 - sc.get(q, 0)) for q, v in gain.items()})
    qs = list(sc); base = 100 * sum(sc.values()) / len(qs)
    print(f"{s}: lychee(8K) dev {base:.1f}; +retrieval-loss recovered {base + 100*sum(gain.values())/len(qs):.1f}"
          f"  (exhaustive qs only: +{100*sum(v for q,v in gain.items() if EXH.search(Q_ALL[s].get(q,'')))/len(qs):.1f})")
