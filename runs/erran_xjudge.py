"""Validate the presence judge of mem/erran.py: re-judge a stratified sample of BEAM-1M lost items with gpt-6.1-sol
and report agreement with the gpt-6-luna labels (in context vs not)."""
import json, os, random, collections, sys
from concurrent.futures import ThreadPoolExecutor
from mem import llm, erran as E
OUT = "runs/beam1m/pro6"
rows = [json.loads(l) for l in open(os.path.join(OUT, "erran.jsonl"))]
ctx = {}
for c in ("hindsight", "rag", "lychee"):
    f = os.path.join(OUT, "ext", c, "ctx.jsonl")
    if os.path.exists(f):
        ctx.update({(r["qid"], c): r["context"] for r in map(json.loads, open(f))})
random.seed(0)
by = collections.defaultdict(list)
for r in rows:
    if (r["qid"], r["cond"]) in ctx: by[r["cond"]].append(r)
sample = [r for c in by for r in random.sample(by[c], min(50, len(by[c])))]
def pres(item, text):
    best = "ABSENT"
    for i in range(0, max(1, len(text)), E.PIECE):
        v = llm.call(E.PRESENT_SYS, f"Requirement: {item}\n\nText:\n{text[i:i + E.PIECE]}", effort="low",
                     max_tokens=2000, tag="erran_xjudge", model="gpt-6.1-sol").upper()
        if "PRESENT" in v and "ABSENT" not in v.replace("PRESENT", ""): return "PRESENT"
        if "PARTIAL" in v: best = "PARTIAL"
    return best
def one(r):
    return r, pres(r["item"], ctx[(r["qid"], r["cond"])]) == "PRESENT"
res = list(ThreadPoolExecutor(8).map(one, sample))
json.dump([{"qid": r["qid"], "cond": r["cond"], "item": r["item"], "luna_in": r["where"] == "reader", "sol_in": s}
           for r, s in res], open("runs/beam1m/pro6/erran_xjudge.json", "w"), indent=1)
for c in by:
    R = [(r["where"] == "reader", s) for r, s in res if r["cond"] == c]
    a = sum(x == y for x, y in R) / len(R)
    p1 = sum(x for x, _ in R) / len(R); p2 = sum(y for _, y in R) / len(R); pe = p1 * p2 + (1 - p1) * (1 - p2)
    print(c, len(R), "agree", round(a, 3), "kappa", round((a - pe) / (1 - pe), 3) if pe < 1 else None, "luna_in", round(p1, 2), "sol_in", round(p2, 2))
