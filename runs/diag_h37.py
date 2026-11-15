"""Where does asp_full lose to hindsight on 1M area questions? For rubric items hindsight got (>= .5) and asp_full did
not (gen 1), is the item's information present in the asp_full context (-> reader) or not (-> coverage)?"""
import collections, json
from concurrent.futures import ThreadPoolExecutor
from mem import erran as E
O = "runs/beam1m/pro6/"
J = {(r["qid"], r["cond"]): r for r in map(json.loads, open(O + "judged.jsonl"))}
ctx = {r["qid"]: r["context"] for r in map(json.loads, open(O + "ext/asp_full/ctx.jsonl"))}
jobs = []
for (q, c), r in J.items():
    if c == "asp_full" and (q, "hindsight") in J:
        h = J[(q, "hindsight")]
        for it, sa, sh in zip(r["rubric"], r["scores"], h["scores"]):
            if sh >= 0.5 and sa < 0.5:
                jobs.append((q, r["ability"], it))
res = list(ThreadPoolExecutor(8).map(lambda j: (j, E.present(j[2], ctx[j[0]])), jobs))
print(len(jobs), "items hindsight got, asp_full missed")
print(collections.Counter((j[1], v) for j, v in res))
json.dump([dict(qid=j[0], ability=j[1], item=j[2], where=v) for j, v in res], open(O + "diag_h37.json", "w"), indent=1)
