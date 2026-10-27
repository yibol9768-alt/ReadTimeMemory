"""D1 (METHOD_TREE, 1M): for every rubric item lychee lost at "retrieval" (erran.jsonl; abstention excluded),
find the ledger record(s) that state it (dense top-15 by item text, luna picks), and report where the best one sits
in lychee's fused order versus how many records the 8K context actually showed.
  shown       -> erran mislabel (record was in the context)
  near        -> rank <= 3x shown: budget-limited (a denser view of the same order would reach it)
  far         -> rank beyond 3x shown: the query routes do not reach it (a planning / routing problem)
Usage: V3_BEAM_SPLIT=1m PYTHONPATH=. python3 runs/diag_rank1m.py
"""
import collections, json, os, sys, argparse
from concurrent.futures import ThreadPoolExecutor
from mem import llm, beam as B, lychee as Y, retrieve as R

B.ARGS = argparse.Namespace(convs=5)
OUT = B.OUT
convs = B.load()
qs = {q["qid"]: q for q in B.questions(convs)}
recs = {ci: json.load(open(os.path.join(OUT, "lychee", f"mem_{ci}.json")))["records"] for ci in B.picked(convs)}
ctx = {r["qid"]: r["context"] for r in map(json.loads, open(os.path.join(OUT, "ext", "lychee", "ctx.jsonl")))}
items = [r for r in map(json.loads, open(os.path.join(OUT, "erran.jsonl")))
         if r["cond"] == "lychee" and r["where"] == "retrieval" and r["ability"] != "abstention"]
print("retrieval-lost items", len(items), flush=True)
plans = {qid: Y.plan(qs[qid]["question"]) for qid in {r["qid"] for r in items}}
docs = {ci: [r["text"] for r in rs] for ci, rs in recs.items()}
R.precompute_ranks(list(docs.values()),
                   [(docs[qs[q]["conv"]], x) for q, p in plans.items() for x in p["queries"]] +
                   [(docs[qs[r["qid"]]["conv"]], r["item"]) for r in items])
PICK = ("You get a requirement and numbered memory records. Output the numbers of the records that state the "
        "information the requirement needs, as a JSON list (e.g. [3, 7]); [] if none.")


def one(r):
    q = qs[r["qid"]]; rs = recs[q["conv"]]
    top = R.dense_rank(r["item"], docs[q["conv"]])[:15]
    v = llm.call(PICK, f"Requirement: {r['item']}\n\nRecords:\n" + "\n".join(f"{k}. {rs[i]['text']}" for k, i in
                 enumerate(top)), effort="low", max_tokens=2000, tag="diag_pick")
    try:
        gold = [top[k] for k in json.loads(v[v.index("["):v.rindex("]") + 1]) if isinstance(k, int) and k < len(top)]
    except Exception:
        gold = []
    order = B.order_records(q["question"], plans[r["qid"]], rs)
    pos = {i: n for n, i in enumerate(order)}
    shown = ctx[r["qid"]].split("\n\nRaw dialogue matches:")[0].count("\n- [") + 1
    best = min((pos.get(i, 10 ** 6) for i in gold), default=None)
    cls = ("notfound" if best is None else "shown" if best < shown else "near" if best < 3 * shown else "far")
    return dict(r, gold=gold, best=best, shown=shown, cls=cls, n_recs=len(rs))


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(one, items))
json.dump(rows, open(os.path.join(OUT, "diag_rank.json"), "w"), indent=1)
print("by class:", collections.Counter(r["cls"] for r in rows))
for ab in sorted({r["ability"] for r in rows}):
    rr = [r for r in rows if r["ability"] == ab]
    print(f"{ab:26s}", collections.Counter(r["cls"] for r in rr), "best ranks", sorted(r["best"] for r in rr if r["best"] is not None))
print("median shown", sorted(r["shown"] for r in rows)[len(rows) // 2])
