"""WP1 error decomposition on the ds41 arm, cache only (any cache miss raises).
For every wrong answer: was the evidence written into memory at all (write miss), written but not in the
context (retrieval miss), or fully in the context (reasoning / label). Evidence turns are LoCoMo dia_ids;
a turn counts as visible if a shown record cites it or the raw turn text itself is shown."""
import json, os, sys, collections
os.environ["V3_MODEL"] = "deepseek-v4.1-flash"
os.environ["V3_NO_GPT"] = "1"
sys.argv = ["x"]


def _no_net(*a, **k):
    raise RuntimeError("cache miss: this analysis makes no LLM calls")


from mem import llm, locomo_exp as L, lychee as Y
llm.urllib.request.urlopen = _no_net
llm.h20_window = lambda: None
d = json.load(open("data/locomo10.json"))
qs = {q["qid"]: q for q in L.questions(d, 20)}
convs = {q["conv"] for q in qs.values()}
sss = {ci: L.sessions(d[ci]) for ci in convs}
turns = {ci: {t["id"]: t for s in sss[ci] for t in s["turns"]} for ci in convs}
recs = {"rec_sess": {ci: json.load(open(f"runs/locomo/ds41/mem_{ci}.json"))["rec_sess"] for ci in convs},
        "lychee": {ci: json.load(open(f"runs/lychee/ds41/mem_{ci}.json"))["records"] for ci in convs}}
recs["hyb_mq"] = recs["rec_sess"]
mems = {ci: json.load(open(f"runs/locomo/ds41/mem_{ci}.json")) for ci in convs}


def evidence(q):
    return {e.strip() for e in q.get("evidence", []) for e in e.split(";") if e.strip()}


def ctx_of(cond, q):
    ci = q["conv"]
    if cond == "lychee":
        return Y.context(q["question"], Y.plan(q["question"]), recs["lychee"][ci], sss[ci])
    return L.context(cond, q["question"], sss[ci], mems[ci])


def visible(ctx, cond, ci):
    v = {s for r in recs[cond][ci] if r["text"] in ctx for s in r["sources"]}
    return v | {i for i, t in turns[ci].items() if len(t["text"]) > 20 and t["text"] in ctx}


rows, seen = [], set()
for r in map(json.loads, open("runs/locomo/ds41/judged.jsonl")):
    if r["qid"] in qs and (r["qid"], r["cond"]) not in seen:
        seen.add((r["qid"], r["cond"])); rows.append(r)
tab = collections.defaultdict(collections.Counter)
out = open("runs/wp1_wrong.txt", "w")
for r in rows:
    q = qs[r["qid"]]; ci = q["conv"]; ev = evidence(q)
    written = {s for x in recs[r["cond"]][ci] for s in x["sources"]}
    if not ev:
        kind = "no-evidence-label"
    else:
        vis = visible(ctx_of(r["cond"], q), r["cond"], ci)
        kind = "in-ctx" if ev <= vis else "retrieval" if ev <= written else "write"
    tab[r["cond"]][(kind, r["correct"])] += 1
    tab[(r["cond"], r["cat"])][(kind, r["correct"])] += 1
    if not r["correct"]:
        out.write(f"[{r['cond']} cat{r['cat']} {kind}] Q: {q['question']}\n  gold: {r['gold']}\n  pred: {r['pred'][:300]}\n\n")

names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
for cond in ("rec_sess", "hyb_mq", "lychee"):
    c = tab[cond]
    print(f"\n{cond}: n={sum(c.values())}, wrong={sum(v for (k, ok), v in c.items() if not ok)}")
    for kind in ("write", "retrieval", "in-ctx", "no-evidence-label"):
        n = c[(kind, 0)] + c[(kind, 1)]
        if n:
            print(f"  evidence {kind:17s} {n:3d} questions, wrong {c[(kind, 0)]:2d}, correct {100 * c[(kind, 1)] / n:5.1f}%")
    print("  wrong by category: " + ", ".join(f"{names[k]} " + "/".join(f"{kind[:4]}{tab[(cond, k)][(kind, 0)]}" for kind in ("write", "retrieval", "in-ctx")) for k in (1, 2, 3, 4)))
