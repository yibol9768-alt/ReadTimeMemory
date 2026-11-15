"""Signal E (11-01, D3_SPEC v2). Per (question type x carrier) cell and budget: ranking gap = (gold the budget oracle fits - gold the selector got) / all gold,
oracle = gold rounds inside the fused top-600 pool, shortest first (fits the most gold). Out-of-pool share reported apart, never part of the signal.
usage: python3 ecell.py QFILE POOLDIR BUDGETS TAG [--json OUT] [--min-n 20]
       TAG = ctx tag; TAG:demote = that selector's own order (its selection order, then the fused pool) with the group-carrier rounds it selected moved after rank 50 (positive control);
       rr8 = 8B teacher order from the pool. Every question must have every budget, else the run is invalid."""
import argparse, collections, json, statistics as st, sys
sys.path.insert(0, "."); from k1 import rounds_of, pack, ntok
ap = argparse.ArgumentParser(); ap.add_argument("qfile"); ap.add_argument("pooldir"); ap.add_argument("budgets"); ap.add_argument("tag")
ap.add_argument("--json", default=""); ap.add_argument("--min-n", type=int, default=20); a = ap.parse_args()
Bs = [int(x) for x in a.budgets.split(",")]; Q = json.load(open(a.qfile))
base, demote = a.tag.split(":")[0], a.tag.endswith(":demote")
C = None if base == "rr8" else {(r["qid"], r["budget"]): r for r in map(json.loads, open(f"ctx_{base}.jsonl"))}
def fill(R, order, B):
    """exact: keeps adding while anything fits (no early stop), so the shortest-first oracle fits the most gold"""
    keep, used = [], 0
    for i in order:
        n_ = ntok(R[i][2])
        if used + n_ <= B: keep.append(i); used += n_
    return set(keep)
def cell(q):
    c = q.get("carriers", ["?"]); return (q["question_type"], c[0] if len(c) == 1 else "mixed")
def group_rounds(q):
    out, i = [], 0
    for s in q["haystack_sessions"]:
        j = 0
        while j < len(s):
            n = 2 if j + 1 < len(s) and s[j]["role"] == "user" and s[j + 1]["role"] == "assistant" else 1
            if s[j].get("carrier") == "group": out.append(i)
            i += 1; j += n
    return set(out)
G = collections.defaultdict(lambda: collections.defaultdict(list)); OUT = collections.defaultdict(list); n = collections.Counter()
for q in Q:
    R = rounds_of(q); g = {i for i, r in enumerate(R) if r[3]}
    if not g: continue
    P = json.load(open(f"{a.pooldir}/{q['question_id']}.json")); F = P["fused"][:600]; c = cell(q)
    orc_order = sorted([i for i in F if i in g], key=lambda i: ntok(R[i][2]))
    if C is not None and any((q["question_id"], B) not in C for B in Bs): raise SystemExit(f"INVALID: {q['question_id']} lacks a budget in ctx_{base}")
    gr = group_rounds(q)
    for B in Bs:
        if C is None:
            rr = {int(k): v for k, v in P["rr8"].items()}; order = sorted(rr, key=lambda i: -rr[i]) + [i for i in F if str(i) not in P["rr8"]]; got = fill(R, order, B)
        elif demote:
            sel = C[(q["question_id"], B)].get("sel") or sorted(C[(q["question_id"], B)]["keep"]); own = sel + [i for i in F if i not in set(sel)]
            kb = C[(q["question_id"], B)]["keep"]; moved = sorted(i for i in kb if i in gr)   # every group round selected at this budget
            rest = [i for i in own if i not in set(moved)]; got = fill(R, rest[:50] + moved + rest[50:], B)
        else: got = set(C[(q["question_id"], B)]["keep"])
        o = fill(R, orc_order, B)
        G[c][B].append((len(o & g) - len(got & g)) / len(g))
    n[c] += 1; OUT[c].append(len(g - set(F)) / len(g))
rows = []
for c in sorted(G, key=str):
    gap = st.mean(st.mean(G[c][B]) for B in Bs)
    rows.append({"cell": list(c), "n": n[c], "gap": gap, "by_budget": {B: st.mean(G[c][B]) for B in Bs}, "out_of_pool": st.mean(OUT[c]), "eligible": n[c] >= a.min_n})
for r in sorted(rows, key=lambda r: (-r["eligible"], -r["gap"], -r["n"])):
    print(f"{str(tuple(r['cell'])):32s} n={r['n']:4d} gap {100*r['gap']:6.1f}  " + " ".join(f"{B}:{100*v:5.1f}" for B, v in r["by_budget"].items()) + f"  out-of-pool {100*r['out_of_pool']:5.1f}" + ("" if r["eligible"] else "  (too few)"))
el = [r for r in rows if r["eligible"]]
tgt = max(el, key=lambda r: (r["gap"], r["n"])) if el else None
print("TARGET", tuple(tgt["cell"]) if tgt else None)
if a.json: json.dump({"rows": rows, "target": tgt["cell"] if tgt else None}, open(a.json, "w"))
