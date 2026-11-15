"""Paired bootstrap (by question, and by world) for H4j area differences between tags. usage: python3 d4/d4ci.py A_TAG B_TAG ..."""
import json, random, statistics as st, sys, collections
BUD = [1000, 2000, 4000, 8000, 16000]; Q = {q["question_id"]: q for q in json.load(open("env_h4.json"))}
grp = lambda q: any("group" in c for c in q.get("carriers", []))
def per_q(tag):
    E = collections.defaultdict(dict)
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if r["gold_total"]: E[r["qid"]][r["budget"]] = r["gold_in"] / r["gold_total"]
    return {k: 100 * st.mean(v[b] for b in BUD) for k, v in E.items() if int(k.split("_")[0][1:]) % 2 == 1}
args = sys.argv[1:]
for i in range(0, len(args), 2):
    A, B = per_q(args[i]), per_q(args[i + 1])
    for name, sel in (("all", lambda q: True), ("group", grp)):
        qs = [k for k in A if k in B and sel(Q[k])]; d = [A[k] - B[k] for k in qs]
        r = random.Random(0); bs = sorted(st.mean(r.choices(d, k=len(d))) for _ in range(4000))
        W = collections.defaultdict(list); [W[k.split("_")[0]].append(A[k] - B[k]) for k in qs]; ws = list(W)
        bw = sorted(st.mean(x for w in r.choices(ws, k=len(ws)) for x in W[w]) for _ in range(4000))
        print(f"{args[i]} - {args[i+1]} {name:5s} n={len(qs)} diff {st.mean(d):+.2f}  q-CI [{bs[100]:+.2f},{bs[3899]:+.2f}]  world-CI [{bw[100]:+.2f},{bw[3899]:+.2f}]")
