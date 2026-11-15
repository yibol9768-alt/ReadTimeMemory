"""No-GPU bridge check: selector recall of gold rounds by gold length, synthetic vs real, plus solo oracle per carrier on feedback worlds.
usage: python3 gbridge.py"""
import collections, json, statistics as st, sys
sys.path.insert(0, "."); from k1 import rounds_of, ntok
S = [("ws6f", "env_ws6f.json", "W20X_ws6f", 1000), ("ws6f", "env_ws6f.json", "W20X_ws6f", 2000), ("ws6f", "env_ws6f.json", "W20X_ws6f", 4000),
     ("lme", "q_lmes150.json", "W20X_lmes150", 2000), ("ep", "q_ep150.json", "W20_ep150", 1000), ("ev", "q_ev150.json", "W20_ev150", 4000)]
ED = [0, 100, 250, 500, 1000, 10**9]
def bk(n): return next(i for i in range(5) if ED[i] <= n < ED[i + 1])
for name, qf, tag, B in S:
    Q = json.load(open(qf)); C = {(r["qid"], r["budget"]): r for r in map(json.loads, open(f"ctx_{tag}.jsonl"))}
    h = collections.defaultdict(list); nf = collections.Counter(); skipped = 0
    for q in Q:
        k = (q["question_id"], B)
        if k not in C: skipped += 1; continue
        keep = set(C[k]["keep"]); R = rounds_of(q)
        for i, r in enumerate(R):
            if r[3]:
                n = ntok(r[2]); h[bk(n)].append(i in keep); nf[bk(n)] += n > B
    print(name, B, "skipped", skipped, " | ".join(f"{ED[i]}-{ED[i+1] if ED[i+1]<10**9 else 'inf'}: n={len(h[i])} rec={100*st.mean(h[i]):.0f} toolong={nf[i]}" for i in sorted(h)))
Q = json.load(open("env_ws6f.json")); C = {(r["qid"], r["budget"]): r for r in map(json.loads, open("ctx_W20X_ws6f.jsonl"))}
def cr(q):
    out, i = {}, 0
    for s in q["haystack_sessions"]:
        j = 0
        while j < len(s):
            n = 2 if j + 1 < len(s) and s[j]["role"] == "user" and s[j + 1]["role"] == "assistant" else 1
            c = s[j].get("carrier"); out[i] = c if c in ("group", "long") else "dialog"; i += 1; j += n
    return out
A = collections.defaultdict(lambda: collections.defaultdict(list))
for q in Q:
    if q["question_type"] != "aggregation": continue
    R = rounds_of(q); g = [i for i, r in enumerate(R) if r[3]]; c = cr(q)
    for k in ("dialog", "group", "long"):
        gc = [i for i in g if c[i] == k]
        if not gc: continue
        A[k]["ntok"].append(st.median(ntok(R[i][2]) for i in gc))
        for b in (1000, 2000, 4000):
            used = n = 0
            for t in sorted(ntok(R[i][2]) for i in gc):
                if used + t <= b: used += t; n += 1
            A[k]["solo"].append(n / len(gc)); A[k]["sel"].append(len(set(C[(q["question_id"], b)]["keep"]) & set(gc)) / len(gc)); A[k]["cost"].append(sum(ntok(R[i][2]) for i in gc))
for k, d in A.items(): print("aggregation", k, "median gold tok", round(st.mean(d["ntok"])), "total gold tok", round(st.mean(d["cost"])), "solo oracle", round(100 * st.mean(d["solo"]), 1), "selector", round(100 * st.mean(d["sel"]), 1), "solo gap", round(100 * (st.mean(d["solo"]) - st.mean(d["sel"])), 1))
