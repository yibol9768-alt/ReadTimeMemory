"""D3 v9 control (spec v8 item 5). usage: python3 pcgate3.py pre W20TAG  |  python3 pcgate3.py post W20TAG TAG0,TAG1,TAG2
Questions: aggregation x mixed on ws7ce. pre: n >= 128, >= 10 per world, W20 headroom to the shared oracle >= 10. post: seeds averaged per question, gain >= 3, 95% lower bound > 0 by resampling whole worlds."""
import collections, json, random, statistics as st, sys
sys.path.insert(0, "."); from k1 import rounds_of, ntok
Bs = [1000, 2000, 4000]; Q = [q for q in json.load(open("env_ws7ce.json")) if q["question_type"] == "aggregation" and len(q.get("carriers", [])) > 1]; ids = [q["question_id"] for q in Q]
def rec(t):
    o = {(r["qid"], r["budget"]): r["gold_in"] / r["gold_total"] for r in map(json.loads, open(f"ctx_{t}.jsonl")) if r["gold_total"]}
    miss = [(q, b) for q in ids for b in Bs if (q, b) not in o]
    if miss: print("INVALID missing", t, len(miss)); sys.exit(2)
    return {q: st.mean(o[(q, b)] for b in Bs) for q in ids}
w = rec(sys.argv[2]); byw = collections.Counter(q.split("_")[0] for q in ids)
if sys.argv[1] == "pre":
    def fill(R, O, B):
        k, u = [], 0
        for i in O:
            n = ntok(R[i][2])
            if u + n <= B: k.append(i); u += n
        return k
    o = []
    for q in Q:
        R = rounds_of(q); g = [i for i, r in enumerate(R) if r[3]]; F = set(json.load(open(f"pool_ws7ce/{q['question_id']}.json"))["fused"][:600]); O = sorted([i for i in g if i in F], key=lambda i: (ntok(R[i][2]), i))
        o.append(st.mean(len(fill(R, O, b)) / len(g) for b in Bs))
    head = 100 * (st.mean(o) - st.mean(w.values())); ok = len(ids) >= 128 and min(byw.values()) >= 10 and head >= 10
    print("PRE", "PASS" if ok else "FAIL", "n", len(ids), "per world", dict(byw), "W20", round(100 * st.mean(w.values()), 1), "headroom", round(head, 1)); sys.exit(0 if ok else 3)
A = [rec(t) for t in sys.argv[3].split(",")]; d = {q: st.mean(a[q] for a in A) - w[q] for q in ids}; D = st.mean(d.values())
wd = collections.defaultdict(list)
for q, v in d.items(): wd[q.split("_")[0]].append(v)
W = sorted(wd); r = random.Random(0); bs = []
for _ in range(10000):
    s = r.choices(W, k=len(W)); xs = [v for x in s for v in wd[x]]; bs.append(st.mean(xs))
bs.sort(); lo, hi = bs[250], bs[9749]; ok = D >= .03 and lo > 0
print("CONTROL", "PASS" if ok else "FAIL", f"n {len(ids)} gain {100*D:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]", "per seed", [round(100 * st.mean(a[q] - w[q] for q in ids), 1) for a in A], "per world", {k: round(100 * st.mean(v), 1) for k, v in sorted(wd.items())}); sys.exit(0 if ok else 3)
