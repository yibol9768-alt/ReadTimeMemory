"""Check the reviewer's open points on W1 (10-31): A0O baseline, LongMemEval by type, conversation-level bootstrap on BEAM."""
import collections, json, random, statistics as st
hard = set(open("hard_ids.txt").read().split()); rng = random.Random(0)
def load(t):
    o = {}
    for r in map(json.loads, open(f"ctx_{t}.jsonl")):
        if r["gold_total"]: o[(r["qid"], r["budget"])] = r["gold_in"] / r["gold_total"]
    return o
def ci(d):
    bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000)); return f"{100 * st.mean(d):+.1f} [{100 * bs[100]:+.1f},{100 * bs[3899]:+.1f}] n={len(d)}"
def clus(pairs):
    by = collections.defaultdict(list)
    for c, d in pairs: by[c].append(d)
    cs = list(by); b = []
    for _ in range(4000):
        v = [x for c in rng.choices(cs, k=len(cs)) for x in by[c]]; b.append(st.mean(v))
    b.sort(); return f"{100 * st.mean(d for _, d in pairs):+.1f} [{100 * b[100]:+.1f},{100 * b[3899]:+.1f}] convs={len(cs)}"
W, A, O = load("W1_H"), load("A0_H"), load("A0O_H")
for lbl, f in (("hard 16k", lambda k: k[1] == 16000 and k[0] in hard), ("hard 4k", lambda k: k[1] == 4000 and k[0] in hard), ("other 4k", lambda k: k[1] == 4000 and k[0] not in hard), ("other 16k", lambda k: k[1] == 16000 and k[0] not in hard)):
    for n, B in (("A0", A), ("A0O", O)):
        ks = [k for k in W if k in B and f(k)]
        print("BEAM", lbl, "vs", n, "byq", ci([W[k] - B[k] for k in ks]), "| byconv", clus([(k[0].split("_")[0], W[k] - B[k]) for k in ks]))
mix = lambda V: {k[0]: V[(k[0], 16000 if k[0] in hard else 4000)] for k in V if (k[0], 16000) in V and (k[0], 4000) in V}
m, a, o = mix(W), mix(A), mix(O)
for n, B in (("A0", a), ("A0O", o)): print("BEAM mixed vs", n, "byconv", clus([(q.split("_")[0], m[q] - B[q]) for q in m if q in B]))
Q = {q["question_id"]: q["question_type"] for q in json.load(open("q_lmes150.json"))}
W, A, O, X = load("W1_lmes150"), load("A0_lmes150"), load("A0O_lmes150"), load("T2X_lmes150")
for n, B in (("A0", A), ("A0O", O), ("T2X", X)): print("LME 2k vs", n, "all", ci([W[k] - B[k] for k in W if k[1] == 2000 and k in B]))
for t in sorted(set(Q.values())):
    print(" ", t, " | ".join(f"{n} {ci([W[k] - B[k] for k in W if k[1] == 2000 and k in B and Q[k[0]] == t])}" for n, B in (("A0", A), ("A0O", O))))
