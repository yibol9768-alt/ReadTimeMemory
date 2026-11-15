"""S1b summary (10-31): world-trained model vs untrained, paired, 95% bootstrap."""
import json, random, statistics as st, sys
rng = random.Random(0); hard = set(open("hard_ids.txt").read().split()); tag = sys.argv[1] if len(sys.argv) > 1 else "W1"
def load(t):
    o = {}
    for r in map(json.loads, open(f"ctx_{t}.jsonl")):
        if r["gold_total"]: o[(r["qid"], r["budget"])] = r["gold_in"] / r["gold_total"]
    return o
def cmp(x, y, f):
    k = [q for q in x if q in y and f(q)]
    if not k: return "none"
    d = [x[q] - y[q] for q in k]; bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
    return f"{100 * st.mean(x[q] for q in k):.1f} vs {100 * st.mean(y[q] for q in k):.1f} diff {100 * st.mean(d):+.1f} [{100 * bs[100]:+.1f},{100 * bs[3899]:+.1f}] n={len(k)}"
W, A = load(f"{tag}_w14"), load("A0_w14"); Wb, Ab, Bb = load(f"{tag}_H"), load("A0_H"), load("T2X_H"); Wl, Al = load(f"{tag}_lmes150"), load("A0_lmes150")
multi = lambda q: "_aggregation_" in q[0] or "_ordering_" in q[0]
for b in (4000, 16000):
    print(f"held-out world {b // 1000}k multi:", cmp(W, A, lambda q: q[1] == b and multi(q)), "| single:", cmp(W, A, lambda q: q[1] == b and not multi(q)))
    print(f"BEAM {b // 1000}k hard vs A0:", cmp(Wb, Ab, lambda q: q[1] == b and q[0] in hard), "| other:", cmp(Wb, Ab, lambda q: q[1] == b and q[0] not in hard))
    print(f"BEAM {b // 1000}k hard vs T2X:", cmp(Wb, Bb, lambda q: q[1] == b and q[0] in hard), "| other:", cmp(Wb, Bb, lambda q: q[1] == b and q[0] not in hard))
Wm, Am = load(f"{tag}M_w14"), load("A0M_w14")
for b in (4000, 16000): print(f"held-out world, dates hidden {b // 1000}k multi:", cmp(Wm, Am, lambda q: q[1] == b and multi(q)), "| single:", cmp(Wm, Am, lambda q: q[1] == b and not multi(q)))
for b in (1000, 2000, 4000): print(f"LME {b // 1000}k vs A0:", cmp(Wl, Al, lambda q: q[1] == b))
