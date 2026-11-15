"""B0 summary (10-31): paired comparison of inference runs on hard and other questions, plus wrong-prefix harm vs untrained.
usage: python3 bsum.py"""
import json, random, statistics as st
hard = set(open("hard_ids.txt").read().split()); rng = random.Random(0)
def load(t):
    o = {}
    for r in map(json.loads, open(f"ctx_{t}.jsonl")):
        if r["gold_total"]: o[(r["qid"], r["budget"])] = r["gold_in"] / r["gold_total"]
    return o
def cmp(x, y, b, H):
    k = [q for q in x if q in y and q[1] == b and (q[0] in hard) == H]
    d = [x[q] - y[q] for q in k]; bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
    return f"{100 * st.mean(x[q] for q in k):.1f} vs {100 * st.mean(y[q] for q in k):.1f} diff {100 * st.mean(d):+.1f} [{100 * bs[34]:+.1f},{100 * bs[3965]:+.1f}] n={len(k)}"
T1 = load("T1_H"); A0 = load("A0_H")
for n in ("SA", "SB", "SC"):
    c = load(n + "C_H"); p = load(n + "P_H")
    for b in (4000, 16000):
        for H in (True, False):
            lbl = f"{n} {'hard' if H else 'other'} {b // 1000}k"
            print(lbl, "cond-T1:", cmp(c, T1, b, H), "| cond-plain:", cmp(c, p, b, H), "| cond-A0:", cmp(c, A0, b, H))
base = {r["qid"]: r for r in json.load(open("cond_a0b.json"))}
for n in ("sa", "sb", "sc"):
    for N in (5, 18):
        r = [x for x in json.load(open(f"cond_{n}b.json")) if x["qid"] in base and f"wrong{N}" in x and f"wrong{N}" in base[x["qid"]]]
        d = [(x[f"wrong{N}"] - x[f"wrongp{N}"]) - (base[x["qid"]][f"wrong{N}"] - base[x["qid"]][f"wrongp{N}"]) for x in r]
        bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
        print(f"{n} wrong-prefix harm vs untrained top{N}: {100 * st.mean(d):+.1f} [{100 * bs[34]:+.1f},{100 * bs[3965]:+.1f}] n={len(r)}")
