"""Date-mask summary (10-31): paired recall of T1 and untrained 0.6B with candidate dates hidden. 95% intervals."""
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
    return f"{100 * st.mean(x[q] for q in k):.1f} vs {100 * st.mean(y[q] for q in k):.1f} diff {100 * st.mean(d):+.1f} [{100 * bs[100]:+.1f},{100 * bs[3899]:+.1f}] n={len(k)}"
T, TM, A, AM = load("T1_H"), load("T1M_H"), load("A0_H"), load("A0M_H")
for b in (4000, 16000):
    for H in (True, False):
        print(f"{'hard' if H else 'other'} {b // 1000}k | T1mask-T1: {cmp(TM, T, b, H)} | A0mask-A0: {cmp(AM, A, b, H)} | T1mask-A0mask: {cmp(TM, AM, b, H)}")
