"""K-S1 analysis: accuracy and cost per (history, context budget, thinking budget), and the pre-registered gates.
cost per answer = prefill tokens of the first call + all generated tokens (prefix caching makes the 2nd call's prefill ~free).
usage: python3 runs/s1/analyze.py runs/s1/out/S_full.judged_Qwen3.8-27B-FP8.jsonl runs/s1/out/M_full.judged_Qwen3.8-27B-FP8.jsonl
"""
import collections, json, random, sys

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
cell = collections.defaultdict(dict)
for r in rows:
    pre = r["in_tok"] if r["think"] == 0 else r["in_tok"] // 2
    cell[(r["hist"], r["budget"], r["think"])][r["qid"]] = (int(r["label"]), pre + r["out_tok"], r["think_tok"], r["think_cut"])

hists = sorted({k[0] for k in cell}); B = sorted({k[1] for k in cell}); T = sorted({k[2] for k in cell})
for h in hists:
    print(f"\n== {h}: accuracy % (mean cost k-tokens) [mean thinking tokens, % hit cap]")
    print("budget \\ think " + "".join(f"{t:>26d}" for t in T))
    for b in B:
        line = f"{b:>14d} "
        for t in T:
            c = cell.get((h, b, t), {})
            if not c:
                line += f"{'-':>26s}"; continue
            acc = 100 * sum(v[0] for v in c.values()) / len(c); cost = sum(v[1] for v in c.values()) / len(c) / 1000
            th = sum(v[2] for v in c.values()) / len(c); cut = 100 * sum(v[3] for v in c.values()) / len(c)
            line += f"{acc:6.1f} ({cost:6.1f}k) [{th:5.0f},{cut:3.0f}%]"
        print(line)


def boot(a, b, n=4000):
    qs = sorted(set(a) & set(b)); rnd = random.Random(0); d = []
    for _ in range(n):
        s = [rnd.choice(qs) for _ in qs]
        d.append(100 * sum(a[q][0] - b[q][0] for q in s) / len(s))
    d.sort(); return 100 * sum(a[q][0] - b[q][0] for q in qs) / len(qs), d[int(n * .025)], d[int(n * .975)], len(qs)


print("\n== Gate 1 (M): small memory (<=16k) + low thinking (<=2k) vs max context + max thinking")
if "M" in hists:
    big = cell[("M", max(B), max(T))]
    best_small = max(((b, t) for b in B if b <= 16000 for t in T if t <= 2000 and ("M", b, t) in cell),
                     key=lambda k: sum(v[0] for v in cell[("M",) + k].values()))
    sm = cell[("M",) + best_small]
    d, lo, hi, n = boot(sm, big)
    cs = sum(v[1] for v in sm.values()) / len(sm); cb = sum(v[1] for v in big.values()) / len(big)
    print(f"small {best_small}: acc diff small-big = {d:+.1f} [{lo:+.1f},{hi:+.1f}] n={n}; cost ratio big/small = {cb / cs:.1f}x")
    print("   PASS" if (lo > 0 and d >= 8) or cb / cs >= 5 else "   FAIL")
print("\n== Gate 2: does more thinking help more on M than on S? (max-think minus no-think, per budget)")
for b in B:
    out = []
    for h in hists:
        if (h, b, 0) in cell and (h, b, max(T)) in cell:
            d, lo, hi, n = boot(cell[(h, b, max(T))], cell[(h, b, 0)])
            out.append(f"{h}: {d:+.1f} [{lo:+.1f},{hi:+.1f}]")
    print(f"budget {b}: " + "  ".join(out))
