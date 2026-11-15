"""Transfer-test summary: EO + summarization round recall (primary 4k, secondary 16k) vs the better untrained 0.6B baseline,
paired bootstrap over questions; other types pooled for the no-regression check. usage: python3 txsum.py T1_H,T2_H"""
import json, random, sys
ids = set(open("ex_ids.txt").read().split())
qt = {q["question_id"]: q["question_type"] for q in json.load(open("beam500k_g.json")) if q["question_id"] in ids}
def load(tag):
    d = {}
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if r["qid"] in qt and r["gold_total"] and r["budget"] in (4000, 16000): d[(r["qid"], r["budget"])] = r["gold_in"] / r["gold_total"]
    return d
HARD = ("event_ordering", "summarization")
def mean(d, B, hard): v = [x for (q, b), x in d.items() if b == B and (qt[q] in HARD) == hard]; return 100 * sum(v) / max(1, len(v)), len(v)
base = {t: load(t) for t in ("A0_H", "A0O_H", "RR8_H")}
for t, d in base.items(): print(t, {B: {"hard": round(mean(d, B, True)[0], 1), "other": round(mean(d, B, False)[0], 1), "n": mean(d, B, True)[1]} for B in (4000, 16000)})
for t in sys.argv[1].split(","):
    d = load(t)
    for B in (4000, 16000):
        bn = max(("A0_H", "A0O_H"), key=lambda k: mean(base[k], B, True)[0]); bs = base[bn]
        for hard in (True, False):
            qs = [q for (q, b) in d if b == B and (qt[q] in HARD) == hard and (q, b) in bs]
            diffs = [d[(q, B)] - bs[(q, B)] for q in qs]; rng = random.Random(0)
            bt = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
            print(f"{t} B={B} {'EO+sum' if hard else 'other '} {100 * sum(diffs) / len(diffs):+.1f} [{100 * bt[50]:+.1f}, {100 * bt[1950]:+.1f}] vs {bn} n={len(qs)}")
