"""recall table across models for one benchmark: python3 mbsum.py NAME [TAGS]"""
import json, sys, statistics as st, collections
N = sys.argv[1]; tags = (sys.argv[2] if len(sys.argv) > 2 else "RR8,A0,MR4,T2X,T1").split(",")
Q = {q["question_id"]: q["question_type"] for q in json.load(open(f"q_{N}.json"))}
print(f"== {N}: {len(Q)} questions; recall of gold rounds (mean over questions with gold)")
rows = {}
for t in tags:
    try: f = open(f"ctx_{t}_{N}.jsonl")
    except FileNotFoundError: continue
    d = collections.defaultdict(list)
    for r in map(json.loads, f):
        if int(r["gold_total"]): d[int(r["budget"])].append(int(r["gold_in"]) / int(r["gold_total"]))
    rows[t] = {b: 100 * st.mean(v) for b, v in d.items()}
    print(f"  {t:5s} " + "  ".join(f"{b // 1000}k {v:5.1f} (n={len(d[b])})" for b, v in sorted(rows[t].items())))
