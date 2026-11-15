"""E groups with the same question-type counts as the fixed set (spec v8 item 6). usage: python3 subq.py ALL_E.jsonl FIXED_SUB.jsonl OUT.jsonl [seed]"""
import collections, json, random, sys
L = open(sys.argv[1]).readlines(); Q = collections.Counter(json.loads(x)["type"] for x in open(sys.argv[2])); s = int(sys.argv[4]) if len(sys.argv) > 4 else 0
by = collections.defaultdict(list)
for x in L: by[json.loads(x)["type"]].append(x)
r = random.Random(s); out = []
for t, n in sorted(Q.items()):
    if len(by[t]) < n: raise SystemExit(f"NOT EXECUTABLE: type {t} has {len(by[t])}, need {n}")
    out += r.sample(by[t], n)
r.shuffle(out); open(sys.argv[3], "w").writelines(out); print("wrote", len(out), dict(Q))
