"""EverMem-style short-positive pilot data (11-01 plan v9): groups whose positive body is under 800 characters, from the fixed-mix pools of ws6x and ws7x. usage: python3 subshort.py OUT N"""
import json, random, re, sys
out, N = sys.argv[1], int(sys.argv[2]); L = []
for f in ("all_fix.jsonl", "v9_all_fix.jsonl"):
    for x in open(f):
        r = json.loads(x); d = r["docs"][r["label"]]; b = re.sub(r"^<Document>:\s*", "", d).split("<|im_end|>")[0]
        if len(b) < 800: L.append(x)
print("short groups", len(L))
if len(L) < N: raise SystemExit(f"NOT EXECUTABLE: {len(L)} of {N}")
r = random.Random(0); s = r.sample(L, N); open(out, "w").writelines(s); print("wrote", N)
