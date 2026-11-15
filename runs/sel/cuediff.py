"""A4 difference of two paired changes (11-01): (B_edit - B_orig) - (A_edit - A_orig), all on the same kept questions, clustered as cuesum.py.
Used for W20 vs A0 dependency and for treatment vs matched control (then A = control run, B = treatment run, both models same).
usage: python3 cuediff.py A_ORIG A_EDIT B_ORIG B_EDIT EDITED_QFILE [EDITED_QFILE_B] BUDGET"""
import json, random, re, statistics as st, sys
sys.path.insert(0, "."); from cuesum import load
a = sys.argv[1:]; b = int(a[-1]); qf = a[4:-1]
want = set.intersection(*[{q.get("orig_question_id", q["question_id"]) for q in json.load(open(f))} for f in qf])
L = [load(t, b) for t in a[:4]]; ks = sorted(want.intersection(*map(set, L)))
if len(ks) != len(want): raise SystemExit(f"INVALID: {len(ks)} of {len(want)} questions present in all four runs")
d = {k: (L[3][k] - L[2][k]) - (L[1][k] - L[0][k]) for k in ks}
conv = lambda k: k.split("_")[0] if re.match(r"[hc]\d+_", k) else k; by = {}
for k in ks: by.setdefault(conv(k), []).append(d[k])
r = random.Random(0); cs = list(by); B = sorted(st.mean(v for c in r.choices(cs, k=len(cs)) for v in by[c]) for _ in range(8000))
print(f"diff of changes {100*st.mean(d.values()):+.1f} [{100*B[200]:+.1f},{100*B[7799]:+.1f}] n={len(ks)} clusters={len(cs)}")
