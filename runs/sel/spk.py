"""Speaker-match probe (11-02, CPU): EverMem questions often say "I (Name) ...". For each model, the share of its top-K ranked
rounds spoken by that Name, compared with the share among gold rounds and among the whole 600 pool.
usage: python3 spk.py [K=20]"""
import json, re, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of
K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
Q = {q["question_id"]: q for q in json.load(open("q_ev150.json"))}
MODELS = ["E0", "A0", "A0O", "RR8", "MR4", "T1", "GTE"]
C = {m: {r["qid"]: r["rank"] for r in map(json.loads, open(f"ctx_P1{m}_ev150.jsonl")) if r["budget"] == 4000} for m in MODELS}
def spk(t):
    m = re.match(r"User: \[[^\]]*\] ([^:]{2,40}):", t); return m.group(1).strip() if m else None
rows = {m: [] for m in MODELS}; gold = []; pool = []; n = 0; types = {}
for qid, q in Q.items():
    m = re.search(r"\bI \(([^)]{2,40})\)", q["question"])
    if not m: continue
    me = m.group(1).strip(); R = rounds_of(q); S = [spk(r[2]) for r in R]; n += 1
    g = [i for i, r in enumerate(R) if r[3]]
    if g: gold.append(sum(S[i] == me for i in g) / len(g))
    pool.append(sum(S[i] == me for i in C["E0"][qid]) / len(C["E0"][qid]))
    types[q["question_type"]] = types.get(q["question_type"], 0) + 1
    for mo in MODELS: rows[mo].append(sum(S[i] == me for i in C[mo][qid][:K]) / K)
print(f"questions with 'I (Name)': {n} of {len(Q)}; types {types}")
print(f"share of rounds spoken by the asker: gold {100*st.mean(gold):.1f}%, whole 600 pool {100*st.mean(pool):.1f}%")
for mo in MODELS: print(f"  top-{K} {mo:4s} {100*st.mean(rows[mo]):5.1f}%")
