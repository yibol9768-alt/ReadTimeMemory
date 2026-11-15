"""For H4 questions of a type where P0 < C: gold rank under C/P0, and the rounds P0 puts above the gold that C puts below."""
import json, sys, collections, statistics as st
sys.path.insert(0, "."); from k1 import rounds_of
TYPE = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
Q = {q["question_id"]: q for q in json.load(open("env_h4.json")) if q["question_type"] == TYPE}
def load(tag):
    o = {}
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if r["qid"] in Q and r["budget"] == 16000: o[r["qid"]] = r["rank"]
    return o
C, P = load("D4Cs600_h4"), load("D4P0s600_h4")
rows = []
for k, q in Q.items():
    R = rounds_of(q); gold = [i for i, r in enumerate(R) if r[3]]
    rc = min(C[k].index(g) for g in gold if g in C[k]) if any(g in C[k] for g in gold) else 999
    rp = min(P[k].index(g) for g in gold if g in P[k]) if any(g in P[k] for g in gold) else 999
    rows.append((rp - rc, k, rc, rp, gold, R))
print(TYPE, "n", len(rows), "first-gold rank median C", st.median(r[2] for r in rows), "P0", st.median(r[3] for r in rows), "| P0 worse", sum(r[0] > 0 for r in rows), "better", sum(r[0] < 0 for r in rows))
for d, k, rc, rp, gold, R in sorted(rows, reverse=True)[:N]:
    q = Q[k]; print(f"\n### {k}  first gold rank C {rc} -> P0 {rp}\nQ: {q['question'][:250]}\nA: {str(q['answer'])[:250]}\nGOLD: {R[gold[0]][2][:300]}")
    above = [i for i in P[k][:rp] if i not in gold][:3]
    for i in above: print(f"  P0 ranks above gold (C rank {C[k].index(i) if i in C[k] else '-'}): {R[i][2][:260]}")
