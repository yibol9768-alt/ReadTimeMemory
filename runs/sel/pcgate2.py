"""Repaired D3 training control (spec v7, frozen before training). usage: python3 pcgate2.py QFILE BUDGETS W20TAG ARMTAG
Questions: aggregation x mixed on the control worlds. PASS needs: mean per-question gain >= 3 points (budgets averaged), paired bootstrap 95% lower bound > 0,
n >= 30 in total and n >= 10 in each world, no missing record. exit 0 PASS, 3 FAIL, 2 INVALID."""
import collections, json, random, statistics as st, sys
qf, Bs, w, x = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3], sys.argv[4]
Q = [q for q in json.load(open(qf)) if q["question_type"] == "aggregation" and len(q.get("carriers", [])) > 1]
def rec(tag):
    return {(r["qid"], r["budget"]): r["gold_in"] / r["gold_total"] for r in map(json.loads, open(f"ctx_{tag}.jsonl")) if r["gold_total"]}
W, A = rec(w), rec(x); ids = [q["question_id"] for q in Q]
miss = [(q, b) for q in ids for b in Bs for o in (W, A) if (q, b) not in o]
if miss: print("PC_TRAIN2 INVALID missing", len(miss), miss[:2]); sys.exit(2)
wq = {q: st.mean(W[(q, b)] for b in Bs) for q in ids}; aq = {q: st.mean(A[(q, b)] for b in Bs) for q in ids}
d = [aq[q] - wq[q] for q in ids]; D = st.mean(d) if d else float("nan")
r = random.Random(0); bs = sorted(st.mean(r.choices(d, k=len(d))) for _ in range(10000)); lo, hi = bs[250], bs[9749]
byw = collections.Counter(q.split("_")[0] for q in ids)
perb = {b: round(100 * st.mean(A[(q, b)] - W[(q, b)] for q in ids), 1) for b in Bs}
ok = D >= .03 and lo > 0 and len(ids) >= 30 and all(n >= 10 for n in byw.values())
print("PC_TRAIN2", "PASS" if ok else "FAIL", f"| n {len(ids)} per world {dict(byw)} | W20 {100*st.mean(wq.values()):.1f} arm {100*st.mean(aq.values()):.1f} diff {100*D:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] | by budget {perb}")
sys.exit(0 if ok else 3)
