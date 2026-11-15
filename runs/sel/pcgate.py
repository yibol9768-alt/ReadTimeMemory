"""D3 validity gates (spec v2/v3). usage:
  python3 pcgate.py loc BEFORE.json AFTER.json      -> PC_LOC PASS/FAIL: a cell with carrier group gains >= 10 points of gap and becomes the largest-gap cell
  python3 pcgate.py train QFILE BUDGETS W20TAG ARMTAG -> PC_TRAIN PASS/FAIL: recall on carrier-group-only questions, arm - W20 >= 3 points"""
import json, statistics as st, sys
if sys.argv[1] == "loc":
    b, a = (json.load(open(x)) for x in sys.argv[2:4]); gb = {tuple(r["cell"]): r["gap"] for r in b["rows"]}
    ga = {tuple(r["cell"]): r for r in a["rows"] if r["eligible"]}
    top = tuple(max(ga.values(), key=lambda r: (r["gap"], r["n"]))["cell"]) if ga else None
    best = top if top and top[1] == "group" else None   # one eligible group cell must both be the largest gap and have gained >= 10 points
    ok = best is not None and ga[best]["gap"] - gb.get(best, 0) >= .10
    print("PC_LOC", "PASS" if ok else "FAIL", "| best group cell", best, "gain", round(100 * (ga[best]["gap"] - gb.get(best, 0)), 1) if best else None, "| top cell", top); sys.exit(0 if ok else 3)
qf, Bs, w, x = sys.argv[2], [int(v) for v in sys.argv[3].split(",")], sys.argv[4], sys.argv[5]
Q = [q for q in json.load(open(qf)) if q.get("carriers") == ["group"]]
def rec(tag):
    o = {(r["qid"], r["budget"]): r["gold_in"] / r["gold_total"] for r in map(json.loads, open(f"ctx_{tag}.jsonl")) if r["gold_total"]}
    return st.mean(st.mean(o[(q["question_id"], b)] for b in Bs) for q in Q)
d = rec(x) - rec(w); ok = len(Q) >= 5 and d >= .03
print("PC_TRAIN", "PASS" if ok else "FAIL", f"| group-only questions {len(Q)} | W20 {100*rec(w):.1f} arm {100*rec(x):.1f} diff {100*d:+.1f}"); sys.exit(0 if ok else 3)
