"""Failure-case pack for stage 1 (11-02, CPU only). For a dev benchmark, pick questions where E0 (CruxRank stage 1) loses
most to a reference model on evidence recall at the answer budget, plus questions where it wins most (contrast).
Each case: question, type, gold answer; gold rounds (text, session, date) with their rank under E0 / A0 / RR8 / MR4;
the top non-gold rounds E0 and the reference put first; the reader's answers and judge scores for E0 and the reference.
usage: python3 cases.py ev|loc [N_LOSS=30] [N_WIN=10] > cases_<b>.md"""
import json, os, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of
B = sys.argv[1]; NL = int(sys.argv[2]) if len(sys.argv) > 2 else 30; NW = int(sys.argv[3]) if len(sys.argv) > 3 else 10
CFG = {"ev": ("q_ev150.json", "ev150", 4000, "A0"), "loc": ("q_loc150.json", "loc150", 2000, "RR8")}
qf, tag, BUD, REF = CFG[B]
Q = {q["question_id"]: q for q in json.load(open(qf))}
MODELS = ["E0", "A0", "RR8", "MR4"]


def load(m):
    out = {}
    for r in map(json.loads, open(f"ctx_P1{m}_{tag}.jsonl")):
        if r["budget"] == BUD: out[r["qid"]] = r
    return out


C = {m: load(m) for m in MODELS}
ANS, SC = {}, {}
for f in ("out/A6_part1.jsonl",):
    for r in map(json.loads, open(f)):
        if r["stratum"] == B and "error" not in r: ANS[(r["m"], r["qid"])] = r["answer"]
for r in map(json.loads, open("out/A6J_beamgemma431bit.jsonl")):
    if r["stratum"] == B and r.get("status") == "ok": SC[(r["m"], r["qid"])] = r["score"]
MM = {"E0": "CX"}


def recall(qid, m):
    R = rounds_of(Q[qid]); keep = C[m][qid]["keep"]
    if B == "ev":
        gs = {r[0] for r in R if r[3]}; return len({R[i][0] for i in keep} & gs) / len(gs)
    return sum(R[i][3] for i in keep) / sum(r[3] for r in R)


def cut(t, n): t = " ".join(t.split()); return t if len(t) <= n else t[:n] + " …"


ids = [i for i in Q if i in C["E0"] and i in C[REF] and C["E0"][i]["gold_total"]]
d = {i: recall(i, "E0") - recall(i, REF) for i in ids}
loss = sorted(ids, key=lambda i: d[i])[:NL]; win = sorted(ids, key=lambda i: -d[i])[:NW]
print(f"# Cases: {B} (dev, {len(ids)} questions), E0 vs {REF}, recall at {BUD} tokens ({'gold sessions' if B == 'ev' else 'gold rounds'})\n")
print(f"mean recall E0 {100 * st.mean(recall(i, 'E0') for i in ids):.1f}, {REF} {100 * st.mean(recall(i, REF) for i in ids):.1f}; "
      f"E0 worse on {sum(d[i] < 0 for i in ids)}, better on {sum(d[i] > 0 for i in ids)}, equal on {sum(d[i] == 0 for i in ids)}\n")
bt = {}
for i in ids: bt.setdefault(Q[i]["question_type"], []).append(d[i])
print("by type (mean E0 minus ref, n): " + "; ".join(f"{t} {100 * st.mean(v):+.1f} ({len(v)})" for t, v in sorted(bt.items())) + "\n")
for title, lst in (("LOSSES (E0 much worse)", loss), ("WINS (E0 much better, for contrast)", win)):
    print(f"\n## {title}\n")
    for i in lst:
        q = Q[i]; R = rounds_of(q); gold = [k for k, r in enumerate(R) if r[3]]
        pos = {m: {k: n for n, k in enumerate(C[m][i]["rank"])} for m in MODELS}
        print(f"### {i}  [{q['question_type']}]  recall E0 {100 * recall(i, 'E0'):.0f} vs {REF} {100 * recall(i, REF):.0f}  (A0 {100 * recall(i, 'A0'):.0f}, RR8 {100 * recall(i, 'RR8'):.0f}, MR4 {100 * recall(i, 'MR4'):.0f}); history {len(R)} rounds, {len(q['haystack_sessions'])} sessions")
        print(f"- Q ({q['question_date']}): {q['question']}")
        print(f"- Gold answer: {cut(q['answer'], 300)}")
        for m in ("E0", REF):
            mm = MM.get(m, m); print(f"- Reader answer with {m} context (judge {SC.get((mm, i), '?')}): {cut(ANS.get((mm, i), '?'), 300)}")
        print(f"- Gold rounds ({len(gold)}): rank under E0 / A0 / RR8 / MR4 (— = not in 600 pool)")
        for k in gold[:12]:
            rk = " / ".join(str(pos[m].get(k, "—")) for m in MODELS)
            print(f"  - [{rk}] s{R[k][0]} {R[k][1]}: {cut(R[k][2], 350)}")
        if len(gold) > 12: print(f"  - … {len(gold) - 12} more gold rounds")
        for m in ("E0", REF):
            top = [k for k in C[m][i]["rank"] if not R[k][3]][:5]
            print(f"- Top non-gold rounds {m} put first:")
            for k in top: print(f"  - #{pos[m][k]} s{R[k][0]} {R[k][1]}: {cut(R[k][2], 250)}")
        print()
