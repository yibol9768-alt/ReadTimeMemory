"""A4 cue dependency: paired recall of an edited run against the same model on the original text (11-01). ssum.py keeps the frozen main criterion and is not for edited runs.
usage: python3 cuesum.py TAG_ORIGINAL TAG_EDITED EDITED_QFILE BUDGET [--session]. --session scores at session level as ssum.py does for EverMem."""
import json, random, re, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of
orig = lambda i: re.sub(r"^[a-z]+s\d+-", "", i)
def load(tag, b, gold=None):
    o = {}
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if r["budget"] != b or not r["gold_total"]: continue
        k = orig(r["qid"])
        if k in o: raise SystemExit(f"INVALID duplicate {k} in ctx_{tag}")
        if gold is None: o[k] = r["gold_in"] / r["gold_total"]
        else: g, R = gold[r["qid"]]; o[k] = len({R[i][0] for i in r["keep"]} & g) / len(g)
    return o
if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]; ses = "--session" in sys.argv; Q = json.load(open(a[2])); b = int(a[3]); G0 = G1 = None
    if ses:
        G1 = {}
        for q in Q: R = rounds_of(q); G1[q["question_id"]] = ({r[0] for r in R if r[3]}, R)
        O = json.load(open(a[4])) if len(a) > 4 else None
        if O is None: raise SystemExit("--session needs the original question file as a fifth argument")
        G0 = {}
        for q in O: R = rounds_of(q); G0[q["question_id"]] = ({r[0] for r in R if r[3]}, R)
    x, y = load(a[0], b, G0), load(a[1], b, G1); want = {q.get("orig_question_id", q["question_id"]) for q in Q}; ks = sorted(want & set(x) & set(y))
    if len(ks) != len(want): raise SystemExit(f"INVALID: {len(ks)} of {len(want)} edited questions scored on both sides")
    conv = lambda k: k.split("_")[0] if re.match(r"[hc]\d+_", k) else k   # BEAM questions share conversations (h dev, c independent)
    by = {}
    for k in ks: by.setdefault(conv(k), []).append(y[k] - x[k])
    r = random.Random(0); cs = list(by); B = sorted(st.mean(v for c in r.choices(cs, k=len(cs)) for v in by[c]) for _ in range(8000))
    print(f"{a[1]} vs {a[0]} budget {b}: {100*st.mean(x[k] for k in ks):.1f} -> {100*st.mean(y[k] for k in ks):.1f} change {100*st.mean(y[k]-x[k] for k in ks):+.1f} [{100*B[200]:+.1f},{100*B[7799]:+.1f}] n={len(ks)}")
