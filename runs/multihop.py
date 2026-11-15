"""WP2, no new LLM calls (cache only; any cache miss raises). For multi-hop questions answered so far:
did the context contain every evidence turn, how many records the evidence is spread over, and was the answer right?
Arms: pro6 (gpt-6-luna reads rec_sess / BM25) and plus56 (5.6-luna reads rec_sess and hyb_mq).
Also dumps the wrong answers whose evidence was fully in context, for reading by hand (runs/multihop_wrong.txt)."""
import json, os, sys, collections, urllib.request
os.environ.setdefault("V3_PROXY_PORT", "18319")
sys.argv = ["x"]


def _no_net(*a, **k):
    raise RuntimeError("cache miss: this analysis makes no LLM calls")


from mem import llm, locomo_exp as L, retrieve as R
llm.urllib.request.urlopen = _no_net
d = json.load(open("data/locomo10.json"))
mems = {ci: json.load(open(f"runs/locomo/mem_{ci}.json")) for ci in range(10)}
sss = {ci: L.sessions(d[ci]) for ci in range(10)}
turn_text = {(ci, t["id"]): f"({s['date']}) {t['speaker']}: {t['text']}" for ci in sss for s in sss[ci] for t in s["turns"]}


def evidence(q):
    return {e.strip() for e in q.get("evidence", []) for e in e.split(";") if e.strip()}


def ctx_sources(ctx, ci):
    """Turn ids visible in a context: cited by a shown record, or quoted directly."""
    src = set()
    for r in mems[ci]["rec_sess"]:
        if r["text"] in ctx:
            src |= set(r["sources"])
    return src


def spread(ev, ci):
    """Fewest records whose sources cover the evidence (greedy), or None if some turn is in no record."""
    recs = [set(r["sources"]) & ev for r in mems[ci]["rec_sess"]]
    left, n = set(ev), 0
    while left:
        best = max(recs, key=lambda s: len(s & left))
        if not best & left:
            return None
        left -= best; n += 1
    return n


qs = {q["qid"]: q for q in L.questions(d, 50)}
rows = {"pro6": [json.loads(l) for l in open("runs/locomo/judged.jsonl")],
        "plus56": [json.loads(l) for l in open("runs/locomo/plus56/judged.jsonl")]}
table = collections.defaultdict(collections.Counter)
wrong = []
for arm, rs in rows.items():
    seen = set()
    for r in rs:
        if r["cat"] != 1 or r["cond"] not in ("rec_sess", "hyb_mq") or (r["qid"], r["cond"]) in seen:
            continue
        seen.add((r["qid"], r["cond"]))
        q = qs[r["qid"]]; ci = q["conv"]; ev = evidence(q)
        if not ev:
            continue
        ctx = L.context(r["cond"], q["question"], sss[ci], mems[ci])
        got = "all" if ev <= ctx_sources(ctx, ci) else "part" if ev & ctx_sources(ctx, ci) else "none"
        sp = spread(ev, ci)
        key = (arm, r["cond"])
        table[key][(got, r["correct"])] += 1
        table[key + ("spread",)][(min(sp, 3) if sp else "unwritten", r["correct"])] += 1
        if got == "all" and not r["correct"]:
            wrong.append((arm, r["cond"], q["question"], r["gold"], r["pred"], [turn_text[(ci, e)] for e in sorted(ev) if (ci, e) in turn_text]))

for key in sorted(k for k in table if len(k) == 2):
    c = table[key]
    print(f"\n{key[0]} {key[1]}  multi-hop n={sum(c.values())}")
    for g in ("all", "part", "none"):
        n = c[(g, 1)] + c[(g, 0)]
        if n:
            print(f"  evidence {g:4s} in context: {n:3d} questions, correct {100 * c[(g, 1)] / n:5.1f}%")
    s = table[key + ("spread",)]
    for k in (1, 2, 3, "unwritten"):
        n = s[(k, 1)] + s[(k, 0)]
        if n:
            print(f"  evidence spread over {str(k) + ('+' if k == 3 else ''):9s} records: {n:3d} questions, correct {100 * s[(k, 1)] / n:5.1f}%")
with open("runs/multihop_wrong.txt", "w") as f:
    for arm, cond, q, g, p, ev in wrong:
        f.write(f"[{arm} {cond}] Q: {q}\n  gold: {g}\n  pred: {p}\n" + "".join(f"  ev: {e}\n" for e in ev) + "\n")
print(f"\n{len(wrong)} wrong answers with all evidence in context -> runs/multihop_wrong.txt")
