"""D4 case analysis 1: area by cell (half, type, count hint, carrier) for C, P0, P0P1, I(s300 vs C s300), E0, with paired CI."""
import json, re, random, statistics as st, collections
BUD = [1000, 2000, 4000, 8000, 16000]
Q = {q["question_id"]: q for q in json.load(open("env_h4.json"))}
CNT = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\b\s*(items?|things?|events?|decisions?|updates?|changes?|problems?|issues?)|\bList \d+|\ball \d+\b|\bthe \d+\b|\b\d+ (key |chronological |distinct )", re.I)
def per_q(tag):
    E = collections.defaultdict(dict)
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if r["gold_total"]: E[r["qid"]][r["budget"]] = r["gold_in"] / r["gold_total"]
    return {k: 100 * st.mean(v[b] for b in BUD) for k, v in E.items()}
T = {"C": "D4Cs600_h4", "P0": "D4P0s600_h4", "P0P1": "D4P0P1s600_h4", "E0": "E0_h4", "C300": "D4Cs300_h4", "I300": "D4Is300_h4"}
A = {k: per_q(v) for k, v in T.items()}
def ci(d):
    r = random.Random(0); b = sorted(st.mean(r.choices(d, k=len(d))) for _ in range(3000)); return b[75], b[2924]
def feat(q):
    return dict(half="H4j" if int(q["question_id"].split("_")[0][1:]) % 2 else "H4s", type=q["question_type"], count="count" if CNT.search(q["question"]) else "nocount",
                group="group" if any("group" in c for c in q["carriers"]) else "nogroup", scope=q.get("scope"), ngold=min(len(q["gold_stages"]) if isinstance(q["gold_stages"], list) else 1, 9))
F = {k: feat(q) for k, q in Q.items()}
print("count-hint share by type:", {t: f"{sum(1 for k in Q if F[k]['type']==t and F[k]['count']=='count')}/{sum(1 for k in Q if F[k]['type']==t)}" for t in sorted({f['type'] for f in F.values()})})
def show(title, key):
    print(f"\n== {title}")
    cells = collections.defaultdict(list)
    for k in Q: cells[key(F[k])].append(k)
    for c in sorted(cells, key=str):
        ks = cells[c]
        if len(ks) < 15: continue
        line = f"{str(c):32s} n={len(ks):3d} C {st.mean(A['C'][k] for k in ks):5.1f} E0 {st.mean(A['E0'][k] for k in ks):5.1f}"
        for m in ("P0", "P0P1"):
            d = [A[m][k] - A["C"][k] for k in ks]; lo, hi = ci(d); line += f" | {m}-C {st.mean(d):+5.1f} [{lo:+.1f},{hi:+.1f}]"
        d = [A["I300"][k] - A["C300"][k] for k in ks]; line += f" | I-C@300 {st.mean(d):+4.1f}"
        print(line)
show("all 604 by type", lambda f: f["type"])
show("by count hint", lambda f: f["count"])
show("by type x count", lambda f: (f["type"], f["count"]))
show("by carrier", lambda f: f["group"])
show("by half x carrier", lambda f: (f["half"], f["group"]))
show("by scope", lambda f: f["scope"])
