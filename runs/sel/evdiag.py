"""EverMem diagnosis (audit 11-02, item 12), CPU only.
For each budget: oracle ceiling (pack gold rounds first, shortest first, then nothing) for session and round recall;
session and round (message) recall of each model; by question type; gold-round length vs others.
usage: python3 evdiag.py [MODELS=A0,E0,MR4,RR8,T1,GTE]"""
import json, statistics as st, sys, collections
sys.path.insert(0, ".")
from k1 import rounds_of, pack, ntok
MODELS = (sys.argv[1] if len(sys.argv) > 1 else "A0,E0,MR4,RR8,T1,GTE").split(",")
BUD = [1000, 2000, 4000, 16000]
Q = {q["question_id"]: q for q in json.load(open("q_ev150.json"))}
RR = {qid: rounds_of(q) for qid, q in Q.items()}
G = {qid: [i for i, r in enumerate(R) if r[3]] for qid, R in RR.items()}
ids = [q for q in Q if G[q]]
def rec(R, keep, g):
    gs = {R[i][0] for i in g}
    return len({R[i][0] for i in keep} & gs) / len(gs), len(set(keep) & set(g)) / len(g)
print("questions", len(ids), "mean gold rounds", round(st.mean(len(G[q]) for q in ids), 1),
      "mean gold sessions", round(st.mean(len({RR[q][i][0] for i in G[q]}) for q in ids), 1))
lg = [ntok(RR[q][i][2]) for q in ids for i in G[q]]; lo = [ntok(r[2]) for q in ids[:20] for r in RR[q]]
print("gold round tokens median", st.median(lg), "| all rounds median", st.median(lo))
types = collections.Counter(Q[q]["question_type"] for q in ids); print("types", dict(types))
print(f"\n{'':10s}" + "".join(f"{'sess@'+str(b//1000)+'k':>10s}{'msg@'+str(b//1000)+'k':>10s}" for b in BUD))
# oracle: one gold round per gold session first (shortest), then remaining gold rounds
row = ""
for b in BUD:
    s_, m_ = [], []
    for q in ids:
        R = RR[q]; g = G[q]; by = collections.defaultdict(list)
        for i in g: by[R[i][0]].append(i)
        first = sorted((min(v, key=lambda i: ntok(R[i][2])) for v in by.values()), key=lambda i: ntok(R[i][2]))
        rest = sorted((i for i in g if i not in set(first)), key=lambda i: ntok(R[i][2]))
        p = pack(Q[q], R, first + rest, b); a, c = rec(R, p["keep"], g); s_.append(a); m_.append(c)
    row += f"{100*st.mean(s_):10.1f}{100*st.mean(m_):10.1f}"
print(f"{'oracle':10s}" + row)
D = {}
for m in MODELS:
    D[m] = {}
    for r in map(json.loads, open(f"ctx_P1{m}_ev150.jsonl")):
        if r["qid"] in G and G[r["qid"]]: D[m][(r["qid"], r["budget"])] = rec(RR[r["qid"]], r["keep"], G[r["qid"]])
    print(f"{m:10s}" + "".join(f"{100*st.mean(D[m][(q,b)][0] for q in ids):10.1f}{100*st.mean(D[m][(q,b)][1] for q in ids):10.1f}" for b in BUD))
print("\nsession recall at 2k by question type")
print(f"{'type':28s}{'n':>4s}" + "".join(f"{m:>8s}" for m in MODELS))
for t in sorted(types):
    qs = [q for q in ids if Q[q]["question_type"] == t]
    print(f"{t[:28]:28s}{len(qs):4d}" + "".join(f"{100*st.mean(D[m][(q,2000)][0] for q in qs):8.1f}" for m in MODELS))
