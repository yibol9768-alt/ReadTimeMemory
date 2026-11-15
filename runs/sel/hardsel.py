"""Hard-question subset with validated hard negatives (10-31).
A question is HARD if the 8B reranker's gold-round recall at the 4k budget is <= --thr. A hard question is kept only if
its hardness is not label noise: the 8B top-K non-gold rounds are judged (27B) against the theme; if any of them reports
progress on the asked aspect, the question is AMBIGUOUS (a false negative in the gold set) and is dropped.
usage: python3 hardsel.py QFILE CTXFILE POOLDIR --ver M@P,... --out hard.json [--thr 0.15 --k 10]"""
import argparse, json, os, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/data/rtm/env"))
from k1 import rounds_of
from fmt import split_round
from gen import Pool

JUDGE = """A user works on one project with an AI assistant. One aspect of the project is "{theme}".
Known steps of this aspect:
{steps}
Does the following message report a development, problem, decision or result that belongs to the aspect "{theme}"
(it may be one of the known steps or another step of the same aspect)? Answer YES or NO only.
MESSAGE: {m}"""

ap = argparse.ArgumentParser()
ap.add_argument("qfile"); ap.add_argument("ctx"); ap.add_argument("pooldir"); ap.add_argument("--ver", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=0.15); ap.add_argument("--k", type=int, default=10)
ap.add_argument("--types", default="event_ordering,summarization")
a = ap.parse_args()
pool = Pool(a.ver); types = a.types.split(",")
Q = {q["question_id"]: q for q in json.load(open(a.qfile))}
rec = {}
for r in map(json.loads, open(a.ctx)):
    if r["budget"] == 4000 and r["gold_total"]: rec[r["qid"]] = r["gold_in"] / r["gold_total"]
hard = [q for q, x in rec.items() if x <= a.thr and Q[q]["question_type"] in types]


def one(qid):
    q = Q[qid]; R = rounds_of(q); p = json.load(open(os.path.join(a.pooldir, qid + ".json")))
    rr = {int(k): v for k, v in p["rr8"].items()}; g = p["gold"]
    order = sorted(p["fused"], key=lambda i: -rr.get(i, -1))
    theme = q["question"]
    for pat in ("my work on ", "related to ", "concerning ", "everything related to ", "about "):
        if pat in theme: theme = theme.split(pat, 1)[1]; break
    theme = theme.split("?")[0].split(" progress")[0].split(" developed")[0].split(" in my project")[0].split(" evolved")[0].split(" that I")[0].strip()
    steps = "\n".join(f"- {s['title']}" for s in q.get("gold_steps", []))
    yes = []
    for i in [i for i in order if not g[i]][: a.k]:
        u = split_round(R[i][2])[0]
        t, _ = pool.chat(JUDGE.format(theme=theme, steps=steps, m=u[:2500]), max_tokens=5, temperature=0)
        if "YES" in t.upper(): yes.append(u[:160].replace("\n", " "))
    return qid, theme, yes


with ThreadPoolExecutor(16) as ex: res = list(ex.map(one, hard))
keep = [qid for qid, _, y in res if not y]
by = collections.Counter(Q[q]["question_type"] for q in hard); byk = collections.Counter(Q[q]["question_type"] for q in keep)
print(f"hard (4k recall <= {a.thr}): {dict(by)}  kept after negative validation: {dict(byk)}")
for qid, th, y in res:
    if y: print(f"  AMBIG {qid} [{th}] {len(y)} on-aspect non-gold, e.g. {y[0]}")
json.dump([Q[q] for q in keep], open(a.out, "w"))
