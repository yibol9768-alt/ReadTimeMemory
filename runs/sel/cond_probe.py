"""B0a conditional-signal probe (10-31): does showing half of the gold rounds as <Selected> raise the remaining gold?
Same untrained model, same candidates, conditioned vs empty <Selected>. Question-level recall of the remaining gold in the top N.
usage: python3 cond_probe.py QFILE IDS POOLDIR --port P1,P2 --out FILE"""
import argparse, json, os, random, statistics as st, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of
from fmt import query_str, doc_str, split_round, round_prev
ap = argparse.ArgumentParser(); ap.add_argument("qfile"); ap.add_argument("ids"); ap.add_argument("pooldir")
ap.add_argument("--port", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args(); ports = a.port.split(",")
keep = set(open(a.ids).read().split()); Q = [q for q in json.load(open(a.qfile)) if q["question_id"] in keep]
def score(port, q, docs):
    out = []
    for i in range(0, len(docs), a.bs):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/rerank", data=json.dumps({"model": "sel", "query": q, "documents": docs[i:i + a.bs]}).encode(), headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=600).read())["results"]; v = [0.0] * len(docs[i:i + a.bs])
        for x in res: v[x["index"]] = x["relevance_score"]
        out += v
    return out
def one(x):
    k, q = x; qid = q["question_id"]; p = json.load(open(os.path.join(a.pooldir, qid + ".json"))); g = p["gold"]; gi = [i for i in range(len(g)) if g[i]]
    if len(gi) < 4: return None
    rng = random.Random(qid); rng.shuffle(gi); S, rest = gi[: len(gi) // 2], gi[len(gi) // 2:]
    R = rounds_of(q); prev = round_prev(q); cand = [i for i in p["fused"][:600] if i not in S]
    if not set(rest) & set(cand): return None
    docs = [doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in cand]
    sel = [(R[i][1], split_round(R[i][2])[0], prev[i]) for i in sorted(S)]
    port = ports[k % len(ports)]; s0 = score(port, query_str(q["question"], q["question_date"], []), docs); s1 = score(port, query_str(q["question"], q["question_date"], sel), docs)
    o = {"qid": qid, "type": q["question_type"], "k": len(gi), "rest": len(rest)}
    for name, s in (("plain", s0), ("cond", s1)):
        top = [cand[j] for j in sorted(range(len(cand)), key=lambda j: -s[j])]
        for N in (5, 18): o[f"{name}{N}"] = len(set(top[:N]) & set(rest)) / len(rest)
    return o
with ThreadPoolExecutor(8) as ex: rows = [r for r in ex.map(one, enumerate(Q[: a.limit] if a.limit else Q)) if r]
json.dump(rows, open(a.out, "w"))
rng = random.Random(0)
for N in (5, 18):
    for t in ("event_ordering", "summarization", None):
        v = [r for r in rows if t is None or r["type"] == t]
        d = [r[f"cond{N}"] - r[f"plain{N}"] for r in v]; bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(2000))
        print(f"top{N} {t or 'all':15s} n={len(v)} plain {100 * st.mean(r[f'plain{N}'] for r in v):.1f} cond {100 * st.mean(r[f'cond{N}'] for r in v):.1f} diff {100 * st.mean(d):+.1f} [{100 * bs[50]:+.1f},{100 * bs[1949]:+.1f}]")
