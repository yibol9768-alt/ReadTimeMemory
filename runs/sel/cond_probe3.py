"""B0 options, no training (10-31). How should selected rounds be shown so that wrong ones do no harm?
Same candidates (plain top 300 minus self5 and placebo), same target. Compared with plain:
 block5 all five in one block; top1 only the best; top2 the best two; each = score with one item at a time, final = (plain + max over items) / 2;
 mix = (plain + block5) / 2; each_plc = `each` on five random low-ranked non-gold rounds (negative control, must not help).
usage: python3 cond_probe3.py QFILE IDS POOLDIR --port P1,P2 --out FILE [--min-gold 4]"""
import argparse, json, os, random, statistics as st, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of
from fmt import query_str, doc_str, split_round, round_prev
ap = argparse.ArgumentParser(); ap.add_argument("qfile"); ap.add_argument("ids"); ap.add_argument("pooldir")
ap.add_argument("--port", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--min-gold", type=int, default=4); ap.add_argument("--C", type=int, default=300)
a = ap.parse_args(); ports = a.port.split(",")
keep = set(open(a.ids).read().split()); Q = [q for q in json.load(open(a.qfile)) if q["question_id"] in keep]
Q = [q for q in Q if sum(map(bool, json.load(open(os.path.join(a.pooldir, q["question_id"] + ".json")))["gold"])) >= a.min_gold]
print("multi-gold questions", len(Q), flush=True)
Ns = (5, 18)
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
    R = rounds_of(q); prev = round_prev(q); port = ports[k % len(ports)]; base = p["fused"][:600]; rng = random.Random(qid)
    docs = {i: doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in base}
    plain = dict(zip(base, score(port, query_str(q["question"], q["question_date"], []), [docs[i] for i in base])))
    order = sorted(base, key=lambda i: -plain[i]); self5 = order[:5]; low = [i for i in order[len(order) // 2:] if not g[i]]; plc = rng.sample(low, min(5, len(low)))
    E = set(self5) | set(plc); cand = [i for i in order[: a.C] if i not in E]; tgt = set(gi) & set(i for i in base if i not in E)
    if not tgt: return None
    def cond(S):
        sel = [(R[i][1], split_round(R[i][2])[0], prev[i]) for i in S]
        return dict(zip(cand, score(port, query_str(q["question"], q["question_date"], sel), [docs[i] for i in cand])))
    s1 = [cond([i]) for i in self5]; sp = [cond([i]) for i in plc]; b5 = cond(self5); b2 = cond(self5[:2])
    each = {i: (plain[i] + max(s[i] for s in s1)) / 2 for i in cand}; eplc = {i: (plain[i] + max(s[i] for s in sp)) / 2 for i in cand}
    S = {"plain": plain, "block5": b5, "top1": s1[0], "top2": b2, "each": each, "each_plc": eplc, "mix": {i: (plain[i] + b5[i]) / 2 for i in cand}}
    o = {"qid": qid, "type": q["question_type"], "k": len(gi), "self5gold": sum(bool(g[i]) for i in self5), "top1gold": int(bool(g[self5[0]]))}
    for n, s in S.items():
        rk = sorted(cand, key=lambda i: -s[i])
        for N in Ns: o[f"{n}{N}"] = len(set(rk[:N]) & tgt) / len(tgt)
    return o
with ThreadPoolExecutor(12) as ex: rows = [r for r in ex.map(one, enumerate(Q[: a.limit] if a.limit else Q)) if r]
json.dump(rows, open(a.out, "w"))
rng = random.Random(0)
print(f"questions {len(rows)}; gold in self5 {st.mean(r['self5gold'] for r in rows):.2f}; top1 gold {st.mean(r['top1gold'] for r in rows):.2f}")
for N in Ns:
    for n in ("block5", "top1", "top2", "each", "mix", "each_plc"):
        d = [r[f"{n}{N}"] - r[f"plain{N}"] for r in rows]; bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
        print(f"{n:9s} top{N} {100 * st.mean(r[f'{n}{N}'] for r in rows):.1f} vs plain {100 * st.mean(r[f'plain{N}'] for r in rows):.1f} diff {100 * st.mean(d):+.1f} [{100 * bs[34]:+.1f},{100 * bs[3965]:+.1f}]")
