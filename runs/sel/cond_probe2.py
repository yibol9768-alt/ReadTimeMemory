"""B0 probe with controls (10-31, after audit). Target = gold not in any selected set; every condition ranks the same candidates.
Conditions: plain (empty <Selected>), self5 (plain top 5), plc (5 random non-gold rounds from the lower half of the plain ranking),
self5m (self5 with dates masked), wrong (top non-gold rounds of the plain ranking, half-gold count; own candidate set, kept comparable with the first probe).
Main number: self5 minus plc. far = only gold more than 3 rounds away from self5. Intervals are 98.3% (three arms).
usage: python3 cond_probe2.py QFILE IDS POOLDIR --port P1,P2 --out FILE [--min-gold 4]"""
import argparse, json, os, random, statistics as st, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of
from fmt import query_str, doc_str, split_round, round_prev
ap = argparse.ArgumentParser(); ap.add_argument("qfile"); ap.add_argument("ids"); ap.add_argument("pooldir")
ap.add_argument("--port", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--min-gold", type=int, default=4)
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
def hit(cand, s, tgt, N):
    return len(set(sorted(cand, key=lambda i: -s[i])[:N]) & tgt) / len(tgt)
def one(x):
    k, q = x; qid = q["question_id"]; p = json.load(open(os.path.join(a.pooldir, qid + ".json"))); g = p["gold"]; gi = [i for i in range(len(g)) if g[i]]
    if len(gi) < a.min_gold: return None
    R = rounds_of(q); prev = round_prev(q); port = ports[k % len(ports)]; base = p["fused"][:600]; rng = random.Random(qid)
    if not set(gi) & set(base): return None
    docs = {i: doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in base}
    plain = dict(zip(base, score(port, query_str(q["question"], q["question_date"], []), [docs[i] for i in base])))
    order = sorted(base, key=lambda i: -plain[i]); ng = [i for i in order if not g[i]]
    self5 = order[:5]; low = [i for i in order[len(order) // 2:] if not g[i]]; plc = rng.sample(low, min(5, len(low)))
    wrong = ng[: max(1, len(gi) // 2)]; E = set(self5) | set(plc); cand = [i for i in base if i not in E]; tgt = set(gi) & set(cand)
    o = {"qid": qid, "type": q["question_type"], "k": len(gi), "self5gold": sum(bool(g[i]) for i in self5)}
    if not tgt: return None
    def cond(S, mask=False, c=cand):
        sel = [("-" if mask else R[i][1], split_round(R[i][2])[0], prev[i]) for i in S]
        return dict(zip(c, score(port, query_str(q["question"], q["question_date"], sel), [docs[i] for i in c])))
    far = {i for i in tgt if all(abs(i - j) > 3 for j in self5)}
    sc = {"plain": plain, "self5": cond(self5), "plc": cond(plc), "self5m": cond(self5, True)}
    for name, s in sc.items():
        for N in Ns:
            o[f"{name}{N}"] = hit(cand, s, tgt, N)
            if far: o[f"{name}_far{N}"] = hit(cand, s, far, N)
    c2 = [i for i in base if i not in wrong]; t2 = set(gi) & set(c2)
    if t2:
        s = cond(wrong, c=c2)
        for N in Ns: o[f"wrong{N}"] = hit(c2, s, t2, N); o[f"wrongp{N}"] = hit(c2, plain, t2, N)
    return o
with ThreadPoolExecutor(16) as ex: rows = [r for r in ex.map(one, enumerate(Q[: a.limit] if a.limit else Q)) if r]
json.dump(rows, open(a.out, "w"))
rng = random.Random(0)
def show(lbl, x, y):
    v = [r for r in rows if x in r and y in r]
    if not v: return
    d = [r[x] - r[y] for r in v]; bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
    print(f"{lbl:22s} n={len(v)} {100 * st.mean(r[x] for r in v):.1f} vs {100 * st.mean(r[y] for r in v):.1f} diff {100 * st.mean(d):+.1f} [{100 * bs[34]:+.1f},{100 * bs[3965]:+.1f}]")
print(f"questions {len(rows)}; gold in self5 {st.mean(r['self5gold'] for r in rows):.2f}")
for N in Ns:
    show(f"MAIN self5-plc top{N}", f"self5{N}", f"plc{N}")
    show(f"self5-plain top{N}", f"self5{N}", f"plain{N}")
    show(f"plc-plain top{N}", f"plc{N}", f"plain{N}")
    show(f"far self5-plc top{N}", f"self5_far{N}", f"plc_far{N}")
    show(f"masked self5m-plc top{N}", f"self5m{N}", f"plc{N}")
    show(f"wrong-plain top{N}", f"wrong{N}", f"wrongp{N}")
