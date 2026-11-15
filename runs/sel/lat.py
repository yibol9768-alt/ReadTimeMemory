"""Step 2 (PLAN_NEXT v2): reranking-stage latency, ConvMemory-style: only the reranker call, after warm-up.
One query = score the top-P pool rounds of one question (same documents and request batching of 64 as infer.py).
Reports per-query latency (sequential, one query at a time) and throughput (--par concurrent queries).
usage: python3 lat.py q_lmes150.json --pooldir pool_lmes150 --port 18400 --tag A0 [--plain --trunc 512] [--n 50 --warm 5 --P 600 --par 16]"""
import argparse, json, os, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of, _post
from fmt import query_str, doc_str, split_round, round_prev
from infer import plain_doc

ap = argparse.ArgumentParser()
ap.add_argument("qfile"); ap.add_argument("--pooldir", required=True); ap.add_argument("--port", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--plain", action="store_true"); ap.add_argument("--trunc", type=int, default=0)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--warm", type=int, default=5); ap.add_argument("--P", type=int, default=600)
ap.add_argument("--par", type=int, default=16); ap.add_argument("--out", default="out/LAT.jsonl")
ap.add_argument("--chunk", type=int, default=0, help="documents per request; 0 = all candidates of a query in one request (audit 11-02)")
a = ap.parse_args()

qs = json.load(open(a.qfile))[: a.warm + a.n]
jobs = []
for q in qs:
    R = rounds_of(q); prev = round_prev(q); P = json.load(open(os.path.join(a.pooldir, q["question_id"] + ".json")))["fused"][: a.P]
    if a.plain: jobs.append((q["question"], [plain_doc(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in P]))
    else: jobs.append((query_str(q["question"], q["question_date"], []), [doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in P]))


def run(job):
    qs_, docs = job; t = time.perf_counter()
    c = a.chunk or len(docs)
    for k in range(0, len(docs), c):
        body = {"model": "sel", "query": qs_, "documents": docs[k:k + c]}
        if a.trunc: body["truncate_prompt_tokens"] = a.trunc
        _post(f"http://127.0.0.1:{a.port}/v1/rerank", body)
    return time.perf_counter() - t


for j in jobs[: a.warm]: run(j)
lat = [run(j) for j in jobs[a.warm:]]
t0 = time.perf_counter()
with ThreadPoolExecutor(a.par) as ex: list(ex.map(run, jobs[a.warm:]))
thr = len(jobs[a.warm:]) / (time.perf_counter() - t0)
ndoc = st.mean(len(d) for _, d in jobs[a.warm:])
row = {"tag": a.tag, "n": len(lat), "docs_per_query": round(ndoc, 1), "lat_median_s": round(st.median(lat), 3),
       "lat_p90_s": round(sorted(lat)[int(.9 * len(lat)) - 1], 3), "throughput_qps": round(thr, 3), "par": a.par, "chunk": a.chunk or "all", "P": a.P}
print(json.dumps(row)); open(a.out, "a").write(json.dumps(row) + "\n")
