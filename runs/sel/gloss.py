"""Per-group listwise loss of a served selector (D3 hard mining). usage: python3 gloss.py GROUPS.jsonl PORTS OUT [--par 16]; OUT lines: index loss"""
import json, math, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
inp, ports, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]; par = int(sys.argv[sys.argv.index("--par") + 1]) if "--par" in sys.argv else 16
L = [json.loads(x) for x in open(inp)]
def score(port, q, docs):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/rerank", data=json.dumps({"model": "sel", "query": q, "documents": docs}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: res = json.loads(r.read())["results"]
    sc = [None] * len(docs)
    for x in res: sc[x["index"]] = x["relevance_score"]   # same endpoint and scores as infer.py
    if any(v is None or not math.isfinite(v) for v in sc): raise SystemExit("INVALID: missing or non-finite score")
    return sc
def one(i):
    r = L[i]; s = score(ports[i % len(ports)], r["query"], r["docs"]); z = [math.log(max(x, 1e-6) / max(1 - x, 1e-6)) for x in s]
    m = max(z); lse = m + math.log(sum(math.exp(x - m) for x in z)); return i, lse - z[r["label"]]
with ThreadPoolExecutor(par) as ex: res = list(ex.map(one, range(len(L))))
open(out, "w").writelines(f"{i} {v:.6f}\n" for i, v in res); print("loss rows", len(res))
