"""S1 data: general relevance groups from BGE full data (retrieval subsets), scored by the Qwen3-Reranker-8B teacher.
Output rows use the same schema as mkdata.py: {query, docs, label=0, kinds, teacher, src}.
The query string is the plain Qwen3-Reranker format (task instruction + query, no <Selected> block), so S1 teaches
general relevance in exactly the surface format the selector uses later.

usage: python3 s1data.py --n 120000 --out s1_groups.jsonl --rr-port 18314,18315 [--par 32]"""
import argparse, ast, collections, glob, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import PFX, SFX
from k1 import _post

D = os.path.expanduser("~/data/rtm/train_data/bge-full-data/data")
SKIP = ("classification", "clustering", "sts", "arxiv_title", "biorxiv_title", "medrxiv_title", "arguana")


def lst(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception:
            try: return json.loads(x)
            except Exception: return [x]
    return list(x) if x is not None else []


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000); ap.add_argument("--out", default="s1_groups.jsonl")
    ap.add_argument("--rr-port", default="18314,18315"); ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--par", type=int, default=32); ap.add_argument("--max-chars", type=int, default=1500)
    a = ap.parse_args()
    rng = random.Random(0)
    by_src = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(D, "*.parquet"))):
        src = os.path.basename(f).split("-0")[0]
        if any(s in src.lower() for s in SKIP): continue
        by_src[src].append(f)
    srcs = sorted(by_src); per = a.n // len(srcs) + 1
    print("sources", len(srcs), srcs, flush=True)
    rows = []
    for s in srcs:
        fs = by_src[s][:]; rng.shuffle(fs); got = 0
        for f in fs:
            df = pd.read_parquet(f)
            for _, r in df.sample(frac=1, random_state=1).iterrows():
                pos, neg = lst(r["pos"]), lst(r["neg"])
                if not pos or len(neg) < 4: continue
                inst = (r.get("prompt") or "Given a query, retrieve relevant passages that answer it").strip().rstrip(".")
                negs = rng.sample(neg, min(a.K, len(neg)))
                rows.append({"src": s, "inst": inst, "q": str(r["query"])[:2000], "docs": [str(rng.choice(pos))] + [str(x) for x in negs]})
                got += 1
                if got >= per: break
            if got >= per: break
        print(s, got, flush=True)
    rng.shuffle(rows); rows = rows[: a.n]
    ports = a.rr_port.split(","); lk = threading.Lock(); cnt = [0]
    done = 0
    if os.path.exists(a.out): done = sum(1 for _ in open(a.out))
    f = open(a.out, "a")

    def one(r):
        q = PFX + f"<Instruct>: {r['inst']}\n<Query>: {r['q']}\n"
        docs = [f"<Document>: {d[:a.max_chars]}" + SFX for d in r["docs"]]
        with lk: cnt[0] += 1; p = ports[cnt[0] % len(ports)]
        try:
            res = _post(f"http://127.0.0.1:{p}/v1/rerank", {"model": "Qwen3-Reranker-8B", "query": q, "documents": docs})
        except Exception:
            return
        t = [0.0] * len(docs)
        for x in res["results"]: t[x["index"]] = x["relevance_score"]
        keep = [i for i in range(len(docs)) if i == 0 or t[i] <= t[0] + 0.1]   # Qwen3-style false-negative mask
        if len(keep) < 8 or t[0] < 0.05: return                                   # teacher disagrees with the label: drop
        out = {"qid": r["src"], "type": "general", "query": q, "docs": [docs[i] for i in keep], "label": 0,
               "kinds": ["pos"] + ["hard"] * (len(keep) - 1), "teacher": [t[i] for i in keep]}
        with lk:
            f.write(json.dumps(out) + "\n"); f.flush()
            if cnt[0] % 2000 == 0: print("scored", cnt[0], flush=True)

    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, rows[done:]))
    print("S1DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
