"""Candidate pools for the selector: dense (Qwen3-Embedding-0.6B) + BM25 RRF over all rounds (top P), plus unconditioned
teacher scores from Qwen3-Reranker-8B on the top T. Same first stage as the k1 b2 baselines, so every method sees the
same pool. Writes pool/<qid>.json = {"fused": [round idx...], "rr8": {idx: score}, "stage": [stage label per round or null]}.

usage: python3 pool.py QFILE --emb-port 18311 --rr-port 18312 [--P 600 --T 300 --par 8]"""
import argparse, hashlib, json, math, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k1  # copied next to this file on H20
from k1 import BM25, rounds_of, rrf, toks, _post


def round_stages(q):
    """stage label per k1 round (from the user turn), env data only; None for BEAM/LME."""
    out = []
    for sess in q["haystack_sessions"]:
        i = 0
        while i < len(sess):
            st = sess[i].get("stage")
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant": i += 1
            i += 1; out.append(st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qfile"); ap.add_argument("--emb-port", default="18311"); ap.add_argument("--emb-model", default="Qwen3-Embedding-0.6B")
    ap.add_argument("--rr-port", default="18312"); ap.add_argument("--rr-model", default="Qwen3-Reranker-8B")
    ap.add_argument("--P", type=int, default=600); ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--par", type=int, default=8); ap.add_argument("--out", default="pool"); ap.add_argument("--ids", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qs = json.load(open(a.qfile))
    if a.ids: keep = set(open(a.ids).read().split()); qs = [q for q in qs if q["question_id"] in keep]
    inst = "Instruct: Given a question about a user's past chats, retrieve chat turns that help answer it\nQuery: "
    lock = threading.Lock(); n = [0]
    cache, cache_locks = {}, {}

    def one(q):
        path = os.path.join(a.out, q["question_id"] + ".json")
        if os.path.exists(path): return
        R = rounds_of(q)
        key = hashlib.md5("\0".join(r[2] for r in R).encode()).hexdigest()
        with lock: klock = cache_locks.setdefault(key, threading.Lock())
        with klock:   # one embedding pass per distinct history; questions in a benchmark share histories
            if key not in cache: cache[key] = k1.embed([r[2] for r in R], a)
        qe = k1.embed([inst + q["question"]], a)[0]; re_ = cache[key]
        dot = lambda u, v: sum(x * y for x, y in zip(u, v))
        nq = math.sqrt(dot(qe, qe)) or 1
        ds = [dot(qe, e) / (nq * (math.sqrt(dot(e, e)) or 1)) for e in re_]
        bs = BM25([toks(r[2]) for r in R]).scores(toks(q["question"]))
        fused = rrf([sorted(range(len(R)), key=lambda i: -ds[i])[:a.P], sorted(range(len(R)), key=lambda i: -bs[i])[:a.P]])[:a.P]
        rr = k1._rr_scores(q, R, a, fused[:a.T], q["question"]) if a.T else {}
        json.dump({"fused": fused, "rr8": {str(i): s for i, s in rr.items()}, "stage": round_stages(q),
                   "gold": [r[3] for r in R]}, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        with lock:
            n[0] += 1
            if n[0] % 20 == 0: print("pool", n[0], flush=True)

    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, qs))
    print("POOL_DONE", len(qs))


if __name__ == "__main__":
    main()
