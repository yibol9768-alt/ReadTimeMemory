"""Extend 8B-teacher scores in pool/<qid>.json from the top-T0 to the top-T1 fused rounds (so every method, including
the 8B baseline, works on the same P=600 pool). Same prompt as pool.py (k1._rr_scores).
usage: python3 pool_ext.py QFILE --rr-port 18314,18315 --T1 600 [--pooldir pool --par 12]"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k1
from k1 import rounds_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qfile"); ap.add_argument("--rr-port", default="18314,18315"); ap.add_argument("--rr-model", default="Qwen3-Reranker-8B")
    ap.add_argument("--T1", type=int, default=600); ap.add_argument("--pooldir", default="pool"); ap.add_argument("--par", type=int, default=12)
    a = ap.parse_args()
    qs = json.load(open(a.qfile)); n = [0]

    def one(q):
        p = os.path.join(a.pooldir, q["question_id"] + ".json")
        if not os.path.exists(p): return
        pool = json.load(open(p)); rr = pool["rr8"]
        todo = [i for i in pool["fused"][: a.T1] if str(i) not in rr]
        if not todo: return
        R = rounds_of(q)
        new = k1._rr_scores(q, R, a, todo, q["question"])
        rr.update({str(i): s for i, s in new.items()})
        json.dump(pool, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        n[0] += 1
        if n[0] % 20 == 0: print("ext", n[0], flush=True)

    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, qs))
    print("EXT_DONE", flush=True)


if __name__ == "__main__":
    main()
