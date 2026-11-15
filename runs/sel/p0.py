"""P0: how far can the candidate pool be pushed without training? (BEAM, gold-round recall at pool size P)

Methods (all at the same pool size P):
  F      fused dense+BM25 RRF (current pools)
  D      dense only (round text)
  DC     dense with context-augmented rounds (previous user turn of the same session + date prepended)
  FC     RRF(DC, BM25)
  STRAT  FC top P/2, then fill round-robin with the best DC round of each session
  PRF    FC top-200 + neighbours of seeds; seeds = top-8 of the 8B reranker in the current pool
  PRFQ   like PRF but the expansion query is normalize(q_emb + seed_emb)
  ORACLE PRF with 5 random gold seeds from FC top-300 (upper bound for set-conditioned expansion)
usage: python3 p0.py Q1.json[,Q2.json] --emb-port 18311 --P 300,600"""
import argparse, collections, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k1
from k1 import rounds_of, BM25, toks, rrf

INST = "Instruct: Given a question about a user's past chats, retrieve chat turns that help answer it\nQuery: "


def ctx_texts(q):
    out = []
    for sess, date in zip(q["haystack_sessions"], q["haystack_dates"]):
        i, prev = 0, ""
        while i < len(sess):
            t = [sess[i]]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant": t.append(sess[i + 1]); i += 1
            i += 1
            body = "\n".join(f"{x['role'].capitalize()}: {x['content']}" for x in t)
            out.append((f"[{date}] (previous message: {prev[:300]})\n" if prev else f"[{date}]\n") + body)
            if t[0]["role"] == "user": prev = t[0]["content"]
    return out


def emb(texts, a):
    class A: pass
    o = A(); o.emb_port = a.emb_port; o.emb_model = a.emb_model
    E = np.array(k1.embed(texts, o), dtype=np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qfiles"); ap.add_argument("--emb-port", default="18311"); ap.add_argument("--emb-model", default="Qwen3-Embedding-0.6B")
    ap.add_argument("--P", default="300,600"); ap.add_argument("--pooldir", default="pool"); ap.add_argument("--cache", default="p0cache")
    a = ap.parse_args()
    Ps = [int(x) for x in a.P.split(",")]
    os.makedirs(a.cache, exist_ok=True)
    qs = [q for f in a.qfiles.split(",") for q in json.load(open(f)) if q["n_src"]]
    by_conv = collections.defaultdict(list)
    for q in qs: by_conv[q["question_id"].split("_")[0]].append(q)
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    rng = random.Random(0)
    for conv, cq in by_conv.items():
        q0 = cq[0]; R = rounds_of(q0)
        cp = os.path.join(a.cache, conv + ".npz")
        if os.path.exists(cp):
            z = np.load(cp); E, EC = z["E"], z["EC"]
        else:
            E = emb([r[2] for r in R], a); EC = emb(ctx_texts(q0), a); np.savez(cp, E=E, EC=EC)
        bm = BM25([toks(r[2]) for r in R])
        sess = np.array([r[0] for r in R])
        for q in cq:
            gold = {i for i, r in enumerate(rounds_of(q)) if r[3]}
            if not gold: continue
            qe = emb([INST + q["question"]], a)[0]
            d, dc = E @ qe, EC @ qe
            bs = bm.scores(toks(q["question"]))
            rk_d = list(np.argsort(-d)); rk_dc = list(np.argsort(-dc)); rk_b = sorted(range(len(R)), key=lambda i: -bs[i])
            F = json.load(open(os.path.join(a.pooldir, q["question_id"] + ".json")))
            rr = {int(k): v for k, v in F["rr8"].items()}
            FC = rrf([rk_dc[:1000], rk_b[:1000]])
            def strat(P):
                out = FC[:P // 2]; seen = set(out)
                order_by_s = collections.defaultdict(list)
                for i in rk_dc: order_by_s[sess[i]].append(i)
                k = 0
                while len(out) < P and k < 50:
                    for s in order_by_s:
                        l = [i for i in order_by_s[s] if i not in seen]
                        if l and len(out) < P: out.append(l[0]); seen.add(l[0])
                    k += 1
                return out
            def prf(seeds, P, mix_q=False):
                out = list(FC[:200]); seen = set(out)
                lists = []
                for s in seeds:
                    v = EC[s] + (qe if mix_q else 0); v = v / np.linalg.norm(v)
                    lists.append([int(i) for i in np.argsort(-(EC @ v))[:400]])
                k = 0
                while len(out) < P and k < 400:
                    for l in lists:
                        if l[k] not in seen and len(out) < P: out.append(l[k]); seen.add(l[k])
                    k += 1
                for i in FC:
                    if len(out) >= P: break
                    if i not in seen: out.append(i); seen.add(i)
                return out
            seeds = sorted([i for i in F["fused"][:300] if i in rr], key=lambda i: -rr[i])[:8]
            gin = [i for i in FC[:300] if i in gold]
            oseeds = rng.sample(gin, min(5, len(gin))) if gin else seeds[:5]
            t = q["question_type"]
            for P in Ps:
                cands = {"F": F["fused"][:P], "D": rk_d[:P], "DC": rk_dc[:P], "FC": FC[:P], "STRAT": strat(P),
                         "PRF": prf(seeds, P), "PRFQ": prf(seeds, P, True), "ORACLE": prf(oseeds, P, True)}
                for m, c in cands.items():
                    res[(t, P)][m].append(len(gold & set(int(x) for x in c)) / len(gold))
        print("conv", conv, "done", flush=True)
    meths = ["F", "D", "DC", "FC", "STRAT", "PRF", "PRFQ", "ORACLE"]
    for P in Ps:
        print(f"\n== pool size {P}: gold-round recall (%)")
        print(f"{'type':24s}" + "".join(f"{m:>8s}" for m in meths))
        for t in sorted({k[0] for k in res}):
            v = res[(t, P)]
            print(f"{t[:24]:24s}" + "".join(f"{100 * np.mean(v[m]):8.1f}" for m in meths))


if __name__ == "__main__":
    main()
