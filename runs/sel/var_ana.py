"""Difficulty-variant analysis (10-31, audit-required metrics). Per arm and per user, EO + summarization only:
  recall@4k/16k (token budget, from ctx rows), recall@N rounds (N = budget / BEAM mean round length ~1000 tok),
  gold AUC vs non-gold in the 8B order over the P=600 pool, gold P(yes) median,
  source of the 8B top-10 non-gold rounds (same-theme other stage / twin theme / other theme / adj / side / idle / fu / filler),
  history length; plus the same numbers for BEAM-500K (reference, no kinds).
usage: python3 var_ana.py V0,VA,VB,VC"""
import json, os, sys, collections, statistics as st, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of

T = ("event_ordering", "summarization")
NR = {4000: 4, 16000: 16}   # rounds that fit at BEAM's ~1000 tok per round


def kinds_of(q):
    out = []
    for sess in q["haystack_sessions"]:
        i = 0
        while i < len(sess):
            k = (sess[i].get("kind"), sess[i].get("stage"))
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant": i += 1
            i += 1; out.append(k)
    return out


def auc(pos, neg):
    if not pos or not neg: return None
    neg = sorted(neg); import bisect
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos) * len(neg))


def analyse(qfile, ctxfile, pooldir, label):
    qs = {q["question_id"]: q for q in json.load(open(qfile)) if q["question_type"] in T}
    ctx = collections.defaultdict(dict)
    for r in map(json.loads, open(ctxfile)):
        if r["qid"] in qs and r["gold_total"]: ctx[r["qid"]][r["budget"]] = r["gold_in"] / r["gold_total"]
    per_user = collections.defaultdict(lambda: collections.defaultdict(list)); src = collections.Counter(); hist = {}
    for qid, q in qs.items():
        p = os.path.join(pooldir, qid + ".json")
        if not os.path.exists(p) or qid not in ctx: continue
        pool = json.load(open(p)); g = pool["gold"]; rr = {int(k): v for k, v in pool["rr8"].items()}
        order = sorted(pool["fused"], key=lambda i: -rr.get(i, -1))
        u = qid.split("_")[0]; d = per_user[u]
        for B in (4000, 16000):
            d[f"rec{B // 1000}k"].append(ctx[qid].get(B, 0))
            n = NR[B]; ng = sum(g)
            d[f"rec{n}r"].append(sum(1 for i in order[:n] if g[i]) / max(1, ng))
        pos = [rr.get(i, 0) for i in order if g[i]]; neg = [rr.get(i, 0) for i in order if not g[i]]
        a = auc(pos, neg)
        if a is not None: d["auc"].append(a)
        d["goldP"].append(st.median(pos) if pos else 0)
        hist[u] = q.get("hist_tok", 0)
        if "haystack_sessions" in q and q["haystack_sessions"] and "kind" in q["haystack_sessions"][0][0]:
            ks = kinds_of(q); mine = q["gold_stages"]; th = mine[0].split("s")[0]
            tnum = int(th[1:]); twin = f"t{tnum ^ 1}"
            for i in [i for i in order if not g[i]][:10]:
                kd, sg = ks[i] if i < len(ks) else (None, None)
                if kd in ("ev", "fu") and sg:
                    c = ("same-theme " if sg.startswith(th + "s") else ("twin-theme " if sg.startswith(twin + "s") else "other-theme ")) + kd
                else: c = kd or "?"
                src[c] += 1
    print(f"== {label}")
    allv = collections.defaultdict(list)
    for u, d in sorted(per_user.items()):
        print(f"  {u:8s} n={len(d['auc']):3d} hist {hist.get(u, 0) / 1000:6.0f}k | " + " ".join(f"{k} {100 * st.mean(v):5.1f}" for k, v in sorted(d.items()) if k != "goldP" and k != "auc")
              + f" | AUC {st.mean(d['auc']):.3f} goldP {st.median(d['goldP']):.3f}")
        for k, v in d.items(): allv[k] += v
    if allv:
        print(f"  ALL      n={len(allv['auc']):3d}             | " + " ".join(f"{k} {100 * st.mean(v):5.1f}" for k, v in sorted(allv.items()) if k not in ("goldP", "auc"))
              + f" | AUC {st.mean(allv['auc']):.3f} goldP {st.median(allv['goldP']):.3f}")
    if src:
        tot = sum(src.values()); print("  top-10 non-gold sources: " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in src.most_common()))
    return per_user


def paired(a, b, key, n=2000):
    """bootstrap over users present in both arms: mean(b) - mean(a)"""
    us = sorted(set(a) & set(b))
    if not us: return None
    rng = random.Random(0); diffs = []
    for _ in range(n):
        s = [rng.choice(us) for _ in us]
        diffs.append(st.mean([x for u in s for x in b[u][key]]) - st.mean([x for u in s for x in a[u][key]]))
    diffs.sort(); return 100 * st.mean(diffs), 100 * diffs[int(0.025 * n)], 100 * diffs[int(0.975 * n)]


if __name__ == "__main__":
    arms = sys.argv[1].split(",")
    analyse("beam500k_g.json", "ctx_RR8_H.jsonl", "pool", "BEAM-500K reference")
    R = {}
    for n in arms:
        if os.path.exists(f"ctx_RR8_VAR_{n}.jsonl"):
            R[n] = analyse(f"env_var_{n}.json", f"ctx_RR8_VAR_{n}.jsonl", f"pool_var_{n}", f"arm {n}")
    if "V0" in R:
        for n in R:
            if n == "V0": continue
            for k in ("rec4k", "rec16k", "rec4r", "rec16r"):
                r = paired(R["V0"], R[n], k)
                if r: print(f"  {n} - V0 {k}: {r[0]:+.1f} [{r[1]:+.1f}, {r[2]:+.1f}]")
