"""Step 3 (PLAN_NEXT v2, 11-02): answer-score table, rerun after the 11-01 batch stopped at 435/5750.
Same 27B reader, judge, prompts and frozen samples (seed 20261101) as a5.py. Rankings come from the step-1 reruns
(P1 tags, one pointwise ranking per model with the same pool, packing and budgets). Methods: rag (fused pool order),
A0 (Qwen3-Reranker-0.6B), RR8 (Qwen3-Reranker-8B), MR4 (MemReranker-4B), T1 (BEAM-label reranker), CX (CruxRank stage 1,
seed 0) and the best generic cross-encoder if given with --extra TAG. Full context runs separately (--full), last, with
few workers and a per-item timeout, so it can never block the table again.
usage: python3 a6.py --ports P1,P2,...;  python3 a6.py --full --ports ... --workers 4;  python3 a6.py --sum [--extra BGEM3]"""
import argparse, collections, json, os, random, re, statistics as st, sys, threading
from concurrent.futures import ThreadPoolExecutor
S1 = os.path.expanduser("~/data/rtm/s1"); sys.path.insert(0, S1); sys.path.insert(0, ".")
LT = {"single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning", "knowledge-update", "single-session-preference"}
OUT = "out/A6.jsonl"; OUTF = "out/A6F.jsonl"
PLAN = {"lme": ("q_lmes150.json", "lmes150", "pool_lmes150", 2000, 150), "ep": ("q_ep150.json", "ep150", "pool_ep150", 1000, 150), "ev": ("q_ev150.json", "ev150", "pool_ev150", 4000, 150), "loc": ("q_loc150.json", "loc150", "pool_loc150", 2000, 150), "hard": ("beam_ex.json", "H", "pool", 16000, 100), "other": ("beam_ex.json", "H", "pool", 4000, 100)}
TAG = {"A0": "P1A0", "RR8": "P1RR8", "MR4": "P1MR4", "T1": "P1T1", "CX": "P1E0"}
def ctx(t, b):
    p = f"ctx_{t}.jsonl"
    return {r["qid"]: r for r in map(json.loads, open(p)) if r["budget"] == b} if os.path.exists(p) else {}
def rrf(a, f):
    s = {}
    for l in (a, f):
        for k, i in enumerate(l): s[i] = s.get(i, 0) + 1 / (61 + k)
    return sorted(s, key=lambda i: (-s[i], i))
IDS = "out/A6_ids.json"; KS = {}   # frozen (stratum -> qids) on first build; every later run reuses it
def build(full=False, kmax=0, only=None):
    from k1 import rounds_of, pack, ntok
    hard = set(open("hard_ids.txt").read().split()); items = []; over = collections.Counter()
    frozen = json.load(open(IDS)) if os.path.exists(IDS) else None; new = {}
    for st_, (qf, n, pd, b, k) in PLAN.items():
        if only and st_ not in only: continue
        Q = {q["question_id"]: q for q in json.load(open(qf))}; W = ctx(f"P1E0_{n}", b)
        ids = sorted(i for i in Q if i in W and W[i]["gold_total"] and (st_ not in ("hard", "other") or (i in hard) == (st_ == "hard"))); random.Random(20261101).shuffle(ids); ids = ids[:k]
        if frozen is not None:
            if set(frozen[st_]) - set(Q): raise SystemExit(f"INVALID frozen ids for {st_} not in question file")
            ids = frozen[st_]
        new[st_] = ids
        if kmax: ids = ids[:kmax]
        KS[st_] = len(ids)   # smoke: prefix of the frozen sample
        C = {m: ctx(f"{t}_{n}", b) for m, t in TAG.items()}
        for i in ids:
            q = Q[i]; R = rounds_of(q); F = json.load(open(os.path.join(pd, i + ".json")))["fused"]; gt = W[i]["gold_total"]
            if not full:
                items.append((st_, "rag", q, pack(q, R, F, b)))
                for m in TAG:
                    if i in C[m]: items.append((st_, m, q, C[m][i]))
            elif st_ in ("lme", "loc"):
                last, parts = None, []
                for si, date, text, _ in R:
                    if si != last: parts.append(f"\n### Session date: {date}"); last = si
                    parts.append(text)
                t = "\n".join(parts)
                if ntok(t) <= 125000: items.append((st_, "full", q, {"ctx": t, "gold_in": gt, "gold_total": gt}))   # room for the prompt, 2000 thinking and 600 answer tokens inside 131072
                else: over[st_] += 1
    if frozen is None and not only: json.dump(new, open(IDS, "w")); print("froze sample ids to", IDS, flush=True)
    print("full skipped as too long", dict(over), flush=True)
    C_ = collections.Counter((x[0], x[1]) for x in items)
    if not full:
        for s_ in [x for x in PLAN if not only or x in only]:
            for m in ("rag",) + tuple(TAG):
                if C_[(s_, m)] != KS[s_]: raise SystemExit(f"INVALID {s_} {m}: {C_[(s_, m)]} of {KS[s_]}")
    return items
def main(a):
    import run as R, qa, judge
    for x in filter(None, a.extra.split(",")): TAG[x] = "P1" + x
    for m in filter(None, a.skip.split(",")): TAG.pop(m)
    if a.full and a.workers > 8: a.workers = 4; print("full context: workers capped at 4", flush=True)
    out = a.out or (OUTF if a.full else OUT); only = set(a.strata.split(",")) if a.strata else None
    items = build(a.full, a.k, only); done = set()   # disjoint strata per instance, separate --out files
    if os.path.exists(out): done = {(r["stratum"], r["qid"], r["m"]) for r in map(json.loads, open(out)) if "error" not in r}
    items = [x for x in items if (x[0], x[2]["question_id"], x[1]) not in done]; f = open(out, "a"); P = a.ports.split(","); n = [0]; lk = threading.Lock(); print("todo", len(items), flush=True)
    def one(x):
        st_, m, q, c = x
        with lk: n[0] += 1; port = P[n[0] % len(P)]
        try:
            res = R.answer(port, c["ctx"], q, a.think); ans = res["answer"]   # run.py's own request timeout (1800 s) bounds each call
            rub = q.get("rubric"); rub = qa.lst(rub) if rub else []
            if not a.self_judge: s = None   # default: independent judge (a6judge.py) grades later
            elif rub:
                sc = []
                for it in rub:
                    t = qa.nothink(port, qa.JP.replace("<rubric_item>", it).replace("<llm_response>", ans), 400)
                    mm = re.search(r'"score"\s*:\s*\[?\s*([01](?:\.\d+)?)', t); sc.append(float(mm.group(1)) if mm else 0.0)
                s = sum(sc) / len(sc)
            else:
                t = judge.chat(port, R.MODEL, judge.ev.get_anscheck_prompt(q["question_type"] if q["question_type"] in LT else "multi-session", q["question"], q["answer"], ans, abstention="_abs" in q["question_id"])); s = float("yes" in t.lower())
            row = dict(stratum=st_, m=m, qid=q["question_id"], score=s, gold_in=c["gold_in"], gold_total=c["gold_total"], think_cut=res["think_cut"], answer=ans, reader=R.MODEL, think=a.think)
        except Exception as e:
            row = dict(stratum=st_, m=m, qid=q["question_id"], error=repr(e)[:300])
        with lk: f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, items))
    print("A5DONE", flush=True)
def summ():
    if a_.extra: TAG[a_.extra] = "P1" + a_.extra
    for m in filter(None, a_.skip.split(",")): TAG.pop(m)
    raise SystemExit("a6.py --sum is retired: answers are graded by the independent judge; use python3 a6judge.py --sum")
    rows = [json.loads(l) for l in open(OUT)] + ([json.loads(l) for l in open(OUTF)] if os.path.exists(OUTF) else []); print("rows", len(rows), "errors", sum("error" in r for r in rows)); rows = [r for r in rows if "error" not in r]; T = collections.defaultdict(dict)
    for r in rows: T[(r["stratum"], r["m"])][r["qid"]] = r
    M = ["rag"] + list(TAG) + ["full"]
    for s in PLAN:
        print(f"{s:6s}", " ".join(f"{m} {100*st.mean(x['score'] for x in T[(s,m)].values()):.1f}/{100*st.mean(x['gold_in']/x['gold_total'] for x in T[(s,m)].values()):.0f}({len(T[(s,m)])})" for m in M if (s, m) in T), "| think cut", sum(x["think_cut"] for m in M for x in T.get((s, m), {}).values()))
    def ci(a, b, s):
        K = sorted(set(T[(s, a)]) & set(T[(s, b)])); by = {}
        for q in K: by.setdefault(q.split("_")[0] if s in ("hard", "other") else q, []).append(T[(s, a)][q]["score"] - T[(s, b)][q]["score"])   # BEAM: resample conversations
        cl = list(by); r = random.Random(0); bs = sorted(st.mean([x for c in r.choices(cl, k=len(cl)) for x in by[c]]) for _ in range(4000))
        return f"{100*st.mean(x for c in cl for x in by[c]):+.1f} [{100*bs[100]:+.1f},{100*bs[3899]:+.1f}] n={len(K)}"
    for a, b in (("CX", "A0"), ("CX", "RR8"), ("CX", "MR4"), ("CX", "T1"), ("CX", "rag"), ("RR8", "A0"), ("CX", "full")) + tuple((("CX", x) for x in TAG if x not in ("A0", "RR8", "MR4", "T1", "CX"))):
        print(f"{a}-{b}", " ".join(f"{s} {ci(a, b, s)}" for s in PLAN if (s, a) in T and (s, b) in T))
    W = {"lme": 1, "ep": 1, "ev": 1, "loc": 1, "hard": .5, "other": .5}
    for name, sets, ms in (("all five benchmarks", list(PLAN), ["rag"] + list(TAG)),):
        out = {m: 0 for m in ms}; miss = 0
        if not all(any((s, m) in T for m in ms) for s in sets) or any((s, m) not in T for s in sets for m in ms): print("same-question mean,", name, "incomplete"); continue
        for s in sets:
            K = set.intersection(*[set(T[(s, m)]) for m in ms]); miss += max(len(T[(s, m)]) for m in ms) - len(K)
            for m in ms: out[m] += W[s] * st.mean(T[(s, m)][q]["score"] for q in K)
        print("same-question mean,", name, {m: round(100 * v / sum(W[s] for s in sets), 1) for m, v in out.items()}, "questions dropped", miss)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ports", default="18300"); ap.add_argument("--think", type=int, default=2000); ap.add_argument("--workers", type=int, default=56); ap.add_argument("--full", action="store_true"); ap.add_argument("--sum", action="store_true"); ap.add_argument("--extra", default=""); ap.add_argument("--skip", default="", help="methods left out for now (e.g. RR8), added back later"); ap.add_argument("--self-judge", action="store_true", help="old behaviour: the reader grades itself (not for the paper table)"); ap.add_argument("--strata", default="", help="only these strata (e.g. lme,loc); run instances on disjoint strata with different --out"); ap.add_argument("--out", default=""); ap.add_argument("--k", type=int, default=0, help="smoke: first k frozen questions per stratum"); a = ap.parse_args()
    a_ = a; summ() if a.sum else main(a)
