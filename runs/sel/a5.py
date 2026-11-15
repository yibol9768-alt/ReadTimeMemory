"""Answer-score table (11-01, batch three): same 27B reader and judge as A3, every benchmark, frozen samples (seed 20261101). Methods: rag (fused pool order, the plain RAG baseline), A0O, A0, W20, FIX (morning fixed seed 0), E (morning E seed 0), FUS (RRF k=60 of A0 and FIX), full (whole history, LME and LoCoMo only).
Then the adaptive packing test: W20 order, full rounds first, then the skipped rounds cut to 100 tokens fill what is left. Development data only. usage: python3 a5.py --ports 18300,18301; python3 a5.py --adaptive --ports ...; python3 a5.py --sum"""
import argparse, collections, json, os, random, re, statistics as st, sys, threading
from concurrent.futures import ThreadPoolExecutor
S1 = os.path.expanduser("~/data/rtm/s1"); sys.path.insert(0, S1); sys.path.insert(0, ".")
LT = {"single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning", "knowledge-update", "single-session-preference"}
OUT = "out/A5.jsonl"; OUT2 = "out/A5B.jsonl"
PLAN = {"lme": ("q_lmes150.json", "lmes150", "pool_lmes150", 2000, 150), "ep": ("q_ep150.json", "ep150", "pool_ep150", 1000, 150), "ev": ("q_ev150.json", "ev150", "pool_ev150", 4000, 150), "loc": ("q_loc150.json", "loc150", "pool_loc150", 2000, 150), "hard": ("beam_ex.json", "H", "pool", 16000, 100), "other": ("beam_ex.json", "H", "pool", 4000, 100)}
TAG = {"A0O": "A0O", "A0": "A0X", "W20": "W20X", "FIX": "D3RFX", "E": "D3REX"}
def ctx(t, b):
    p = f"ctx_{t}.jsonl"
    return {r["qid"]: r for r in map(json.loads, open(p)) if r["budget"] == b} if os.path.exists(p) else {}
def rrf(a, f):
    s = {}
    for l in (a, f):
        for k, i in enumerate(l): s[i] = s.get(i, 0) + 1 / (61 + k)
    return sorted(s, key=lambda i: (-s[i], i))
def build():
    from k1 import rounds_of, pack, ntok
    hard = set(open("hard_ids.txt").read().split()); items = []; over = collections.Counter()
    for st_, (qf, n, pd, b, k) in PLAN.items():
        Q = {q["question_id"]: q for q in json.load(open(qf))}; W = ctx(f"W20X_{n}", b)
        ids = sorted(i for i in Q if i in W and W[i]["gold_total"] and (st_ not in ("hard", "other") or (i in hard) == (st_ == "hard"))); random.Random(20261101).shuffle(ids); ids = ids[:k]
        C = {m: ctx(f"{t}_{n}", b) for m, t in TAG.items()}; fx = ctx(f"D3RFR_{n}", b) if n in ("ep150", "ev150") else C["FIX"]
        for i in ids:
            q = Q[i]; R = rounds_of(q); F = json.load(open(os.path.join(pd, i + ".json")))["fused"]; gt = W[i]["gold_total"]
            items.append((st_, "rag", q, pack(q, R, F, b)))
            for m in TAG:
                if i in C[m]: items.append((st_, m, q, C[m][i]))
            a, f = C["A0"].get(i, {}).get("rank"), fx.get(i, {}).get("rank")
            if a and f:
                o = rrf(a, f); seen = set(o); items.append((st_, "FUS", q, pack(q, R, o + [x for x in F if x not in seen], b)))
            if st_ in ("lme", "loc"):
                last, parts = None, []
                for si, date, text, _ in R:
                    if si != last: parts.append(f"\n### Session date: {date}"); last = si
                    parts.append(text)
                t = "\n".join(parts)
                if ntok(t) <= 125000: items.append((st_, "full", q, {"ctx": t, "gold_in": gt, "gold_total": gt}))   # room for the prompt, 2000 thinking and 600 answer tokens inside 131072
                else: over[st_] += 1
    print("full skipped as too long", dict(over), flush=True)
    C_ = collections.Counter((x[0], x[1]) for x in items)
    for s_, (qf, n, pd, b, k) in PLAN.items():
        for m in ("rag", "A0", "W20", "FIX", "E", "FUS") + (("A0O",) if s_ != "loc" else ()):
            if C_[(s_, m)] != k: raise SystemExit(f"INVALID {s_} {m}: {C_[(s_, m)]} of {k}")
    return items
def adaptive():
    from k1 import rounds_of, ntok
    from c1 import trim
    items = []
    for st_, qf, tag, pd, bs in (("lme", "q_lmes150.json", "W20X_lmes150", "pool_lmes150", (1000, 2000, 4000)), ("other", "beam_ex.json", "W20X_H", "pool", (4000,))):
        hard = set(open("hard_ids.txt").read().split()); Q = {q["question_id"]: q for q in json.load(open(qf))}; C0 = ctx(tag, bs[0])
        ids = sorted(i for i in Q if i in C0 and C0[i]["gold_total"] and (st_ == "lme" or i not in hard)); random.Random(20261101).shuffle(ids); ids = ids[:60]
        for b in bs:
            C = ctx(tag, b)
            for i in ids:
                q = Q[i]; R = rounds_of(q); F = json.load(open(os.path.join(pd, i + ".json")))["fused"]; seen = set(C[i]["rank"]); order = C[i]["rank"] + [x for x in F if x not in seen]; used = 0; keep = {}
                for r in order:
                    n = ntok(R[r][2])
                    if used + n <= b: keep[r] = R[r][2]; used += n
                for r in order:
                    if r in keep: continue
                    t = trim(R[r][2], 100); n = ntok(t)
                    if used + n <= b: keep[r] = t; used += n
                parts, last = [], None
                for r in sorted(keep):
                    si, date = R[r][0], R[r][1]
                    if si != last: parts.append(f"\n### Session date: {date}"); last = si
                    parts.append(keep[r])
                items.append((f"{st_}_{b}", "adapt", q, {"ctx": "\n".join(parts), "gold_in": sum(R[r][3] for r in keep), "gold_total": sum(r[3] for r in R)}))
    return items
def main(a):
    import run as R, qa, judge
    out = OUT2 if a.adaptive else OUT; items = adaptive() if a.adaptive else build(); done = set()
    if os.path.exists(out): done = {(r["stratum"], r["qid"], r["m"]) for r in map(json.loads, open(out)) if "error" not in r}
    items = [x for x in items if (x[0], x[2]["question_id"], x[1]) not in done]; f = open(out, "a"); P = a.ports.split(","); n = [0]; lk = threading.Lock(); print("todo", len(items), flush=True)
    def one(x):
        st_, m, q, c = x
        with lk: n[0] += 1; port = P[n[0] % len(P)]
        try:
            res = R.answer(port, c["ctx"], q, a.think); ans = res["answer"]; rub = q.get("rubric"); rub = qa.lst(rub) if rub else []
            if rub:
                sc = []
                for it in rub:
                    t = qa.nothink(port, qa.JP.replace("<rubric_item>", it).replace("<llm_response>", ans), 400)
                    mm = re.search(r'"score"\s*:\s*\[?\s*([01](?:\.\d+)?)', t); sc.append(float(mm.group(1)) if mm else 0.0)
                s = sum(sc) / len(sc)
            else:
                t = judge.chat(port, R.MODEL, judge.ev.get_anscheck_prompt(q["question_type"] if q["question_type"] in LT else "multi-session", q["question"], q["answer"], ans, abstention="_abs" in q["question_id"])); s = float("yes" in t.lower())
            row = dict(stratum=st_, m=m, qid=q["question_id"], score=s, gold_in=c["gold_in"], gold_total=c["gold_total"], think_cut=res["think_cut"], answer=ans[:400])
        except Exception as e:
            row = dict(stratum=st_, m=m, qid=q["question_id"], error=repr(e)[:300])
        with lk: f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, items))
    print("A5DONE", flush=True)
def summ():
    rows = [json.loads(l) for l in open(OUT)]; print("rows", len(rows), "errors", sum("error" in r for r in rows)); rows = [r for r in rows if "error" not in r]; T = collections.defaultdict(dict)
    for r in rows: T[(r["stratum"], r["m"])][r["qid"]] = r
    M = ["rag", "A0O", "A0", "W20", "FIX", "E", "FUS", "full"]
    for s in PLAN:
        print(f"{s:6s}", " ".join(f"{m} {100*st.mean(x['score'] for x in T[(s,m)].values()):.1f}/{100*st.mean(x['gold_in']/x['gold_total'] for x in T[(s,m)].values()):.0f}({len(T[(s,m)])})" for m in M if (s, m) in T), "| think cut", sum(x["think_cut"] for m in M for x in T.get((s, m), {}).values()))
    def ci(a, b, s):
        K = sorted(set(T[(s, a)]) & set(T[(s, b)])); d = [T[(s, a)][q]["score"] - T[(s, b)][q]["score"] for q in K]; r = random.Random(0); bs = sorted(st.mean(r.choices(d, k=len(d))) for _ in range(4000)); return f"{100*st.mean(d):+.1f} [{100*bs[100]:+.1f},{100*bs[3899]:+.1f}]"
    for a, b in (("W20", "A0"), ("FIX", "A0"), ("E", "A0"), ("FUS", "A0"), ("W20", "rag"), ("FIX", "rag"), ("E", "rag"), ("FUS", "rag"), ("W20", "A0O"), ("E", "A0O")):
        print(f"{a}-{b}", " ".join(f"{s} {ci(a, b, s)}" for s in PLAN if (s, a) in T and (s, b) in T))
    W = {"lme": 1, "ep": 1, "ev": 1, "loc": 1, "hard": .5, "other": .5}
    for name, sets, ms in (("all five benchmarks", list(PLAN), ["rag", "A0", "W20", "FIX", "E", "FUS"]), ("four without LoCoMo, with A0O", [s for s in PLAN if s != "loc"], ["rag", "A0O", "A0", "W20", "FIX", "E", "FUS"])):
        out = {m: 0 for m in ms}; miss = 0
        if not all(any((s, m) in T for m in ms) for s in sets) or any((s, m) not in T for s in sets for m in ms): print("same-question mean,", name, "incomplete"); continue
        for s in sets:
            K = set.intersection(*[set(T[(s, m)]) for m in ms]); miss += max(len(T[(s, m)]) for m in ms) - len(K)
            for m in ms: out[m] += W[s] * st.mean(T[(s, m)][q]["score"] for q in K)
        print("same-question mean,", name, {m: round(100 * v / sum(W[s] for s in sets), 1) for m, v in out.items()}, "questions dropped", miss)
    if os.path.exists(OUT2):
        B = collections.defaultdict(dict)
        for r in map(json.loads, open(OUT2)):
            if "error" not in r: B[r["stratum"]][r["qid"]] = r
        print("adaptive answer score", {k: (round(100 * st.mean(x["score"] for x in v.values()), 1), len(v)) for k, v in sorted(B.items())})
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ports", default="18300"); ap.add_argument("--think", type=int, default=2000); ap.add_argument("--workers", type=int, default=56); ap.add_argument("--adaptive", action="store_true"); ap.add_argument("--sum", action="store_true"); a = ap.parse_args()
    summ() if a.sum else main(a)
