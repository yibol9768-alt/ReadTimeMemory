"""A3 answer diagnostic (11-01): 30 stratified dev questions, same reader, four contexts. Oracle (gold rounds in the pool) and blind (no evidence) also act as judge controls.
Diagnostic only, not a gate. usage: python3 a3.py --port 18301 [--think 2000]; python3 a3.py --sum"""
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
S1 = os.path.expanduser("~/data/rtm/s1"); sys.path.insert(0, S1); sys.path.insert(0, ".")
LT = {"single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning", "knowledge-update", "single-session-preference"}   # other benchmarks use the plain "contains the correct answer" template
OUT = "out/A3.jsonl"; PLAN = {"hard": ("beam_ex.json", "H", "pool", 16000, 10), "other": ("beam_ex.json", "H", "pool", 4000, 8), "lme": ("q_lmes150.json", "lmes150", "pool_lmes150", 2000, 6), "ever": ("q_ev150.json", "ev150", "pool_ev150", 4000, 3), "ep": ("q_ep150.json", "ep150", "pool_ep150", 1000, 3)}
def ctxs(tag, b):
    return {r["qid"]: r for r in map(json.loads, open(f"ctx_{tag}.jsonl")) if r["budget"] == b}
def build():
    from k1 import rounds_of, pack
    hard = set(open("hard_ids.txt").read().split()); rng = random.Random(20261101); items = []
    for st, (qf, n, pd, b, k) in PLAN.items():
        Q = {q["question_id"]: q for q in json.load(open(qf))}; A, W = ctxs(f"A0_{n}", b), ctxs(f"W20_{n}", b)
        ids = sorted(i for i in Q if i in A and i in W and A[i]["gold_total"] and (st not in ("hard", "other") or (i in hard) == (st == "hard")))
        # half where the two selectors differ in recall, half where they tie, so the sample shows both
        gap = lambda i: A[i]["gold_in"] != W[i]["gold_in"]; d = [i for i in ids if gap(i)]; s = [i for i in ids if not gap(i)]
        rng.shuffle(d); rng.shuffle(s); pick = d[: (k + 1) // 2] + s[: k // 2]; pick += [i for i in d + s if i not in pick][: k - len(pick)]
        for i in pick:
            q = Q[i]; R = rounds_of(q); P = json.load(open(os.path.join(pd, i + ".json")))["fused"]
            o = pack(q, R, [x for x in P if R[x][3]], b)
            for tag, c in (("A0", A[i]), ("W20", W[i]), ("oracle", o), ("blind", {"ctx": "", "gold_in": 0, "gold_total": A[i]["gold_total"]})):
                items.append((st, b, tag, q, c))
    return items
def main():
    import run as R, qa, judge
    ap = argparse.ArgumentParser(); ap.add_argument("--port", default="18301"); ap.add_argument("--think", type=int, default=2000); a = ap.parse_args()
    items = build(); done = set()
    if os.path.exists(OUT): done = {(r["qid"], r["ctx"]) for r in map(json.loads, open(OUT)) if "error" not in r}
    items = [x for x in items if (x[3]["question_id"], x[2]) not in done]; f = open(OUT, "a"); print("todo", len(items), flush=True)
    def one(x):
        st, b, tag, q, c = x
        try:
            res = R.answer(a.port, c["ctx"], q, a.think); ans = res["answer"]; rub = q.get("rubric"); rub = qa.lst(rub) if rub else []
            if rub:
                sc = []
                for it in rub:
                    t = qa.nothink(a.port, qa.JP.replace("<rubric_item>", it).replace("<llm_response>", ans), 400)
                    m = re.search(r'"score"\s*:\s*\[?\s*([01](?:\.\d+)?)', t); sc.append(float(m.group(1)) if m else 0.0)
                s = sum(sc) / len(sc)
            else:
                t = judge.chat(a.port, R.MODEL, judge.ev.get_anscheck_prompt(q["question_type"] if q["question_type"] in LT else "multi-session", q["question"], q["answer"], ans, abstention="_abs" in q["question_id"])); s = float("yes" in t.lower()); sc = [s]
            row = dict(stratum=st, qid=q["question_id"], ctx=tag, budget=b, question=q["question"], gold=str(q.get("answer")), answer=ans, score=s, scores=sc, gold_in=c["gold_in"], gold_total=c["gold_total"], think_cut=res["think_cut"])
        except Exception as e:
            row = dict(stratum=st, qid=q["question_id"], ctx=tag, error=repr(e)[:300])
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    with ThreadPoolExecutor(8) as ex: list(ex.map(one, items))
    print("A3DONE", flush=True)
def summ():
    import collections
    rows = [json.loads(l) for l in open(OUT)]; bad = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]; print("rows", len(rows), "errors", len(bad))
    for st in list(PLAN) + ["all"]:
        v = [r for r in rows if st in ("all", r["stratum"])]; o = []
        for t in ("blind", "A0", "W20", "oracle"):
            w = [r for r in v if r["ctx"] == t]
            if w: o.append(f"{t} score {100*sum(r['score'] for r in w)/len(w):.0f} recall {100*sum(r['gold_in']/r['gold_total'] for r in w)/len(w):.0f} n={len(w)}")
        print(f"{st:6s}", " | ".join(o))
    print("think cut", sum(r["think_cut"] for r in rows), "of", len(rows))
if __name__ == "__main__":
    summ() if "--sum" in sys.argv else main()
