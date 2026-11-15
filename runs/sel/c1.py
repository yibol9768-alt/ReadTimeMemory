"""Compression reader test (11-01 plan v9): does trimming assistant text before packing raise answer scores, not only recall?
Same W20 order, same reader and judge. Conditions: full round, assistant cut to 300 / 100 / 30 tokens, blind (judge control). LME at 1k and BEAM other at 4k, 60 questions each, seed 20261101.
Frozen read: paired score difference c100 minus full per stratum, 95% bootstrap by question. Recall is reported beside it. usage: python3 c1.py --strata lme,other --port 18305; python3 c1.py --sum"""
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
S1 = os.path.expanduser("~/data/rtm/s1"); sys.path.insert(0, S1); sys.path.insert(0, ".")
LT = {"single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning", "knowledge-update", "single-session-preference"}
OUT = "out/C1.jsonl"; PLAN = {"lme": ("q_lmes150.json", "W20X_lmes150", "pool_lmes150", 1000), "other": ("beam_ex.json", "W20X_H", "pool", 4000)}; CUT = {"full": None, "c300": 300, "c100": 100, "c30": 30}
def trim(text, n):
    m = re.match(r"(User: .*?)\nAssistant: (.*)$", text, re.S)
    if not m or n is None: return text
    c = int(n * 3.6 / 0.78); a = m.group(2)
    return f"{m.group(1)}\nAssistant: {a[:c].rsplit(' ', 1)[0]} ..." if len(a) > c else text
def build(strata):
    from k1 import rounds_of, pack
    hard = set(open("hard_ids.txt").read().split()); items = []
    for st in strata:
        qf, tag, pd, b = PLAN[st]; Q = {q["question_id"]: q for q in json.load(open(qf))}; C = {r["qid"]: r for r in map(json.loads, open(f"ctx_{tag}.jsonl")) if r["budget"] == b}
        ids = sorted(i for i in Q if i in C and C[i]["gold_total"] and (st == "lme" or i not in hard)); random.Random(20261101).shuffle(ids); ids = ids[:60]
        for i in ids:
            q = Q[i]; R0 = rounds_of(q); F = json.load(open(os.path.join(pd, i + ".json")))["fused"]; seen = set(C[i]["rank"]); order = C[i]["rank"] + [x for x in F if x not in seen]
            for name, n in CUT.items():
                R = [(s, d, trim(t, n), g) for s, d, t, g in R0]; items.append((st, b, name, q, pack(q, R, order, b)))
            items.append((st, b, "blind", q, {"ctx": "", "gold_in": 0, "gold_total": C[i]["gold_total"]}))
    return items
def main():
    import run as R, qa, judge
    ap = argparse.ArgumentParser(); ap.add_argument("--port", default="18305"); ap.add_argument("--think", type=int, default=2000); ap.add_argument("--strata", default="lme"); a = ap.parse_args()
    items = build(a.strata.split(",")); done = set()
    if os.path.exists(OUT): done = {(r["qid"], r["ctx"]) for r in map(json.loads, open(OUT)) if "error" not in r}
    items = [x for x in items if (x[3]["question_id"], x[2]) not in done]; f = open(OUT, "a"); print("todo", len(items), flush=True)
    def one(x):
        st, b, tag, q, c = x
        try:
            res = R.answer(a.port, c["ctx"], q, a.think); ans = res["answer"]; rub = q.get("rubric"); rub = qa.lst(rub) if rub else []; sc = []
            if rub:
                for it in rub:
                    t = qa.nothink(a.port, qa.JP.replace("<rubric_item>", it).replace("<llm_response>", ans), 400)
                    m = re.search(r'"score"\s*:\s*\[?\s*([01](?:\.\d+)?)', t); sc.append(float(m.group(1)) if m else 0.0)
                s = sum(sc) / len(sc)
            else:
                t = judge.chat(a.port, R.MODEL, judge.ev.get_anscheck_prompt(q["question_type"] if q["question_type"] in LT else "multi-session", q["question"], q["answer"], ans, abstention="_abs" in q["question_id"]))
                s = float("yes" in t.lower())
            row = dict(stratum=st, qid=q["question_id"], ctx=tag, budget=b, answer=ans, score=s, gold_in=c["gold_in"], gold_total=c["gold_total"], think_cut=res["think_cut"])
        except Exception as e:
            row = dict(stratum=st, qid=q["question_id"], ctx=tag, error=repr(e)[:300])
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    with ThreadPoolExecutor(8) as ex: list(ex.map(one, items))
    print("C1DONE", flush=True)
def summ():
    import statistics as st
    rows = [json.loads(l) for l in open(OUT)]; print("rows", len(rows), "errors", sum("error" in r for r in rows)); rows = [r for r in rows if "error" not in r]
    for s in PLAN:
        v = [r for r in rows if r["stratum"] == s]; by = {}
        if not v: continue
        for r in v: by.setdefault(r["ctx"], {})[r["qid"]] = r
        print(s, " | ".join(f"{k} score {100*st.mean(x['score'] for x in d.values()):.1f} recall {100*st.mean(x['gold_in']/x['gold_total'] for x in d.values()):.1f} n={len(d)}" for k, d in by.items()))
        for k in ("c300", "c100", "c30"):
            if k in by and "full" in by:
                K = sorted(set(by[k]) & set(by["full"])); d = [by[k][q]["score"] - by["full"][q]["score"] for q in K]; r = random.Random(0); bs = sorted(st.mean(r.choices(d, k=len(d))) for _ in range(4000))
                print(f"  {k}-full score {100*st.mean(d):+.1f} [{100*bs[100]:+.1f},{100*bs[3899]:+.1f}]")
if __name__ == "__main__":
    summ() if "--sum" in sys.argv else main()
