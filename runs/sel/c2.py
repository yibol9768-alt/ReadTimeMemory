"""Second reader batch (11-01 15:00, Codex frozen read): c1 cells are reused, this adds W20 and A0 order x full and c100 at LME 1k, 2k, 4k. Read: paired score, compression within an order, W20 minus A0 under each representation, and the interaction. usage: python3 c2.py --port 18300; python3 c2.py --sum
Derived from c1: Compression reader test (11-01 plan v9): does trimming assistant text before packing raise answer scores, not only recall?
Same W20 order, same reader and judge. Conditions: full round, assistant cut to 300 / 100 / 30 tokens, blind (judge control). LME at 1k and BEAM other at 4k, 60 questions each, seed 20261101.
Frozen read: paired score difference c100 minus full per stratum, 95% bootstrap by question. Recall is reported beside it. usage: python3 c1.py --strata lme,other --port 18305; python3 c1.py --sum"""
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
S1 = os.path.expanduser("~/data/rtm/s1"); sys.path.insert(0, S1); sys.path.insert(0, ".")
LT = {"single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning", "knowledge-update", "single-session-preference"}
OUT = "out/C2.jsonl"; CELLS = [("W20X_lmes150", "W20", 2000), ("W20X_lmes150", "W20", 4000), ("A0X_lmes150", "A0", 1000), ("A0X_lmes150", "A0", 2000)]; CUT = {"full": None, "c100": 100}
def trim(text, n):
    m = re.match(r"(User: .*?)\nAssistant: (.*)$", text, re.S)
    if not m or n is None: return text
    c = int(n * 3.6 / 0.78); a = m.group(2)
    return f"{m.group(1)}\nAssistant: {a[:c].rsplit(' ', 1)[0]} ..." if len(a) > c else text
def build():
    from k1 import rounds_of, pack
    items = []; Q = {q["question_id"]: q for q in json.load(open("q_lmes150.json"))}
    C0 = {r["qid"]: r for r in map(json.loads, open("ctx_W20X_lmes150.jsonl")) if r["budget"] == 1000}
    ids = sorted(i for i in Q if i in C0 and C0[i]["gold_total"]); random.Random(20261101).shuffle(ids); ids = ids[:60]
    for tag, nm, b in CELLS:
        C = {r["qid"]: r for r in map(json.loads, open(f"ctx_{tag}.jsonl")) if r["budget"] == b}
        for i in ids:
            q = Q[i]; R0 = rounds_of(q); F = json.load(open(os.path.join("pool_lmes150", i + ".json")))["fused"]; seen = set(C[i]["rank"]); order = C[i]["rank"] + [x for x in F if x not in seen]
            for name, n in CUT.items():
                R = [(s, d, trim(t, n), g) for s, d, t, g in R0]; items.append((f"{nm}_{b}", b, name, q, pack(q, R, order, b)))
    return items
def main():
    import run as R, qa, judge
    ap = argparse.ArgumentParser(); ap.add_argument("--port", default="18305"); ap.add_argument("--think", type=int, default=2000); a = ap.parse_args()
    items = build(); done = set()
    if os.path.exists(OUT): done = {(r["stratum"], r["qid"], r["ctx"]) for r in map(json.loads, open(OUT)) if "error" not in r}
    items = [x for x in items if (x[0], x[3]["question_id"], x[2]) not in done]; f = open(OUT, "a"); print("todo", len(items), flush=True)
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
    print("C2DONE", flush=True)
def summ():
    import statistics as st
    rows = [json.loads(l) for l in open(OUT)] + [dict(r, stratum="W20_1000") for r in map(json.loads, open("out/C1.jsonl")) if r.get("stratum") == "lme" and r.get("ctx") in ("full", "c100")]
    print("rows", len(rows), "errors", sum("error" in r for r in rows)); rows = [r for r in rows if "error" not in r]
    S = {}
    for r in rows: S.setdefault((r["stratum"], r["ctx"]), {})[r["qid"]] = r
    def ci(d):
        r = random.Random(0); bs = sorted(st.mean(r.choices(d, k=len(d))) for _ in range(4000)); return f"{100*st.mean(d):+.1f} [{100*bs[100]:+.1f},{100*bs[3899]:+.1f}] n={len(d)}"
    for k, d in sorted(S.items()): print(k, f"score {100*st.mean(x['score'] for x in d.values()):.1f} recall {100*st.mean(x['gold_in']/x['gold_total'] for x in d.values()):.1f} n={len(d)} think_cut {sum(x['think_cut'] for x in d.values() if 'think_cut' in x)}")
    g = lambda s, c, q: S[(s, c)][q]["score"]
    for b in (1000, 2000, 4000):
        w, a = f"W20_{b}", f"A0_{b}"
        if (w, "full") in S and (w, "c100") in S:
            K = sorted(set(S[(w, "full")]) & set(S[(w, "c100")])); print(b, "compression W20", ci([g(w, "c100", q) - g(w, "full", q) for q in K]))
        if (a, "full") in S and (a, "c100") in S and (w, "full") in S:
            K = sorted(set(S[(a, "full")]) & set(S[(a, "c100")]) & set(S[(w, "full")]) & set(S[(w, "c100")]))
            print(b, "compression A0 ", ci([g(a, "c100", q) - g(a, "full", q) for q in K])); print(b, "W20-A0 full   ", ci([g(w, "full", q) - g(a, "full", q) for q in K])); print(b, "W20-A0 c100   ", ci([g(w, "c100", q) - g(a, "c100", q) for q in K]))
            print(b, "interaction   ", ci([(g(w, "c100", q) - g(w, "full", q)) - (g(a, "c100", q) - g(a, "full", q)) for q in K]))
if __name__ == "__main__":
    summ() if "--sum" in sys.argv else main()
