"""BEAM QA for the exclusion study (10-31 audit fixes over runs/k1/beam_qa.py).

reader : run.py answer() (Qwen3.8-27B, thinking <= --think), identical for every method
judge  : official BEAM unified prompt per rubric item (0/0.5/1, averaged) for every type (official-style score);
         event_ordering additionally gets the alignment judge (gen.py ALIGN_PROMPT): coverage of rubric steps and
         Kendall tau over the aligned order (our port; not the official ordering metric)
loops over ctx tags x budgets; resumable on (ctx, budget, qid); a failed call writes an error row instead of killing the run.

usage (in ~/data/rtm/sel): python3 qa.py --q beam500k_g.json --ctx RR8_H,BOR_H --budgets 4000,16000 --ids ex_ids.txt \
                             --ports 18301,18302 --out EX_H"""
import argparse, ast, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
S1 = os.path.expanduser("~/data/rtm/s1")
sys.path.insert(0, S1)
import run as R
JP = open(os.path.join(S1, "beam_judge_prompt.txt")).read()
ALIGN = """Below are the true steps of a user's project (numbered) and a RESPONSE that lists or describes steps.
For each item/step mentioned in the RESPONSE, in the order they appear, give the number of the true step it describes
(same event, any wording), or 0 if it matches none. Return only a JSON list of integers, e.g. [1, 3, 0, 2].

TRUE STEPS:
{steps}

RESPONSE:
{resp}"""


def nothink(port, p, n):
    r = R.post(port, {"model": R.MODEL, "prompt": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      "max_tokens": n, "temperature": 0, "stop": ["<|im_end|>"]})
    return r["choices"][0]["text"]


def kendall(seq):
    seq = [x for x in seq if x > 0]
    if len(seq) < 2: return 0.0
    c = d = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            c += seq[i] < seq[j]; d += seq[i] > seq[j]
    return (c - d) / max(1, c + d)


def lst(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True); ap.add_argument("--ctx", required=True); ap.add_argument("--budgets", required=True)
    ap.add_argument("--ports", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ids", default="")
    ap.add_argument("--ctxdir", default="."); ap.add_argument("--think", type=int, default=2000); ap.add_argument("--par", type=int, default=32)
    a = ap.parse_args(); ports = a.ports.split(","); budgets = [int(b) for b in a.budgets.split(",")]
    qs = {q["question_id"]: q for q in json.load(open(a.q))}
    if a.ids: keep = set(open(a.ids).read().split()); qs = {k: v for k, v in qs.items() if k in keep}
    os.makedirs("out", exist_ok=True); out = os.path.join("out", f"{a.out}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["ctx"], r["budget"], r["qid"]) for r in map(json.loads, open(out)) if "error" not in r}
    todo = []
    for tag in a.ctx.split(","):
        ctx = {}
        for r in map(json.loads, open(os.path.join(a.ctxdir, f"ctx_{tag}.jsonl"))):
            if r["budget"] in budgets and r["qid"] in qs: ctx[(r["budget"], r["qid"])] = r
        miss = [(b, q) for b in budgets for q in qs if (b, q) not in ctx]
        if miss: print(f"WARN {tag}: {len(miss)} (budget,qid) missing, e.g. {miss[:3]}", flush=True)
        todo += [(tag, b, q, c) for (b, q), c in ctx.items() if (tag, b, q) not in done]
    print(f"todo {len(todo)}", flush=True)
    lock = threading.Lock(); f = open(out, "a"); n = [0]

    def one(item):
        k, (tag, b, qid, c) = item; q = qs[qid]; port = ports[k % len(ports)]
        try:
            res = R.answer(port, c["ctx"], q, a.think); ans = res["answer"]
            rub = lst(q["rubric"])
            sc = []
            for it in rub:
                t = nothink(port, JP.replace("<rubric_item>", it).replace("<llm_response>", ans), 400)
                m = re.search(r'"score"\s*:\s*\[?\s*([01](?:\.\d+)?)', t); sc.append(float(m.group(1)) if m else 0.0)
            row = {"qid": qid, "ability": q["question_type"], "ctx": tag, "budget": b, "answer": ans, "scores": sc,
                   "score": sum(sc) / max(1, len(sc)), "gold_in": c.get("gold_in"), "gold_total": c.get("gold_total"),
                   "ctx_tok": c.get("ctx_tok"), "in_tok": res["in_tok"], "think_tok": res["think_tok"], "think_cut": res["think_cut"]}
            if q["question_type"] == "event_ordering" and rub:
                steps = "\n".join(f"{j + 1}. {s}" for j, s in enumerate(rub))
                t = nothink(port, ALIGN.format(steps=steps, resp=ans[:6000]), 300)
                m = re.search(r"\[[\d,\s]*\]", t)
                try: seq = [int(x) for x in json.loads(m.group())] if m else []
                except Exception: seq = []
                seq = [x for x in seq if 0 <= x <= len(rub)]
                row.update(align=seq, align_cov=len({x for x in seq if x > 0}) / len(rub), order_tau=kendall(seq))
        except Exception as e:
            row = {"qid": qid, "ctx": tag, "budget": b, "error": repr(e)[:300]}
        with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); n[0] += 1
            if n[0] % 100 == 0: print(f"{n[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, enumerate(todo)))
    print("QA_DONE", a.out, len(todo), flush=True)


if __name__ == "__main__":
    main()
