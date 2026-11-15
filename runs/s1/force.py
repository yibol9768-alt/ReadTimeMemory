"""Premise test: budget forcing (s1-style). Same prompts as run.py; when the model ends its thinking before MIN tokens,
append 'Wait,' and continue, up to MAX thinking tokens. Rows match run.py (judge.py reads them).
usage: QFILE=questions_all.json python3 force.py --hist MOA --budget 4000 --min 3000 --max 8000 --tag MOA_force --ports P,..."""
import argparse, json, os, random, threading
from concurrent.futures import ThreadPoolExecutor
import run as R

def answer_forced(port, ctx, q, mn, mx):
    user = R.USER.format(ctx=ctx, date=q["question_date"], q=q["question"])
    p = f"<|im_start|>system\n{R.EFFORT}\n\n{R.SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    reas, used, waits, intok = "", 0, 0, 0
    while used < mx:
        r = R.post(port, {"model": R.MODEL, "prompt": p + reas, "max_tokens": mx - used, "temperature": 0.6, "top_p": 0.95,
                          "stop": ["</think>"], "seed": 0})
        reas += r["choices"][0]["text"]; used += r["usage"]["completion_tokens"]; intok += r["usage"]["prompt_tokens"]
        if r["choices"][0]["finish_reason"] == "length" or used >= mn:
            break
        reas = reas.rstrip() + "\nWait,"; waits += 1
    p2 = p + reas.rstrip() + "\n</think>\n\n"
    r2 = R.post(port, {"model": R.MODEL, "prompt": p2, "max_tokens": 600, "temperature": 0, "stop": ["<|im_end|>"]})
    return {"reasoning": reas[-3000:], "think_tok": used, "waits": waits, "answer": r2["choices"][0]["text"].strip(),
            "in_tok": intok + r2["usage"]["prompt_tokens"], "out_tok": used + r2["usage"]["completion_tokens"], "think_cut": used >= mx}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True); ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--min", type=int, default=3000); ap.add_argument("--max", type=int, default=8000)
    ap.add_argument("--tag", required=True); ap.add_argument("--ports", required=True); ap.add_argument("--par", type=int, default=48)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    qs = {x["question_id"]: x for x in json.load(open(os.path.join(R.D, os.environ.get("QFILE", "questions60.json"))))}
    ctx = {r["qid"]: r for r in map(json.loads, open(os.path.join(R.D, f"ctx_{a.hist}.jsonl"))) if r["budget"] == a.budget}
    ids = open(a.ids).read().split() if a.ids else sorted(ctx)
    out = os.path.join(R.D, "out", f"{a.tag}.jsonl")
    done = {json.loads(l)["qid"] for l in open(out)} if os.path.exists(out) else set()
    todo = [i for i in ids if i not in done and i in ctx]; random.Random(0).shuffle(todo)
    ports = a.ports.split(","); lock = threading.Lock(); f = open(out, "a")
    def one(iq):
        i, qid = iq; c = ctx[qid]; q = qs[qid]
        res = answer_forced(ports[i % len(ports)], c["ctx"], q, a.min, a.max)
        row = {"qid": qid, "type": q["question_type"], "hist": a.hist, "budget": a.budget, "think": a.min, "ctx_tok": c["ctx_tok"],
               "full": c["full"], "gold_in": c["gold_in"], "gold_total": c["gold_total"], **res}
        with lock: f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, enumerate(todo)))
    print("forced", a.tag, len(todo))

if __name__ == "__main__":
    main()
