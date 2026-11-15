"""K-S1 reader: Qwen3.8-27B (vLLM /v1/completions, raw ChatML) with an exact thinking budget.
think=0  -> '<think>\n\n</think>\n\n' then answer (no reasoning instruction, like enable_thinking=false)
think=T  -> generate reasoning up to T tokens (stop at </think>), then close the think block and generate the answer.
usage: python3 runs/s1/run.py --hist S,M --budget 4000,16000 --think 0,2000 --ids runs/s1/ids60.txt --tag t1 --ports 18031,18032 --par 24
"""
import argparse, itertools, json, os, random, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(D))
MODEL = "Qwen3.8-27B-FP8"
EFFORT = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
          "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
SYS = "You are a helpful assistant."
USER = ("I will give you several history chats between you and a user. Please answer the question based on the "
        "relevant chat history.\n\n\nHistory Chats:\n{ctx}\n\nCurrent Date: {date}\nQuestion: {q}\nAnswer:")


def post(port, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            err = e
    raise err


def answer(port, ctx, q, think):
    user = USER.format(ctx=ctx, date=q["question_date"], q=q["question"])
    if think == 0:
        p = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        r = post(port, {"model": MODEL, "prompt": p, "max_tokens": 600, "temperature": 0, "stop": ["<|im_end|>"]})
        return {"reasoning": "", "think_tok": 0, "answer": r["choices"][0]["text"].strip(),
                "in_tok": r["usage"]["prompt_tokens"], "out_tok": r["usage"]["completion_tokens"], "think_cut": False}
    p = f"<|im_start|>system\n{EFFORT}\n\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    r1 = post(port, {"model": MODEL, "prompt": p, "max_tokens": think, "temperature": 0.6, "top_p": 0.95,
                     "stop": ["</think>"], "seed": 0})
    reas = r1["choices"][0]["text"]
    cut = r1["choices"][0]["finish_reason"] == "length"
    p2 = p + reas.rstrip() + "\n</think>\n\n"
    r2 = post(port, {"model": MODEL, "prompt": p2, "max_tokens": 600, "temperature": 0, "stop": ["<|im_end|>"]})
    return {"reasoning": reas[-3000:], "think_tok": r1["usage"]["completion_tokens"], "answer": r2["choices"][0]["text"].strip(),
            "in_tok": r1["usage"]["prompt_tokens"] + r2["usage"]["prompt_tokens"],
            "out_tok": r1["usage"]["completion_tokens"] + r2["usage"]["completion_tokens"], "think_cut": cut}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="S,M"); ap.add_argument("--budget", default="4000,16000,64000,200000")
    ap.add_argument("--think", default="0,2000,8000,24000"); ap.add_argument("--ids", default=os.path.join(D, "ids60.txt"))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", required=True)
    ap.add_argument("--ports", required=True); ap.add_argument("--par", type=int, default=24)
    a = ap.parse_args()
    ids = open(a.ids).read().split()
    if a.limit:
        ids = ids[: a.limit]
    qs = {x["question_id"]: x for x in json.load(open(os.path.join(D, "questions60.json"))) if x["question_id"] in set(ids)}
    ctx = {}
    for h in a.hist.split(","):
        for l in open(os.path.join(D, f"ctx_{h}.jsonl")):
            r = json.loads(l); ctx[(r["qid"], h, r["budget"])] = r
    out = os.path.join(D, "out", f"{a.tag}.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["qid"], r["hist"], r["budget"], r["think"]))
    cells = [(q, h, int(b), int(t)) for q, h, b, t in itertools.product(ids, a.hist.split(","), a.budget.split(","), a.think.split(","))
             if (q, h, int(b), int(t)) not in done]
    random.Random(0).shuffle(cells)   # mix long and short prompts across replicas
    ports = a.ports.split(","); lock = threading.Lock(); f = open(out, "a")

    def one(i_cell):
        i, (qid, h, b, t) = i_cell
        c = ctx[(qid, h, b)]; q = qs[qid]
        res = answer(ports[i % len(ports)], c["ctx"], q, t)
        row = {"qid": qid, "type": q["question_type"], "hist": h, "budget": b, "think": t, "ctx_tok": c["ctx_tok"],
               "full": c["full"], "gold_in": c["gold_in"], "gold_total": c["gold_total"], **res}
        with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()

    with ThreadPoolExecutor(a.par) as ex:
        list(ex.map(one, enumerate(cells)))
    print("answered", a.tag, len(cells))


if __name__ == "__main__":
    main()
