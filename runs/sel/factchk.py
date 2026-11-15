"""Rubric-free answer check (10-31, audit item 2): how much of a reader's answer is true and on-topic, ignoring the rubric?
Each answer is split into items; 27B marks each as supported by the context it was given and on-topic for the question.
Context comes from the shared user history, so a supported item is a true statement about that history.
usage: python3 factchk.py --q Q --qa out/EX_H.jsonl --ctxs RR8_H,BOR_H --budgets 4000,16000 --ids IDS --ports P1,P2 --out FC [--limit N]"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.expanduser("~/data/rtm/s1"))
import run as R

P = """A user asked a question about their own past conversations. Below are excerpts of those conversations, the question, and a response.
Split the RESPONSE into its distinct factual items (events, decisions, results, steps). For each item decide:
 s: 1 if the EXCERPTS state or directly imply it, otherwise 0
 r: 1 if it is relevant to what the QUESTION asks, otherwise 0
Return only JSON: {{"items":[{{"t":"at most 8 words","s":1,"r":1}}]}}

EXCERPTS:
{ctx}

QUESTION: {q}

RESPONSE:
{a}"""


def nothink(port, p, n):
    r = R.post(port, {"model": R.MODEL, "prompt": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      "max_tokens": n, "temperature": 0, "stop": ["<|im_end|>"]})
    return r["choices"][0]["text"]


ap = argparse.ArgumentParser()
for x in ("--q", "--qa", "--ctxs", "--budgets", "--ids", "--ports", "--out"): ap.add_argument(x, required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--par", type=int, default=24); a = ap.parse_args()
keep = set(open(a.ids).read().split()); Q = {q["question_id"]: q for q in json.load(open(a.q)) if q["question_id"] in keep}
tags = a.ctxs.split(","); B = {int(x) for x in a.budgets.split(",")}; ports = a.ports.split(",")
C = {}
for t in tags:
    for r in map(json.loads, open(f"ctx_{t}.jsonl")):
        if r["qid"] in Q and int(r["budget"]) in B: C[(t, int(r["budget"]), r["qid"])] = r["ctx"]
rows = [r for r in map(json.loads, open(a.qa)) if r.get("ctx") in tags and r.get("budget") in B and r.get("qid") in Q and "error" not in r and (r["ctx"], r["budget"], r["qid"]) in C]
if a.limit: rows = rows[: a.limit]
out = open(f"out/{a.out}.jsonl", "a"); lock = threading.Lock(); print("todo", len(rows), flush=True)


def one(x):
    k, r = x; key = (r["ctx"], r["budget"], r["qid"])
    try:
        t = nothink(ports[k % len(ports)], P.format(ctx=C[key], q=Q[r["qid"]]["question"], a=r["answer"][:6000]), 2500)
        m = re.search(r"\{.*\}", t, re.S); it = json.loads(m.group())["items"]
        o = {"qid": r["qid"], "ctx": r["ctx"], "budget": r["budget"], "type": r["ability"], "score": r["score"], "n": len(it),
             "s": sum(int(i.get("s", 0)) for i in it), "r": sum(int(i.get("r", 0)) for i in it),
             "sr": sum(int(i.get("s", 0)) * int(i.get("r", 0)) for i in it), "items": it}
    except Exception as e:
        o = {"qid": r["qid"], "ctx": r["ctx"], "budget": r["budget"], "error": repr(e)[:200]}
    with lock: out.write(json.dumps(o, ensure_ascii=False) + "\n"); out.flush()


with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, enumerate(rows)))
print("FC_DONE", flush=True)
