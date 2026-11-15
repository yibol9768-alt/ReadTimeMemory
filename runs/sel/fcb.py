"""second step of factchk: strict editor relevance (prompt B, chosen on the swapped-answer control) over out/FC16.jsonl -> out/FC16B.jsonl"""
import json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.expanduser("~/data/rtm/s1"))
import run as R
Q = {q["question_id"]: q for q in json.load(open("beam500k_g.json"))}
P = """QUESTION: {q}
A correct answer to this question would be restricted to one narrow topic. An answer often wrongly includes neighbouring material. Be strict: when unsure, mark 0.
For each ITEM give 1 if a careful editor would keep it in the answer, 0 if the editor would cut it as off-topic.
ITEMS:
{it}
Return only JSON: {{"marks":[0,1,...]}}"""
rows = [r for r in map(json.loads, open("out/FC16.jsonl")) if "error" not in r]
f = open("out/FC16B.jsonl", "w"); lock = threading.Lock()
def one(a):
    k, r = a
    it = "\n".join(f"{i + 1}. {x['t']}" for i, x in enumerate(r["items"]))
    p = P.format(q=Q[r["qid"]]["question"], it=it)
    try:
        t = R.post(["18307", "18308"][k % 2], {"model": R.MODEL, "prompt": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "max_tokens": 1500, "temperature": 0, "stop": ["<|im_end|>"]})["choices"][0]["text"]
        m = [int(x) for x in json.loads(re.search(r"\{.*\}", t, re.S).group())["marks"]]
        if len(m) != len(r["items"]): raise ValueError("len")
        r["keep"] = m; r["good"] = sum(a * int(b.get("s", 0)) for a, b in zip(m, r["items"]))
    except Exception as e: r["error"] = repr(e)[:100]
    with lock: f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
with ThreadPoolExecutor(24) as ex: list(ex.map(one, enumerate(rows)))
print("FCB_DONE", len(rows))
