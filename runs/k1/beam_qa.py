"""BEAM answer + official rubric judge on a k1 context file.
answer: same raw-ChatML reader as run.py (Qwen3.8-27B, thinking <= T); judge: official BEAM unified prompt per rubric item
(0 / 0.5 / 1, averaged per question), judge = same 27B without thinking.
usage (in ~/data/rtm/s1): python3 beam_qa.py --q BEAMJSON --ctx TAG --budget 16000 --ports P,.. --out TAG_qa"""
import argparse, json, os, re, threading
from concurrent.futures import ThreadPoolExecutor
import run as R
JP = open(os.path.join(R.D, "beam_judge_prompt.txt")).read()

def judge_item(port, it, pred):
    p = JP.replace("<rubric_item>", it).replace("<llm_response>", pred)
    r = R.post(port, {"model": R.MODEL, "prompt": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      "max_tokens": 400, "temperature": 0, "stop": ["<|im_end|>"]})
    m = re.search(r'"score"\s*:\s*\[?\s*([01](?:\.\d+)?)', r["choices"][0]["text"])
    return float(m.group(1)) if m else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--q", required=True); ap.add_argument("--ctx", required=True)
    ap.add_argument("--budget", type=int, default=16000); ap.add_argument("--ports", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--think", type=int, default=2000); ap.add_argument("--par", type=int, default=32)
    a = ap.parse_args(); ports = a.ports.split(",")
    qs = {q["question_id"]: q for q in json.load(open(a.q))}
    ctx = {r["qid"]: r for r in map(json.loads, open(os.path.join(R.D, f"ctx_{a.ctx}.jsonl"))) if r["budget"] == a.budget}
    out = os.path.join(R.D, "out", f"{a.out}.jsonl")
    done = {json.loads(l)["qid"] for l in open(out)} if os.path.exists(out) else set()
    todo = [q for q in ctx if q not in done and q in qs]
    lock = threading.Lock(); f = open(out, "a")
    def one(iq):
        i, qid = iq; q = qs[qid]; port = ports[i % len(ports)]
        res = R.answer(port, ctx[qid]["ctx"], q, a.think)
        sc = [judge_item(port, it, res["answer"]) for it in q["rubric"]] or [0.0]
        row = {"qid": qid, "ability": q["question_type"], "ctx": a.ctx, "budget": a.budget, "answer": res["answer"],
               "scores": sc, "score": sum(sc) / len(sc), "gold_in": ctx[qid]["gold_in"], "gold_total": ctx[qid]["gold_total"],
               "think_tok": res["think_tok"]}
        with lock: f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, enumerate(todo)))
    print("beam_qa", a.out, len(todo))

if __name__ == "__main__":
    main()
