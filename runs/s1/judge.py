"""K-S1 judge: LongMemEval official answer-check prompt (data/longmemeval/evaluate_qa_official.py), judge model via
vLLM chat completions without thinking. usage: python3 runs/s1/judge.py --tag t1 --port 18031 [--model Qwen3.8-27B-FP8]
Writes runs/s1/out/<tag>.judged_<model>.jsonl (resumable)."""
import argparse, importlib.util, json, os, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(D))
_src = open(os.path.join(D, "evaluate_qa_official.py")).read()
_fn = _src[_src.index("def get_anscheck_prompt"):_src.index("if __name__ == '__main__':")]
ev = type("ev", (), {})
_ns = {}
exec(_fn, _ns)                      # only the official prompt function, not the script's API-client imports
ev.get_anscheck_prompt = staticmethod(_ns["get_anscheck_prompt"])


def chat(port, model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 20, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception as e:
            err = e
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--port", required=True)
    ap.add_argument("--model", default="Qwen3.8-27B-FP8"); ap.add_argument("--par", type=int, default=32)
    a = ap.parse_args()
    qs = {x["question_id"]: x for x in json.load(open(os.path.join(D, "questions60.json")))}
    rows = [json.loads(l) for l in open(os.path.join(D, "out", f"{a.tag}.jsonl"))]
    out = os.path.join(D, "out", f"{a.tag}.judged_{a.model}.jsonl")
    key = lambda r: (r["qid"], r["hist"], r["budget"], r["think"])
    done = {key(json.loads(l)) for l in open(out)} if os.path.exists(out) else set()
    todo = [r for r in rows if key(r) not in done]
    lock = threading.Lock(); f = open(out, "a")

    ports = a.port.split(",")

    def one(ir):
        i, r = ir
        port = ports[i % len(ports)]
        q = qs[r["qid"]]
        p = ev.get_anscheck_prompt(q["question_type"], q["question"], q["answer"], r["answer"],
                                   abstention="_abs" in r["qid"])
        verdict = chat(port, a.model, p)
        r2 = {**r, "label": "yes" in verdict.lower(), "judge_raw": verdict[:40]}
        r2.pop("reasoning", None)
        with lock:
            f.write(json.dumps(r2, ensure_ascii=False) + "\n"); f.flush()

    with ThreadPoolExecutor(a.par) as ex:
        list(ex.map(one, enumerate(todo)))
    print("judged", a.tag, len(todo))


if __name__ == "__main__":
    main()
