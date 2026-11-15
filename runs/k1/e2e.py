"""Candidate 1 killer test: an untrained small model searches the history itself (same tool as k1 b5) and answers
itself (thinking on). n samples per question -> pass@n. Writes out/<tag>.jsonl rows that judge.py reads
(qid, hist=tag, budget=sample index, think=0, answer, ...).
One process per question (BM25 index built once, shared by its n samples); samples of a question run in threads.
usage (in ~/data/rtm/s1): QFILE=questions_all.json python3 e2e.py devM.json --model Qwen3.5-4B --port 18061,18062 --tag E2E_4B --n 8"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "k1.py")).read()
exec(_src[: _src.index("def " + "main():")])          # BM25, rounds_of, pack, AGENT, jparse, _post, toks

ANS = ("I will give you several history chats between you and a user. Please answer the question based on the "
       "relevant chat history.\n\n\nHistory Chats:\n{ctx}\n\nCurrent Date: {date}\nQuestion: {q}\nAnswer:")
G = None


def chat(port, model, prompt, max_tokens, temp, think):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temp, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": think}}
    r = _post(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    u = r.get("usage", {})
    return (r["choices"][0]["message"]["content"] or ""), u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)


def episode(q, R, a, port, s, bm, qscore):
    seen, log, ev, tok, steps = [], "", [], 0, 0
    for step in range(a.steps + 1):
        msg, t = chat(port, a.model, AGENT.format(n=len({r[0] for r in R}), q=q["question"], qdate=q["question_date"],
                                                  k=a.steps, log=log), 300, a.temp, False)
        tok += t; act = jparse(msg, {})
        if not isinstance(act, dict) or act.get("action") != "search" or step == a.steps:
            ev = [int(x) for x in (act.get("evidence", []) if isinstance(act, dict) else []) if str(x).isdigit() and int(x) < len(R)]
            break
        steps += 1
        sc = bm.scores(toks(str(act.get("query", "")))); hits = sorted(range(len(R)), key=lambda i: -sc[i])[:8]
        seen += [h for h in hits if h not in seen]
        log += f"\nsearch({act.get('query')!r}):\n" + "\n".join(f"[{h}] ({R[h][1]}) {R[h][2][:500]}" for h in hits)
    order = ev + [i for i in seen if i not in ev]; sset = set(order)
    order += [i for i in qscore if i not in sset]
    c = pack(q, R, order, a.budget)
    ans, t = chat(port, a.model, ANS.format(ctx=c["ctx"], date=q["question_date"], q=q["question"]), a.think_tokens, a.temp, bool(a.answer_think))
    tok += t
    return {"qid": q["question_id"], "type": q["question_type"], "hist": a.tag, "budget": s, "think": 0,
            "answer": ans.strip(), "tokens": tok, "steps": steps, "n_evidence": len(ev), "ctx_tok": c["ctx_tok"],
            "gold_in": c["gold_in"], "gold_total": c["gold_total"]}


def worker(i):
    qs, a, done = G; q = qs[i]; ports = a.port.split(",")
    R = rounds_of(q); bm = BM25([toks(r[2]) for r in R])
    qs_ = bm.scores(toks(q["question"])); qscore = sorted(range(len(R)), key=lambda j: -qs_[j])
    ss = [s for s in range(a.n) if (q["question_id"], s) not in done]
    with ThreadPoolExecutor(max(1, len(ss))) as ex:
        return list(ex.map(lambda s: episode(q, R, a, ports[(i * a.n + s) % len(ports)], s, bm, qscore), ss))


def main():
    global G
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--model", required=True); ap.add_argument("--port", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=6); ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--think-tokens", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--answer-think", type=int, default=1)
    a = ap.parse_args()
    qs = json.load(open(a.data))[: a.limit or None]
    out = os.path.join("out", f"{a.tag}.jsonl")
    done = {(r["qid"], r["budget"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    G = (qs, a, done)
    todo = [i for i, q in enumerate(qs) if any((q["question_id"], s) not in done for s in range(a.n))]
    n = 0
    with open(out, "a") as f, ProcessPoolExecutor(a.procs) as ex:     # fork: workers share qs copy-on-write
        for rows in ex.map(worker, todo):
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush(); n += len(rows)
    print("e2e done", a.tag, n)


if __name__ == "__main__":
    main()
