"""Strong-LLM listwise set selection baseline (SetR / RankGPT style, carrying the selected set), and MMR baseline.

l27 : candidates = top-N of the 8B reranker over the shared P=600 pool; the 27B reads them in windows of W (each
      round shown as date + context + user message + start of the reply) together with the list of rounds it already
      selected, and returns the ids that add needed, not-yet-covered information. Selected rounds come first (in
      selection order), then the rest by 8B score. No question type is given.
mmr : greedy MMR over the 8B scores with Qwen3-Embedding-0.6B round vectors (p0cache/<conv>.npz, context version).

usage: python3 l27.py l27 QFILE --tag L27 --ports 18301,18302 [--N 120 --W 20 --par 8]
       python3 l27.py mmr QFILE --tag MMR [--lam 0.7]"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of, pack, BUDGETS, _post
from fmt import split_round, round_prev, clean

PROMPT = """A user asks a question about their own past chats with an assistant (asked on {qdate}):
QUESTION: {question}

Rounds you have ALREADY selected as evidence (short form):
{selected}

New candidate rounds:
{cands}

Select the candidate rounds that add information needed to answer the question and NOT already provided by the
selected rounds. For questions asking for an ordered list, a summary or several facts, select one round for every
part that is not covered yet; skip rounds that repeat a covered part or are unrelated. Short follow-up rounds count
only if they add needed detail.
Return only a JSON list of candidate ids (e.g. [3, 7]); return [] if none."""


def show(R, prev, i, n=700):
    u, a = split_round(R[i][2])
    ctx = f" [follow-up to: {clean(prev[i])[:120]}]" if prev[i] and len(u.split()) < 30 else ""
    return f"({R[i][1]}){ctx} USER: {clean(u)[:n]} || ASSISTANT: {clean(a)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("qfile"); ap.add_argument("--tag", required=True)
    ap.add_argument("--ports", default="18301,18302,18305"); ap.add_argument("--model", default="Qwen3.8-27B-FP8")
    ap.add_argument("--N", type=int, default=120); ap.add_argument("--W", type=int, default=20); ap.add_argument("--P", type=int, default=600)
    ap.add_argument("--par", type=int, default=8); ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--pooldir", default="pool"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ids", default="")
    a = ap.parse_args()
    qs = json.load(open(a.qfile))   # 10-31 audit: no n_src filter (every method must cover the same question set)
    if a.ids: keep = set(open(a.ids).read().split()); qs = [q for q in qs if q["question_id"] in keep]
    qs = qs[: a.limit or None]
    ports = a.ports.split(","); lk = threading.Lock(); cnt = [0]
    path = f"ctx_{a.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")

    def chat(p):
        with lk: cnt[0] += 1; port = ports[cnt[0] % len(ports)]
        r = _post(f"http://127.0.0.1:{port}/v1/chat/completions",
                  {"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}})
        return r["choices"][0]["message"]["content"] or ""

    def one(q):
        R = rounds_of(q); prev = round_prev(q)
        pool = json.load(open(os.path.join(a.pooldir, q["question_id"] + ".json")))
        rr = {int(k): v for k, v in pool["rr8"].items()}
        P = pool["fused"][: a.P]
        ranked = sorted(P, key=lambda i: -rr.get(i, -1))
        sel = []
        if a.mode == "l27":
            cands = ranked[: a.N]
            for w in range(0, len(cands), a.W):
                win = cands[w:w + a.W]
                s = "\n".join(f"- ({R[i][1]}) {clean(split_round(R[i][2])[0])[:150]}" for i in sel) or "(none)"
                c = "\n".join(f"[{k}] {show(R, prev, i)}" for k, i in enumerate(win))
                txt = chat(PROMPT.format(qdate=q["question_date"], question=q["question"], selected=s, cands=c))
                m = re.search(r"\[[\d,\s]*\]", txt)
                try: ids = json.loads(m.group()) if m else []
                except Exception: ids = []
                sel += [win[k] for k in ids if isinstance(k, int) and 0 <= k < len(win) and win[k] not in sel]
        else:  # mmr
            import numpy as np
            z = np.load(os.path.join("p0cache", q["question_id"].split("_")[0] + ".npz")); EC = z["EC"]
            # 10-31 audit: whole P=600 pool; relevance = rank-normalised 8B order (raw P(yes) is ~0 past rank 40,
            # which made the cosine term dominate after the first pick)
            cand = ranked[: a.P]; rel = 1.0 - np.arange(len(cand)) / max(1, len(cand))
            chosen = []
            while len(chosen) < 80 and len(chosen) < len(cand):
                if chosen:
                    sim = (EC[cand] @ EC[chosen].T).max(1)
                else:
                    sim = np.zeros(len(cand))
                sc = a.lam * rel - (1 - a.lam) * sim
                for k, i in enumerate(cand):
                    if i in chosen: sc[k] = -1e9
                chosen.append(cand[int(sc.argmax())])
            sel = [int(i) for i in chosen]
        order = sel + [i for i in ranked if i not in sel] + [i for i in pool["fused"][a.P:]]
        order += [i for i in range(len(R)) if i not in set(order)]
        rows = [pack(q, R, order, B) for B in BUDGETS]
        with lk:
            for r in rows: f.write(json.dumps({**r, "hist": a.tag, "sel": sel[:80]}) + "\n")
            f.flush()

    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, [q for q in qs if q["question_id"] not in done]))
    print("DONE", a.tag)


if __name__ == "__main__":
    main()
