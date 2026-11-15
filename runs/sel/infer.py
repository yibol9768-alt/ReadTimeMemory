"""Stepwise set-conditioned selection with a selector served by vLLM (/v1/rerank, Qwen3-reranker overrides).

pool  = the shared fused dense+BM25 pool (pool/<qid>.json), top --P (default 600, same for every method)
step 0: base score b(i) for every pool round with an empty <Selected> block (= the pointwise reranker)
step t: candidates = top-M by base score ∪ top-M by latest conditional score (unselected, fits budget);
        all candidates are re-scored with the current selected set -> c(i); final s(i) = b(i) + lam * (c(i) - b(i))
        (lam=1: pure conditional; lam=0: pointwise); add the best --per-step
--cond 0  : pointwise ranking with the same model (no conditioning)
--xexp 1  : exchange expansion: after choosing a round, also add the immediately following rounds of the same
            session whose user message is a short follow-up (< 30 words), if they fit
No question type is used anywhere. Writes ctx_<TAG>.jsonl (k1/run.py format) with the selection order.

usage: python3 infer.py QFILE --port 18321 --model sel --tag X [--P 600 --M 32 --lam 1 --cond 1 --xexp 0 --par 8]"""
import argparse, json, math, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of, pack, ntok, _post, BUDGETS
from fmt import query_str, doc_str, split_round, round_prev, clean


# official-style format, copied from k1._rr_scores (A0O and the pool's 8B scores were produced with it)
OFF_PFX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
           "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
OFF_SFX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
OFF_INST = "Given information a user needs from their own past chats, judge whether this chat round contains it"


def plain_doc(date, user, assistant, prev="", max_chars=3000):
    """same content as fmt.doc_str (date, follow-up context, 3000-char cap) without the Qwen reranker template"""
    ctx = f"[follow-up to: {clean(prev)[:200]}]\n" if prev else ""
    body = "User: " + user + "\nAssistant: " + assistant
    return f"({date}) {ctx}{body[:max_chars]}"


def followups(R, i, prev_of):
    """indices of the short follow-up rounds right after round i in the same session"""
    out, j = [], i + 1
    while j < len(R) and R[j][0] == R[i][0]:
        u = split_round(R[j][2])[0]
        if len(u.split()) >= 30: break
        out.append(j); j += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qfile"); ap.add_argument("--port", required=True); ap.add_argument("--model", default="sel")
    ap.add_argument("--tag", required=True); ap.add_argument("--pooldir", default="pool")
    ap.add_argument("--P", type=int, default=600); ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--per-step", type=int, default=1); ap.add_argument("--cond", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--xexp", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0, help="selection budget (default: largest of BUDGETS)")
    ap.add_argument("--par", type=int, default=8); ap.add_argument("--ids", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plain", action="store_true", help="generic cross-encoder input (BGE, gte, jina): query = the question, document = (date) [follow-up] User/Assistant text; no Qwen template; pointwise only")
    ap.add_argument("--official", action="store_true", help="Qwen3-Reranker official-style instruction (as k1._rr_scores, used for A0O and the 8B pool scores): no question date, no follow-up context, raw round text; pointwise only")
    ap.add_argument("--trunc", type=int, default=0, help="truncate_prompt_tokens for /v1/rerank (models with short context, e.g. 512)")
    a = ap.parse_args()
    if a.plain or a.official: assert a.cond == 0 and not (a.plain and a.official), "--plain/--official are pointwise and exclusive"
    qs = json.load(open(a.qfile))
    if a.ids: keep = set(open(a.ids).read().split()); qs = [q for q in qs if q["question_id"] in keep]
    qs = qs[: a.limit or None]
    ports = a.port.split(","); cnt = [0]; lk = threading.Lock()

    def score(qstr, docs):
        out = []
        for k in range(0, len(docs), 64):
            with lk: cnt[0] += 1; p = ports[cnt[0] % len(ports)]
            body = {"model": a.model, "query": qstr, "documents": docs[k:k + 64]}
            if a.trunc: body["truncate_prompt_tokens"] = a.trunc
            r = _post(f"http://127.0.0.1:{p}/v1/rerank", body)
            nb = len(docs[k:k + 64]); sc = [None] * nb
            for x in r["results"]:
                j, v = x["index"], x["relevance_score"]
                if not (isinstance(j, int) and 0 <= j < nb) or sc[j] is not None or not math.isfinite(v):
                    raise RuntimeError(f"bad rerank result: index {j} score {v}")
                sc[j] = v
            if any(v is None for v in sc): raise RuntimeError(f"rerank returned {sum(v is not None for v in sc)} of {nb} scores")
            out += sc
        return out

    path = f"ctx_{a.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")

    def one(q):
        R = rounds_of(q); prev = round_prev(q)
        pool = json.load(open(os.path.join(a.pooldir, q["question_id"] + ".json")))
        P = pool["fused"][: a.P]
        if a.official:
            D = {i: "<Document>: (" + R[i][1] + ") " + R[i][2][:3000] + OFF_SFX for i in P}
            base = dict(zip(P, score(OFF_PFX + "<Instruct>: " + OFF_INST + "\n<Query>: " + q["question"] + "\n", [D[i] for i in P])))
        elif a.plain:
            D = {i: plain_doc(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in P}
            base = dict(zip(P, score(q["question"], [D[i] for i in P])))
        else:
            D = {i: doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in P}
            base = dict(zip(P, score(query_str(q["question"], q["question_date"], []), [D[i] for i in P])))
        cond = dict(base); sel = []; used = 0; B = a.budget or max(BUDGETS); calls = 1
        fits = lambda i: i not in sel and used + ntok(R[i][2]) <= B
        while used < B * 0.98:
            if not a.cond or not sel:
                cand = sorted((i for i in P if fits(i)), key=lambda i: -base[i])[: max(a.M, a.per_step)]
                if not cand: break
                if not a.cond:
                    order = cand + sorted((i for i in P if fits(i) and i not in cand), key=lambda i: -base[i])
                    for i in order:
                        if fits(i):
                            sel.append(i); used += ntok(R[i][2])
                            if a.xexp:
                                for j in followups(R, i, prev):
                                    if fits(j): sel.append(j); used += ntok(R[j][2])
                    break
                pick = cand[: a.per_step]
            else:
                c1 = sorted((i for i in P if fits(i)), key=lambda i: -base[i])[: a.M]
                c2 = sorted((i for i in P if fits(i) and i not in c1), key=lambda i: -cond[i])[: a.M // 2]
                cand = c1 + c2
                if not cand: break
                s = score(query_str(q["question"], q["question_date"],
                                    [(R[i][1], split_round(R[i][2])[0], prev[i]) for i in sel]), [D[i] for i in cand]); calls += 1
                for i, v in zip(cand, s): cond[i] = v
                fin = {i: base[i] + a.lam * (cond[i] - base[i]) for i in cand}
                pick = sorted(cand, key=lambda i: -fin[i])[: a.per_step]
            for i in pick:
                if fits(i):
                    sel.append(i); used += ntok(R[i][2])
                    if a.xexp:
                        for j in followups(R, i, prev):
                            if fits(j): sel.append(j); used += ntok(R[j][2])
        rest = sorted((i for i in P if i not in sel), key=lambda i: -base[i])
        ranked = sorted(P, key=lambda i: -base[i])
        # without conditioning, every budget packs the raw score order on its own: `sel` was built for the largest budget and must not steer smaller ones (11-01 audit)
        order = (ranked if not a.cond and not a.xexp else sel + rest) + [i for i in pool["fused"] if i not in set(P)]
        order += [i for i in range(len(R)) if i not in set(order)]
        rows = [pack(q, R, order, Bk) for Bk in BUDGETS]
        with lk:
            for r in rows: f.write(json.dumps({**r, "hist": a.tag, "sel": sel[:80], "calls": calls, "rank": ranked, "score": [round(float(base[i]), 4) for i in ranked]}) + "\n")
            f.flush()

    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, [q for q in qs if q["question_id"] not in done]))
    print("INFER_DONE", a.tag)


if __name__ == "__main__":
    main()
