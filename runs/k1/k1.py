"""K1 (direction 3 killer test): strongest-baseline evidence contexts on LongMemEval-M dev.
Each builder writes ctx_<NAME>.jsonl in the format run.py reads (qid, hist, budget, ctx, ctx_tok, full, gold_in, gold_total)
plus cost = {model: tokens}. Answering/judging reuse runs/s1/run.py and judge.py (QFILE=questions_all.json).

usage (on H20, in ~/data/rtm/s1):
  python3 k1.py prep  M.json splitM.json devM.json            # dev-only cache (CPU)
  python3 k1.py b6    devM.json                               # session BM25, whole sessions, time-sorted (CPU)
  python3 k1.py b1    devM.json --model Qwen3.5-4B --port 18041[,..] --tag B1_4B   # full scan, P(yes) per session
  python3 k1.py b3    devM.json --model Qwen3.5-9B --port ...  # LongMemEval fact-key + time-aware expansion
  python3 k1.py b4    devM.json --model Qwen3.8-27B-FP8 --port ...   # multi-query + RRF
  python3 k1.py b2    devM.json --emb-port P --rr-port P       # dense + BM25 hybrid + rerank
  python3 k1.py b5    devM.json --model Qwen3.8-27B-FP8 --port ...   # search agent (<=6 steps)
K1_MOCK=1 replaces every model call with a deterministic stub (CPU smoke test of the plumbing).
Never uses session ids or the has_answer flag for ranking (only for gold accounting)."""
import argparse, collections, hashlib, json, math, os, re, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

MOCK = os.environ.get("K1_MOCK") == "1"
_fact_cache, _fc_lock = {}, threading.Lock()
BUDGETS = [int(x) for x in os.environ.get("K1_BUDGETS", "4000,16000").split(",")]
TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to and or in on for is are was were be i you my me it that this with as at by from".split())
toks = lambda t: [w for w in TOK.findall(t.lower()) if w not in STOP]
ntok = lambda s: int(len(s) / 3.6 * 0.78) + 1          # calibrated on Qwen tokenizer (K-S1 smoke: 0.78 x chars/3.6)


# ---------------- BM25 ----------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [collections.Counter(d) for d in docs]; self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs)); df = collections.Counter(w for d in self.tf for w in d)
        n = len(docs); self.idf = {w: math.log(1 + (n - f + .5) / (f + .5)) for w, f in df.items()}

    def scores(self, q):
        out = []
        for tf, L in zip(self.tf, self.len):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f: s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


# ---------------- data ----------------
def rounds_of(q):
    """(session_idx, date, text, n_gold) per user+assistant round, chronological as stored."""
    R = []
    for si, (sess, date) in enumerate(zip(q["haystack_sessions"], q["haystack_dates"])):
        i = 0
        while i < len(sess):
            turns = [sess[i]]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant":
                turns.append(sess[i + 1]); i += 1
            i += 1
            R.append((si, date, "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns),
                      sum(bool(t.get("has_answer")) for t in turns)))
    return R


def pack(q, R, order, B, whole_sessions=False):
    """Fill budget B following `order` (round indices, or session indices if whole_sessions); print chronologically."""
    keep, used = [], 0
    if whole_sessions:
        by = collections.defaultdict(list)
        for i, r in enumerate(R): by[r[0]].append(i)
        for s in order:
            n = sum(ntok(R[i][2]) for i in by[s])
            if used + n > B: continue
            keep += by[s]; used += n
    else:
        for i in order:
            n = ntok(R[i][2])
            if used + n > B: continue
            keep.append(i); used += n
            if used > B * 0.98: break
    keep.sort(); parts, last = [], None
    for i in keep:
        si, date, text, _ = R[i]
        if si != last: parts.append(f"\n### Session date: {date}"); last = si
        parts.append(text)
    return {"qid": q["question_id"], "budget": B, "ctx": "\n".join(parts), "ctx_tok": used, "full": False,
            "gold_in": sum(R[i][3] for i in keep), "gold_total": sum(r[3] for r in R), "keep": keep}


def round_robin(sess_order, R, rscore):
    """Interleave rounds of the ranked sessions: best round of each session first, then second best, ..."""
    by = collections.defaultdict(list)
    for i, r in enumerate(R): by[r[0]].append(i)
    lists = [sorted(by[s], key=lambda i: -rscore[i]) for s in sess_order]
    out, k = [], 0
    while any(k < len(l) for l in lists):
        out += [l[k] for l in lists if k < len(l)]; k += 1
    return out


def rrf(rankings, k=60):
    s = collections.Counter()
    for rk in rankings:
        for pos, i in enumerate(rk): s[i] += 1 / (k + pos + 1)
    return [i for i, _ in s.most_common()]


# ---------------- model calls ----------------
def _post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    err = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
        except Exception as e: err = e
    raise err


class LLM:
    def __init__(self, model, ports):
        self.model, self.ports, self.n, self.lock = model, ports.split(",") if ports else [], 0, threading.Lock()
        self.tokens = collections.Counter()

    def _port(self):
        with self.lock: self.n += 1; return self.ports[self.n % len(self.ports)]

    def chat(self, prompt, max_tokens=512, logprobs=False, qid=None):
        if MOCK:
            return ("Yes" if logprobs else "[]"), (0.9 if logprobs and "MOCKYES" in prompt else 0.1)
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        if logprobs: body.update(logprobs=True, top_logprobs=10)
        r = _post(f"http://127.0.0.1:{self._port()}/v1/chat/completions", body)
        u = r.get("usage", {}); self.tokens[qid] += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        c = r["choices"][0]; text = c["message"]["content"] or ""
        p = None
        if logprobs:
            top = collections.Counter()
            for t in c["logprobs"]["content"][0]["top_logprobs"]: top[t["token"].strip().lower()] += math.exp(t["logprob"])
            y, n = top.get("yes", 0.0), top.get("no", 0.0); p = y / (y + n) if y + n > 0 else 0.0
        return text, p


def jparse(text, default):
    m = re.search(r"(\[.*\]|\{.*\})", text or "", re.S)
    try: return json.loads(m.group(1)) if m else default
    except Exception: return default


# ---------------- builders ----------------
def b6(q, R, a, llm):        # session-level BM25, whole sessions, chronological
    sess = collections.defaultdict(str)
    for r in R: sess[r[0]] += " " + r[2]
    sids = sorted(sess); sc = BM25([toks(sess[s]) for s in sids]).scores(toks(q["question"]))
    order = [sids[i] for i in sorted(range(len(sids)), key=lambda i: -sc[i])]
    return [pack(q, R, order, B, whole_sessions=True) for B in BUDGETS]


SCAN = ("Below is one past chat session between a user and an assistant (date: {date}).\n\n{sess}\n\n"
        "Question asked later (on {qdate}): {q}\n"
        "Does this session contain any information that is needed, even partially, to answer the question "
        "(for example one of several items to count, a date, or an earlier/later value)? Answer Yes or No.")


ROUNDQ = ("Below is one round of a past chat between a user and an assistant (date: {date}).\n\n{text}\n\n"
          "Question asked later (on {qdate}): {q}\n"
          "Does this round contain any information that is needed, even partially, to answer the question? Answer Yes or No.")


def b1(q, R, a, llm):        # full scan: P(yes) per session, then P(yes) per round inside the top relevant sessions
    sess = collections.defaultdict(list)
    for i, r in enumerate(R): sess[r[0]].append(i)
    def one(s):
        text = "\n".join(R[i][2] for i in sess[s])[:60000]
        if MOCK and any(R[i][3] for i in sess[s]): text += " MOCKYES"
        return s, llm.chat(SCAN.format(date=R[sess[s][0]][1], sess=text, qdate=q["question_date"], q=q["question"]),
                           max_tokens=1, logprobs=True, qid=q["question_id"])[1]
    with ThreadPoolExecutor(a.inner) as ex: p = dict(ex.map(one, sorted(sess)))
    top = [s for s in sorted(p, key=lambda s: -p[s]) if p[s] >= a.thr][:40] or sorted(p, key=lambda s: -p[s])[:5]
    cand = [i for s in top for i in sess[s]]
    def rone(i):
        t = R[i][2][:12000] + (" MOCKYES" if MOCK and R[i][3] else "")
        return i, llm.chat(ROUNDQ.format(date=R[i][1], text=t, qdate=q["question_date"], q=q["question"]),
                           max_tokens=1, logprobs=True, qid=q["question_id"])[1]
    with ThreadPoolExecutor(a.inner) as ex: pr = dict(ex.map(rone, cand))
    rs = BM25([toks(r[2]) for r in R]).scores(toks(q["question"]))
    order = sorted(cand, key=lambda i: (-pr[i], -p[R[i][0]]))
    order += [i for i in sorted(range(len(R)), key=lambda i: -rs[i]) if i not in set(cand)]
    with _fc_lock:
        with open(f"scan_{a.tag or 'B1'}.jsonl", "a") as g:
            g.write(json.dumps({"qid": q["question_id"], "p_sess": p, "p_round": pr}) + "\n")
    out = [pack(q, R, order, B) for B in BUDGETS]
    for o in out: o["n_relevant_sessions"] = sum(v >= a.thr for v in p.values())
    return out


FACTS = ("Extract every fact about the USER (events, purchases, plans, preferences, numbers, people, places, dates) "
         "from this chat session dated {date}. Return a JSON list of objects {{\"round\": <round number>, \"fact\": <one "
         "self-contained sentence, include the date if known>}}. Rounds:\n{rounds}")
TIMEQ = ("Today is {qdate}. Question: {q}\nIf answering needs only events from a specific time range, return JSON "
         "{{\"start\": \"YYYY/MM/DD\", \"end\": \"YYYY/MM/DD\"}}; otherwise return {{}}.")


def b3(q, R, a, llm):        # LongMemEval best: round value + fact-expanded key, BM25; time-aware filtering
    sess = collections.defaultdict(list)
    for i, r in enumerate(R): sess[r[0]].append(i)
    sids = q["haystack_session_ids"]
    def facts(s):
        key = sids[s]
        with _fc_lock:
            if key in _fact_cache: return s, _fact_cache[key]
        rounds = "\n".join(f"[{k}] {R[i][2][:1500]}" for k, i in enumerate(sess[s]))[:60000]
        f = jparse(llm.chat(FACTS.format(date=R[sess[s][0]][1], rounds=rounds), max_tokens=800, qid=q["question_id"])[0], [])
        with _fc_lock: _fact_cache[key] = f
        return s, f
    with ThreadPoolExecutor(a.inner) as ex: F = dict(ex.map(facts, sorted(sess)))
    keys = [R[i][2] for i in range(len(R))]
    for s, fl in F.items():
        for f in fl if isinstance(fl, list) else []:
            try: keys[sess[s][int(f["round"])]] += " " + str(f["fact"])
            except Exception: pass
    sc = BM25([toks(k) for k in keys]).scores(toks(q["question"]))
    order = sorted(range(len(R)), key=lambda i: -sc[i])
    rng = jparse(llm.chat(TIMEQ.format(qdate=q["question_date"], q=q["question"]), max_tokens=60, qid=q["question_id"])[0], {})
    if isinstance(rng, dict) and rng.get("start") and rng.get("end"):
        d = lambda i: R[i][1][:10]
        order = [i for i in order if rng["start"] <= d(i) <= rng["end"]] + [i for i in order if not rng["start"] <= d(i) <= rng["end"]]
    return [pack(q, R, order, B) for B in BUDGETS]


MQ = ("Today is {qdate}. A user asks their assistant: \"{q}\"\nThe answer is somewhere in hundreds of past chat sessions. "
      "Write 4 diverse search queries (keywords and phrasings the user may have used; for questions that count or "
      "compare things, one query per likely item or sub-event). Return a JSON list of strings.")


def b4(q, R, a, llm):        # multi-query + RRF over round-level BM25
    qs = jparse(llm.chat(MQ.format(qdate=q["question_date"], q=q["question"]), max_tokens=200, qid=q["question_id"])[0], [])
    bm = BM25([toks(r[2]) for r in R]); ranks = []
    for s in [q["question"]] + [x for x in qs if isinstance(x, str)][:4]:
        sc = bm.scores(toks(s)); ranks.append(sorted(range(len(R)), key=lambda i: -sc[i])[:300])
    order = rrf(ranks); order += [i for i in range(len(R)) if i not in set(order)]
    return [pack(q, R, order, B) for B in BUDGETS]


def embed(texts, a):
    if MOCK:
        return [[int(h, 16) % 7 - 3 for h in hashlib.md5(t.encode()).hexdigest()[:16]] for t in texts]
    out = []
    for k in range(0, len(texts), 64):
        ep = a.emb_port.split(",")[(k // 64) % len(a.emb_port.split(","))]
        r = _post(f"http://127.0.0.1:{ep}/v1/embeddings", {"model": a.emb_model, "input": [t[:6000] for t in texts[k:k + 64]]})
        out += [d["embedding"] for d in r["data"]]
    return out


def b2(q, R, a, llm):        # dense + BM25 hybrid (RRF) + cross-encoder rerank of top 100
    cpath = os.path.join("cand", q["question_id"] + ".json")
    if os.path.exists(cpath):
        fused = json.load(open(cpath))
    else:
        fused = None
    inst = "Instruct: Given a question about a user's past chats, retrieve chat turns that help answer it\nQuery: "
    if fused is not None:
        return _b2_rest(q, R, a, fused)
    E = embed([inst + q["question"]] + [r[2] for r in R], a); qe, re_ = E[0], E[1:]
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    nq = math.sqrt(dot(qe, qe)) or 1
    ds = [dot(qe, e) / (nq * (math.sqrt(dot(e, e)) or 1)) for e in re_]
    bs = BM25([toks(r[2]) for r in R]).scores(toks(q["question"]))
    fused = rrf([sorted(range(len(R)), key=lambda i: -ds[i])[:400], sorted(range(len(R)), key=lambda i: -bs[i])[:400]])
    os.makedirs("cand", exist_ok=True); json.dump(fused, open(cpath, "w"))
    return _b2_rest(q, R, a, fused)


def _b2_rest(q, R, a, fused):
    top = fused[:a.rr_top]
    if MOCK: rr = {i: ds[i] for i in top}
    else:
        PFX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
               "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
        SFX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        query = (PFX + "<Instruct>: Given a question a user asks about their own past chats (asked on " + q["question_date"]
                 + "), judge whether this chat round contains information needed to answer it, even partially\n<Query>: " + q["question"] + "\n")
        rports = a.rr_port.split(","); rr = {}
        for k in range(0, len(top), 50):
            chunk = top[k:k + 50]
            r = _post(f"http://127.0.0.1:{rports[(hash(q['question_id']) + k // 50) % len(rports)]}/v1/rerank",
                      {"model": a.rr_model, "query": query, "documents": ["<Document>: (" + R[i][1] + ") " + R[i][2][:3000] + SFX for i in chunk]})
            for x in r["results"]: rr[chunk[x["index"]]] = x["relevance_score"]
    order = sorted(top, key=lambda i: -rr[i]) + fused[a.rr_top:]
    order += [i for i in range(len(R)) if i not in set(order)]
    return [pack(q, R, order, B) for B in BUDGETS]


DECOMP = ("A user asks their assistant a question about their own past chats (asked on {qdate}): \"{q}\"\n"
          "List the separate pieces of information that must be found in the chat history to answer it (one per item, "
          "event, quantity or date involved; e.g. 'how many courses on edX', 'how many courses on Coursera'). If one piece "
          "is enough, list just one. Return a JSON list of at most 5 short search descriptions.")


def _rr_scores(q, R, a, top, query_text):
    PFX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
           "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SFX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    query = (PFX + "<Instruct>: Given information a user needs from their own past chats, judge whether this chat round "
             "contains it\n<Query>: " + query_text + "\n")
    rports = a.rr_port.split(","); rr = {}
    for k in range(0, len(top), 50):
        chunk = top[k:k + 50]
        r = _post(f"http://127.0.0.1:{rports[(hash(query_text) + k // 50) % len(rports)]}/v1/rerank",
                  {"model": a.rr_model, "query": query, "documents": ["<Document>: (" + R[i][1] + ") " + R[i][2][:3000] + SFX for i in chunk]})
        for x in r["results"]: rr[chunk[x["index"]]] = x["relevance_score"]
    return rr


def b8(q, R, a, llm):        # G1: LLM decomposes the question into sub-needs; rerank per sub-need; round-robin, no duplicates
    fused = json.load(open(os.path.join("cand", q["question_id"] + ".json")))
    top = fused[:a.rr_top]
    subs = jparse(llm.chat(DECOMP.format(qdate=q["question_date"], q=q["question"]), max_tokens=300, qid=q["question_id"])[0], [])
    subs = [x for x in subs if isinstance(x, str)][:5] or [q["question"]]
    whole = _rr_scores(q, R, a, top, q["question"])
    lists = [sorted(top, key=lambda i: -whole[i])] + [sorted(top, key=lambda i, s=_rr_scores(q, R, a, top, sub): -s[i]) for sub in subs]
    order, seen, k = [], set(), 0
    while len(order) < len(top) and k < len(top):
        for l in lists:
            if k < len(l) and l[k] not in seen: order.append(l[k]); seen.add(l[k])
        k += 1
    order += [i for i in fused if i not in seen]
    order += [i for i in range(len(R)) if i not in set(order)]
    out = [pack(q, R, order, B) for B in BUDGETS]
    for o in out: o["n_subs"] = len(subs)
    return out


def b9(q, R, a, llm):        # oracle: all gold rounds first (chronological), then BM25 fill
    bm = BM25([toks(r[2]) for r in R]).scores(toks(q["question"]))
    gold = [i for i, r in enumerate(R) if r[3]]
    order = gold + [i for i in sorted(range(len(R)), key=lambda i: -bm[i]) if not R[i][3]]
    return [pack(q, R, order, B) for B in BUDGETS]


def b10(q, R, a, llm):       # (c) topic aggregation: reranked seeds, then same-topic rounds (BM25 on each seed), chronological fill
    fused = json.load(open(os.path.join("cand", q["question_id"] + ".json")))
    top = fused[:a.rr_top]
    whole = _rr_scores(q, R, a, top, q["question"])
    ranked = sorted(top, key=lambda i: -whole[i])
    seeds = ranked[:a.seeds]
    bm = BM25([toks(r[2]) for r in R])
    lists = [ranked]
    for sd in seeds:
        sc = bm.scores(toks(R[sd][2][:2000]))
        lists.append([i for i in sorted(range(len(R)), key=lambda i: -sc[i])[:60]])
    order, seen, k = [], set(), 0
    while k < 400:
        for l in lists:
            if k < len(l) and l[k] not in seen: order.append(l[k]); seen.add(l[k])
        k += 1
    order += [i for i in fused if i not in seen]
    order += [i for i in range(len(R)) if i not in set(order)]
    return [pack(q, R, order, B) for B in BUDGETS]


def b11(q, R, a, llm):       # (d) 27B pointwise scan over the candidate pool (round level), P(yes) order
    fused = json.load(open(os.path.join("cand", q["question_id"] + ".json")))
    top = fused[:a.rr_top]
    def rone(i):
        return i, llm.chat(ROUNDQ.format(date=R[i][1], text=R[i][2][:12000], qdate=q["question_date"], q=q["question"]),
                           max_tokens=1, logprobs=True, qid=q["question_id"])[1]
    with ThreadPoolExecutor(a.inner) as ex: pr = dict(ex.map(rone, top))
    order = sorted(top, key=lambda i: -pr[i]) + fused[a.rr_top:]
    order += [i for i in range(len(R)) if i not in set(order)]
    return [pack(q, R, order, B) for B in BUDGETS]


def b12(q, R, a, llm):       # (e) time-stratified: rerank the pool, then round-robin across sessions (one best round per session per pass)
    fused = json.load(open(os.path.join("cand", q["question_id"] + ".json")))
    top = fused[:a.rr_top]
    whole = _rr_scores(q, R, a, top, q["question"])
    by = collections.defaultdict(list)
    for i in sorted(top, key=lambda i: -whole[i]): by[R[i][0]].append(i)
    lists = [by[s] for s in sorted(by)]
    order, k = [], 0
    while any(k < len(l) for l in lists):
        order += [l[k] for l in lists if k < len(l)]; k += 1
    order += [i for i in fused if i not in set(order)]
    order += [i for i in range(len(R)) if i not in set(order)]
    return [pack(q, R, order, B) for B in BUDGETS]


AGENT = ("You are searching a user's chat history ({n} past sessions) to collect ALL evidence needed to answer: "
         "\"{q}\" (asked on {qdate}).\nTool: search(query) returns the 8 best-matching chat rounds with ids.\n"
         "Each turn reply with JSON only: {{\"action\": \"search\", \"query\": \"...\"}} or "
         "{{\"action\": \"finish\", \"evidence\": [round ids that are needed]}}. You have at most {k} searches.\n\n{log}")


def b5(q, R, a, llm):        # search agent; collected evidence first, then everything it saw, then BM25
    bm = BM25([toks(r[2]) for r in R]); seen, log, ev = [], "", []
    for step in range(a.steps + 1):
        msg = llm.chat(AGENT.format(n=len({r[0] for r in R}), q=q["question"], qdate=q["question_date"], k=a.steps, log=log),
                       max_tokens=300, qid=q["question_id"])[0]
        act = jparse(msg, {}) if not MOCK else ({"action": "search", "query": q["question"]} if step == 0 else {"action": "finish", "evidence": seen[:3]})
        if not isinstance(act, dict) or act.get("action") != "search" or step == a.steps:
            ev = [int(x) for x in (act.get("evidence", []) if isinstance(act, dict) else []) if str(x).isdigit() and int(x) < len(R)]
            break
        sc = bm.scores(toks(str(act.get("query", "")))); hits = sorted(range(len(R)), key=lambda i: -sc[i])[:8]
        seen += [h for h in hits if h not in seen]
        log += f"\nsearch({act.get('query')!r}):\n" + "\n".join(f"[{h}] ({R[h][1]}) {R[h][2][:500]}" for h in hits)
    qs = bm.scores(toks(q["question"]))
    order = ev + [i for i in seen if i not in ev]; order += [i for i in sorted(range(len(R)), key=lambda i: -qs[i]) if i not in set(order)]
    out = [pack(q, R, order, B) for B in BUDGETS]
    for o in out: o["agent_evidence"] = len(ev)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("data"); ap.add_argument("split", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--model", default=""); ap.add_argument("--port", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--par", type=int, default=4); ap.add_argument("--inner", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--emb-port", default=""); ap.add_argument("--emb-model", default="Qwen3-Embedding-8B")
    ap.add_argument("--rr-port", default=""); ap.add_argument("--rr-model", default="Qwen3-Reranker-8B")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--rr-top", type=int, default=200); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--ids", default="")
    a = ap.parse_args()
    if a.cmd == "prep":
        dev = set(json.load(open(a.split))["dev"])
        json.dump([q for q in json.load(open(a.data)) if q["question_id"] in dev], open(a.out, "w"))
        print("dev cached"); return
    tag = a.tag or a.cmd.upper()
    qs = json.load(open(a.data))
    if a.ids: keep = set(open(a.ids).read().split()); qs = [x for x in qs if x["question_id"] in keep]
    qs = qs[: a.limit or None]
    llm = LLM(a.model, a.port); fn = globals()[a.cmd]
    path = f"ctx_{tag}.jsonl"; done = set()
    if os.path.exists(path): done = {json.loads(l)["qid"] for l in open(path)}
    f = open(path, "a"); lock = threading.Lock()
    def one(q):
        R = rounds_of(q); rows = fn(q, R, a, llm)
        cost = {a.model or "cpu": llm.tokens.get(q["question_id"], 0)}
        with lock:
            for r in rows: f.write(json.dumps({**r, "hist": tag, "cost": cost}) + "\n")
            f.flush()
    with ThreadPoolExecutor(a.par) as ex: list(ex.map(one, [q for q in qs if q["question_id"] not in done]))
    rows = [json.loads(l) for l in open(path)]
    for B in BUDGETS:
        rb = [r for r in rows if r["budget"] == B and r["gold_total"]]
        print(f"{tag} {B}: all-gold {100 * sum(r['gold_in'] == r['gold_total'] for r in rb) / max(1, len(rb)):.1f}% "
              f"of {len(rb)}; mean cost {sum(sum(r['cost'].values()) for r in rb) / max(1, len(rb)):.0f} tok")


if __name__ == "__main__":
    main()
