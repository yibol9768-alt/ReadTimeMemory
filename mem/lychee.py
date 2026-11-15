"""Lychee-lite: our re-implementation of the LycheeMemory V2 pipeline (arXiv 2608.12990), as the base system that
V3 builds on. Paper -> here:
  1. segmentation by semantic boundaries, no LLM          -> remote_segment.py on the H20 (embeddings + rule)
  2. one LLM call per segment -> typed, self-contained records with entities, topic, time range, sources,
     plus a disambiguation state carried to the next segment                      -> encode()
  3. vector store + full-text index                       -> dense ranks on the H20 + BM25 over text/entities/topic
  4. one LLM call for query planning; four routes (record vectors, evidence/full-text, time filter, raw dialogue);
     fusion, rerank, dedupe                               -> plan() + context(); rerank is not reproduced
     (no bge-reranker here), fusion is RRF.
Same memory budget (locomo_exp.BUDGET characters) as every other condition. Answers go to the arm's answers.jsonl
as condition "lychee", so locomo_exp judge/report pick them up.

Usage: python3 -m mem.lychee build|answer [--per-cat 20] [--par 8]
"""
import time, argparse, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, retrieve as R

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "lychee", llm.ARM)

ENC_SYS = (
    "You maintain the long-term memory of a conversation between two people. You are given the current "
    "disambiguation state (who is who, what pronouns and nicknames refer to, ongoing threads) and the next segment "
    "of the conversation. Turn the segment into memory records so that any later question about it can be answered "
    "without the original. Each record is self-contained: full names, absolute dates (resolve relative dates "
    "against the session date), no unresolved pronouns. Keep details: names, numbers, places, objects, titles, "
    "what was shown in shared photos, feelings and preferences.\n"
    "Output one JSON object per line, nothing else:\n"
    '{"type": "event|fact|preference|plan|relation|opinion", "text": "...", "entities": ["..."], "topic": "...", '
    '"time": "YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD or empty", "sources": ["D1:3", ...]}\n'
    'Finish with one line {"state": "..."}: the updated disambiguation state, at most 800 characters.')

PLAN_SYS = (
    "You plan the search of a memory of a long conversation between two people (records with entities, topics and "
    "dates). Given a question, output one JSON object, nothing else: "
    '{"queries": [up to 3 short search queries covering every fact the question needs], '
    '"entities": [names or things the answer must involve], '
    '"time": "YYYY-MM-DD..YYYY-MM-DD if the question restricts the time, else empty"}')


def show(t, date):
    return L.show_turn(t, date)


def segments(data, convs, out=None, sess=None):
    """Segment starts per conversation, computed on the H20 (only indices come back)."""
    sess = sess or L.sessions
    path = os.path.join(out or OUT, "segments.json")
    seg = json.load(open(path)) if os.path.exists(path) else {}
    todo = [ci for ci in convs if str(ci) not in seg]
    if todo:
        job = {"convs": {str(ci): [[s["n"], f"{t['speaker']}: {t['text']}"] for s in sess(data[ci])
                                   for t in s["turns"]] for ci in todo}}
        here = os.path.dirname(__file__)
        def ssh(cmd, **kw):                   # the H20 tunnel drops a connection now and then: retry
            for k in range(5):
                try:
                    return subprocess.run(R.SSH + cmd, check=True, **kw)
                except subprocess.CalledProcessError as e:
                    if e.returncode != 255 or k == 4:
                        raise
                    time.sleep(20)
        for f in ("remote_rank.py", "remote_segment.py"):
            ssh([f"mkdir -p ~/v3_emb && cat > ~/v3_emb/{f}"], stdin=open(os.path.join(here, f), "rb"))
        out = ssh(["python3 ~/v3_emb/remote_segment.py"], input=json.dumps(job).encode(), capture_output=True)
        seg.update(json.loads(out.stdout))
        json.dump(seg, open(path, "w"))
    return {ci: seg[str(ci)] for ci in convs}


def norm(x, valid):
    """Source id as the turn id: the prompt's example is "D1:3", so on data whose turns are numbered "T178" the
    model writes "D1:178"; map that back when the bare number names a turn."""
    x = str(x).strip()
    if x in valid:
        return x
    m = re.search(r"(\d+)$", x)
    return "T" + m.group(1) if m and "T" + m.group(1) in valid else None


def parse(text, valid):
    recs, state = [], None
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "state" in r and "text" not in r:
            state = str(r["state"])[:800]
        elif r.get("text"):
            recs.append({"type": r.get("type", ""), "text": r["text"], "entities": r.get("entities") or [],
                         "topic": r.get("topic", ""), "time": r.get("time", "") or "",
                         "sources": [y for y in (norm(x, valid) for x in r.get("sources") or []) if y]})
    return recs, state


def encode(ci, data, starts, out=None, model=None, port=None, sess=None):
    path = os.path.join(out or OUT, f"mem_{ci}.json")
    if os.path.exists(path):
        return
    flat = [(s, t) for s in (sess or L.sessions)(data[ci]) for t in s["turns"]]
    bounds = starts + [len(flat)]
    state, recs, calls = "(empty)", [], 0
    for a, b in zip(bounds, bounds[1:]):
        piece = flat[a:b]
        s = piece[0][0]
        body = (f"Disambiguation state:\n{state}\n\nNext segment (session {s['n']}, date {s['date']}):\n"
                + "\n".join(show(t, s["date"]) for _, t in piece))
        rs, st = parse(llm.call(ENC_SYS, body, tag="ly_enc", model=model, port=port), {t["id"] for _, t in piece})
        recs += rs; calls += 1
        state = st or state
    json.dump({"records": recs, "segments": len(bounds) - 1}, open(path, "w"), ensure_ascii=False, indent=1)
    print("encoded", ci, "segments", len(bounds) - 1, "records", len(recs), flush=True)


def build(data):
    convs = sorted({q["conv"] for q in L.questions(data, ARGS.per_cat)})
    seg = segments(data, convs)
    print("segments", {ci: len(v) for ci, v in seg.items()}, flush=True)
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(convs))) as ex:
        list(ex.map(lambda ci: encode(ci, data, seg[ci]), convs))


def plan(q, model=None, port=None):
    out = llm.call(PLAN_SYS, q, effort="low", max_tokens=4000, tag="ly_plan", model=model, port=port)
    try:
        p = json.loads(out[out.index("{"):out.rindex("}") + 1])
    except Exception:
        p = {}
    return {"queries": [q] + [x for x in p.get("queries") or [] if isinstance(x, str)][:3],
            "entities": [x for x in p.get("entities") or [] if isinstance(x, str)],
            "time": p.get("time") or ""}


def span(s):
    d = re.findall(r"\d{4}-\d{2}-\d{2}", s or "")
    return (d[0], d[-1]) if d else None


def doc(r):
    return f"{r['text']} | {' '.join(r['entities'])} | {r['topic']}"


def context(q, p, recs, ss):
    turns = {t["id"]: show(t, s["date"]) for s in ss for t in s["turns"]}
    docs = [r["text"] for r in recs]
    bm = L.BM25([doc(r) for r in recs])
    routes = []
    for x in p["queries"]:
        routes.append(R.dense_rank(x, docs))                                     # record vectors
        routes.append(bm.rank(x + " " + " ".join(p["entities"])))                # full text / evidence
    w = span(p["time"])
    if w:                                                                        # time filter
        hit = [i for i in bm.rank(q) if span(recs[i]["time"]) and
               span(recs[i]["time"])[0] <= w[1] and span(recs[i]["time"])[1] >= w[0]]
        if hit:
            routes.append(hit)
    order, seen_words, items, used = R.rrf(routes), [], [], set()
    for rank, i in enumerate(order):                                             # dedupe near-duplicates
        ws = set(L.toks(recs[i]["text"]))
        if any(len(ws & o) / max(1, len(ws | o)) > 0.8 for o in seen_words):
            continue
        seen_words.append(ws)
        items.append(f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}")
        if len(items) <= 6 and not recs[i]["type"].startswith("rollup"):         # provenance for the top records
            for sid in recs[i]["sources"]:
                if sid not in used:
                    used.add(sid); items.append("    " + turns[sid])
    raw_ids = list(turns)
    raw = [turns[raw_ids[i]] for i in L.BM25([turns[i] for i in raw_ids]).rank(q)[:8] if raw_ids[i] not in used]
    head = L.fill_to(items, L.BUDGET - 1500)
    return head + "\n\nRaw dialogue matches:\n" + L.fill_to(raw, L.BUDGET - len(head) - 25)


def answer(data):
    qs = L.questions(data, ARGS.per_cat)
    mems = {ci: json.load(open(os.path.join(OUT, f"mem_{ci}.json")))["records"] for ci in {q["conv"] for q in qs}}
    sss = {ci: L.sessions(data[ci]) for ci in mems}
    llm.check_quota()
    with ThreadPoolExecutor(ARGS.par) as ex:
        plans = list(ex.map(lambda q: plan(q["question"]), qs))
    R.precompute_ranks([[r["text"] for r in m] for m in mems.values()],
                       [([r["text"] for r in mems[q["conv"]]], x) for q, p in zip(qs, plans) for x in p["queries"]])
    path = os.path.join(L.AOUT, "answers.jsonl")
    done = {(r["qid"], r["cond"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(q, p) for q, p in zip(qs, plans) if (q["qid"], "lychee") not in done]

    def one(job):
        q, p = job
        ctx = context(q["question"], p, mems[q["conv"]], sss[q["conv"]])
        a = llm.call(L.ANS_SYS, f"Memory:\n{ctx}\n\nQuestion: {q['question']}", tag="ans_lychee")
        return {"qid": q["qid"], "cond": "lychee", "cat": q["category"], "q": q["question"],
                "gold": str(q["answer"]), "pred": a.strip(), "ctx_chars": len(ctx)}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["build", "answer"])
    p.add_argument("--per-cat", type=int, default=20)
    p.add_argument("--par", type=int, default=8)
    ARGS = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(L.AOUT, exist_ok=True)
    data = json.load(open(L.DATA))
    {"build": build, "answer": answer}[ARGS.step](data)
