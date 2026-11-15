"""Offline (CPU only, no LLM): how much harder is FINDING the evidence when the same LongMemEval question gets the
M history (~500 sessions) instead of the S history (~50)? BM25 over the question's own haystack sessions, query =
question text; rank of each gold evidence session (answer_session_ids). Per question type, S vs M.
usage: <venv python> runs/lme_sm_recall.py data/longmemeval/longmemeval_s.json S  > runs/lme_sm/S.jsonl
"""
import json, re, sys
import ijson
from rank_bm25 import BM25Okapi

TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to and or in on for is are was were be i you my me it that this with as at by from".split())


def toks(t):
    return [w for w in TOK.findall(t.lower()) if w not in STOP]


def kind(sid):
    if sid.startswith("answer_"):
        return "evidence"
    if sid.startswith("sharegpt_") or sid.startswith("ultrachat_"):
        return "generic_chat"
    return "other_user_sessions"   # simulated user sessions written for other questions = topical distractors


path, tag = sys.argv[1], sys.argv[2]
for q in ijson.items(open(path, "rb"), "item"):
    sids = q["haystack_session_ids"]
    docs = [toks(" ".join(t["content"] for t in s)) for s in q["haystack_sessions"]]
    bm = BM25Okapi(docs)
    sc = bm.get_scores(toks(q["question"]))
    order = sorted(range(len(sids)), key=lambda i: -sc[i])
    rank = {sids[i]: r + 1 for r, i in enumerate(order)}
    gold = [s for s in q["answer_session_ids"] if s in rank]
    kinds = {}
    for s in sids:
        kinds[kind(s)] = kinds.get(kind(s), 0) + 1
    top20 = [kind(sids[i]) for i in order[:20]]
    print(json.dumps({"tag": tag, "qid": q["question_id"], "type": q["question_type"],
                      "abs": q["question_id"].endswith("_abs"), "n_sess": len(sids),
                      "n_tok": sum(len(d) for d in docs), "gold_ranks": [rank[s] for s in gold],
                      "kinds": kinds, "top20_kinds": {k: top20.count(k) for k in set(top20)}}), flush=True)
