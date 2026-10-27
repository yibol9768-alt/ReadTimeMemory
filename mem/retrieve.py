"""Dense retrieval with the H20 embedding service (qwen3-embedding-8b; the ranking itself runs on the H20),
hybrid fusion with BM25, LLM-written sub-queries, and per-person profile pages rebuilt from the records."""
import hashlib, json, math, os, threading, urllib.request
from . import llm



def _key(t):
    return hashlib.sha256(t.encode()).hexdigest()[:32]


SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "h20"]   # ~/.ssh/config alias (key ~/.ssh/h20_key)
_RANKS = os.path.join(os.path.dirname(__file__), "..", "runs", "rank_cache.json")
_ranks = json.load(open(_RANKS)) if os.path.exists(_RANKS) else {}


def group_of(docs):
    return _key("\n".join(docs))


def precompute_ranks(doc_sets, queries):
    """doc_sets: list of doc lists; queries: list of (docs, query). Ships the texts to the H20, where
    remote_rank.py embeds them next to the embedding server; only the rankings come back."""
    import subprocess
    docs = {group_of(d): d for d in doc_sets}
    qs = [[group_of(d), q] for d, q in queries if group_of(d) + "\t" + q not in _ranks]
    if not qs:
        return
    here = os.path.dirname(__file__)
    for attempt in range(6):                    # the upload hits the same flaky tunnel; retry it too
        try:
            subprocess.run(SSH + ["mkdir -p ~/v3_emb && cat > ~/v3_emb/remote_rank.py"],
                           stdin=open(os.path.join(here, "remote_rank.py"), "rb"), check=True)
            break
        except subprocess.CalledProcessError:
            if attempt == 5:
                raise
            import time; time.sleep(30 * (attempt + 1))
    job = json.dumps({"docs": {g: docs[g] for g in {g for g, q in qs}}, "queries": qs}).encode()
    for attempt in range(6):                    # the ssh tunnel to the H20 drops now and then (exit 255)
        try:
            out = subprocess.run(SSH + ["python3 ~/v3_emb/remote_rank.py"], input=job, capture_output=True, check=True)
            break
        except subprocess.CalledProcessError:
            if attempt == 5:
                raise
            import time; time.sleep(30 * (attempt + 1))
    _ranks.update(json.loads(out.stdout))
    import fcntl                                # several runs share this file: merge under a lock, replace atomically
    with open(_RANKS + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            disk = json.load(open(_RANKS)) if os.path.exists(_RANKS) else {}
        except Exception:
            disk = {}
        disk.update(_ranks); _ranks.update(disk)
        json.dump(disk, open(_RANKS + ".tmp", "w")); os.replace(_RANKS + ".tmp", _RANKS)


def dense_rank(query, docs):
    return _ranks[group_of(docs) + "\t" + query]


def rrf(rankings, k=60):
    score = {}
    for r in rankings:
        for pos, i in enumerate(r):
            score[i] = score.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


SUBQ_SYS = ("You help search a memory of a long conversation between two people. Given a question, write 3 short "
            "search queries that together would find every piece of information needed to answer it (for questions "
            "that combine several facts, one query per fact; otherwise rephrasings with different keywords). "
            "Output one query per line, nothing else.")


def sub_queries(q):
    out = llm.call(SUBQ_SYS, q, effort="low", max_tokens=4000, tag="subq")
    return [q] + [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:3]


PROFILE_SYS = ("You are given memory records extracted from a long conversation between two people, in time order. "
               "Write a profile page for each of the two people that gathers what is scattered across the records: "
               "identity, family and relationships, work and studies, hobbies and recurring activities, possessions "
               "and pets, values and opinions, goals and plans, and a dated timeline of main events. "
               "Aggregate (e.g. list ALL hobbies mentioned anywhere), keep absolute dates, do not invent anything.")


def profiles(recs, budget):
    material = "\n".join("- " + r["text"] for r in recs)
    prompt = f"Records:\n{material}\n\nWrite the two profile pages, at most {budget} characters in total."
    text = llm.call(PROFILE_SYS, prompt, tag="profile")
    for i in range(3):
        if len(text) <= budget:
            break
        text = llm.call(PROFILE_SYS, prompt + f"\n\nYour previous attempt was {len(text)} characters, over the "
                        f"limit. Write it again in at most {budget - 600 * (i + 1)} characters.", tag="profile_retry")
    return text[:budget]
