"""K-S1 step 1 (CPU only): per question, per history (S/M), per budget, a BM25 round-level context.
Rounds = (user turn + following assistant turn) inside each haystack session, ranked by BM25 against the question,
filled greedily up to the token budget (chars/3.6 estimate), then printed in chronological order with session dates.
Also records how many gold evidence turns (has_answer) the context contains.
usage: <venv python> runs/s1/build_ctx.py ids.txt S|M data.json out.jsonl
"""
import json, re, sys
import math
from collections import Counter
try:
    import ijson
except ImportError:
    ijson = None


class BM25Okapi:   # minimal Okapi BM25 (k1=1.5, b=0.75, same formula as rank_bm25)
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]; self.dl = [len(d) for d in docs]
        self.avg = sum(self.dl) / max(1, len(docs))
        df = Counter(w for d in docs for w in set(d)); N = len(docs)
        self.idf = {w: math.log((N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}

    def get_scores(self, q):
        out = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out

BUDGETS = [4000, 16000, 64000, 200000]
TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to and or in on for is are was were be i you my me it that this with as at by from".split())


def toks(t):
    return [w for w in TOK.findall(t.lower()) if w not in STOP]


def ntok(s):
    return int(len(s) / 3.6) + 1


ids_path, tag, data_path, out_path = sys.argv[1:5]
want = set(open(ids_path).read().split())
out = open(out_path, "w")
items = ijson.items(open(data_path, "rb"), "item") if ijson else json.load(open(data_path))
for q in items:
    if q["question_id"] not in want:
        continue
    rounds = []   # (session_idx, date, text, n_gold)
    for si, (sess, date) in enumerate(zip(q["haystack_sessions"], q["haystack_dates"])):
        i = 0
        while i < len(sess):
            turns = [sess[i]]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant":
                turns.append(sess[i + 1]); i += 1
            i += 1
            text = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)
            rounds.append((si, date, text, sum(bool(t.get("has_answer")) for t in turns)))
    gold_total = sum(r[3] for r in rounds)
    bm = BM25Okapi([toks(r[2]) for r in rounds])
    sc = bm.get_scores(toks(q["question"]))
    order = sorted(range(len(rounds)), key=lambda i: -sc[i])
    total = sum(ntok(r[2]) for r in rounds)
    for B in BUDGETS:
        keep, used = [], 0
        for i in order:
            n = ntok(rounds[i][2])
            if used + n > B:
                continue
            keep.append(i); used += n
            if used > B * 0.98:
                break
        keep.sort()
        parts, last = [], None
        for i in keep:
            si, date, text, _ = rounds[i]
            if si != last:
                parts.append(f"\n### Session date: {date}"); last = si
            parts.append(text)
        out.write(json.dumps({"qid": q["question_id"], "hist": tag, "budget": B, "ctx": "\n".join(parts),
                              "ctx_tok": used, "hist_tok": total, "full": used >= total * 0.98,
                              "gold_in": sum(rounds[i][3] for i in keep), "gold_total": gold_total}) + "\n")
    out.flush()
