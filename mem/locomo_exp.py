"""Pilot on LoCoMo: does a memory lose more by being rewritten again and again, and does reading the raw turns
matter more than how the memory records are cut? Every memory condition answers from at most BUDGET characters.

Conditions
  full        whole conversation in context (ceiling)
  once        one summary written with the whole conversation in view
  roll        a summary rewritten after every session (sees only the old summary and the new session)
  raw         BM25 over raw turns
  rec_sess    records extracted per session, BM25 over records only
  rec_chunk   records extracted per 4-turn chunk (eager), records only
  rec_sess+raw, rec_chunk+raw   the same records, each followed by the raw turns it came from

Usage: python3 -m mem.locomo_exp build|answer|judge|report [--per-cat 50] [--par 12]
"""
import argparse, collections, json, math, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from . import llm, retrieve as R

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "locomo10.json")
OUT = os.path.join(ROOT, "runs", "locomo")          # memories (written by gpt-6-luna on Pro) live here
# Views, answers and judgments belong to the model that reads: pro6 (gpt-6-luna) keeps the old paths,
# plus56 (gpt-5.6-luna, see llm.ARM) gets its own folder so the two are never mixed in one table.
AOUT = OUT if llm.ARM == "pro6" else os.path.join(OUT, llm.ARM)
# Memories: pro6 and plus56 read the gpt-6-luna memories in OUT; ds41 writes and reads its own (DeepSeek end to end).
MDIR = os.path.join(OUT, llm.ARM) if llm.ARM in ("ds41", "qwen38") else OUT
BUDGET = 8000          # characters of memory shown when answering, for every condition but full
CHUNK = 4
CONDS = ["full", "once", "roll", "raw", "rec_sess", "rec_chunk", "rec_sess+raw", "rec_chunk+raw",
         "emb", "hyb", "hyb_mq", "view+hyb_mq", "lychee", "lychee_roll"]
VIEW_BUDGET = 4000     # characters of profile pages in view+hyb_mq; the records get the rest of BUDGET
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


# ---------- data ----------
def sessions(conv):
    c = conv["conversation"]
    n = sorted(int(m.group(1)) for k in c for m in [re.fullmatch(r"session_(\d+)", k)] if m)
    out = []
    for i in n:
        turns = []
        for t in c[f"session_{i}"]:
            txt = t["text"] + (f" [shares a photo: {t['blip_caption']}]" if t.get("blip_caption") else "")
            turns.append({"id": t["dia_id"], "speaker": t["speaker"], "text": txt})
        out.append({"n": i, "date": c.get(f"session_{i}_date_time", ""), "turns": turns})
    return out


def show_turn(t, date):
    return f"[{t['id']}] ({date}) {t['speaker']}: {t['text']}"


def show_session(s):
    return f"=== Session {s['n']} — {s['date']} ===\n" + "\n".join(show_turn(t, s["date"]) for t in s["turns"])


def questions(data, per_cat, seed=0):
    rng = random.Random(seed)
    qs = [dict(q, conv=ci, qid=f"{ci}-{j}") for ci, conv in enumerate(data) for j, q in enumerate(conv["qa"])
          if q.get("category") in CAT and "answer" in q]
    out = []
    for c in CAT:
        pool = [q for q in qs if q["category"] == c]
        rng.shuffle(pool)
        out += pool[:per_cat]
    return out


# ---------- BM25 ----------
def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.d = [toks(x) for x in docs]
        self.avg = sum(map(len, self.d)) / max(1, len(self.d))
        df = collections.Counter(w for d in self.d for w in set(d))
        N = len(self.d)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [collections.Counter(d) for d in self.d]
        self.k1, self.b = k1, b

    def rank(self, q):
        qs = toks(q)
        sc = []
        for i, (tf, d) in enumerate(zip(self.tf, self.d)):
            s = 0.0
            for w in qs:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            sc.append((s, i))
        return [i for s, i in sorted(sc, reverse=True)]


# ---------- memory construction ----------
SUM_SYS = ("You keep the memory of a long conversation between two people. Later, someone will ask questions about "
           "ANY detail of it: events, dates, names, places, plans, preferences, feelings, things shared in photos. "
           "The original conversation will NOT be available then; only your memory will. "
           f"Write the memory as compact plain-text notes, at most {BUDGET} characters. Use absolute dates.")
REC_SYS = ("Extract memory records from a piece of a conversation between two people, so that questions about any "
           "detail can be answered later without the original. Each record must be self-contained: full names, "
           "absolute dates (resolve 'yesterday', 'last week' against the session date), no pronouns without referent. "
           "Output one JSON object per line, nothing else: "
           '{"text": "...", "sources": ["D1:3", ...]}  where sources are the turn ids the record comes from.')


def retry_limit(i):
    """Shrinking limit for the i-th retry: 1500 less each time (as in all 8000-character runs), never below
    half the budget once that would drop under 1000 characters (tight budgets)."""
    x = BUDGET - 1500 * (i + 1)
    return x if x >= 1000 else BUDGET // 2


def within_budget(prompt, tag):
    """The model often overshoots the length limit. Cutting the tail would drop whatever came last, so instead
    ask again with the same material in view; cut only if it still overshoots after three tries."""
    text = llm.call(SUM_SYS, prompt, tag=tag)
    for i in range(3):
        if len(text) <= BUDGET:
            break
        text = llm.call(SUM_SYS, prompt + f"\n\nYour previous attempt was {len(text)} characters, over the limit. "
                        f"Write it again in at most {retry_limit(i)} characters.", tag=tag + "_retry")
    if len(text) > BUDGET:
        print("cut", tag, len(text), flush=True)
    return text[:BUDGET]


def build_once(ss):
    return within_budget("The whole conversation:\n\n" + "\n\n".join(map(show_session, ss)) +
                         f"\n\nWrite the memory (at most {BUDGET} characters).", "once")


def build_roll(ss):
    mem = ""
    for s in ss:
        mem = within_budget(f"Your current memory of the earlier sessions:\n\n{mem or '(empty)'}\n\n"
                            f"The next session:\n\n{show_session(s)}\n\n"
                            f"Write the updated memory of everything so far (at most {BUDGET} characters).", "roll")
    return mem


def parse_records(text, valid):
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("text"):
            out.append({"text": r["text"], "sources": [x for x in r.get("sources", []) if x in valid]})
    return out


def build_records(ss, gran):
    pieces = []
    for s in ss:
        if gran == "sess":
            pieces.append((s, s["turns"]))
        else:
            pieces += [(s, s["turns"][i:i + CHUNK]) for i in range(0, len(s["turns"]), CHUNK)]

    def one(p):
        s, turns = p
        body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(show_turn(t, s["date"]) for t in turns)
        return parse_records(llm.call(REC_SYS, body, tag="rec_" + gran), {t["id"] for t in turns})
    with ThreadPoolExecutor(ARGS.par) as ex:
        return [r for rs in ex.map(one, pieces) for r in rs]


def build(data):
    os.makedirs(OUT, exist_ok=True)
    needed = sorted({q["conv"] for q in questions(data, ARGS.per_cat)})
    def conv_mem(ci):
        path = os.path.join(MDIR, f"mem_{ci}.json")
        if os.path.exists(path):
            return
        ss = sessions(data[ci])
        kinds = {"once": lambda: build_once(ss), "roll": lambda: build_roll(ss),
                 "rec_sess": lambda: build_records(ss, "sess"), "rec_chunk": lambda: build_records(ss, "chunk")}
        m = {k: kinds[k]() for k in ARGS.kinds}
        json.dump(m, open(path, "w"), ensure_ascii=False, indent=1)
        print("built", ci, {k: (len(v) if isinstance(v, str) else f"{len(v)} records") for k, v in m.items()}, flush=True)
    llm.check_quota()
    with ThreadPoolExecutor(min(10, ARGS.par)) as ex:
        list(ex.map(conv_mem, needed))


def ranks(data):
    """Sub-queries come from luna here; dense rankings for every query are computed on the H20 in one trip."""
    qs = questions(data, ARGS.per_cat)
    recs = {ci: [r["text"] for r in json.load(open(os.path.join(MDIR, f"mem_{ci}.json")))["rec_sess"]]
            for ci in {q["conv"] for q in qs}}
    llm.check_quota()
    with ThreadPoolExecutor(ARGS.par) as ex:
        subs = list(ex.map(lambda q: R.sub_queries(q["question"]), qs))
    pairs = [(recs[q["conv"]], x) for q, s in zip(qs, subs) for x in s]
    R.precompute_ranks(list(recs.values()), pairs)
    print("ranked", len(pairs), "queries", flush=True)


def views(data):
    llm.check_quota()
    def one(ci):
        vp = os.path.join(AOUT, f"view_{ci}.json")
        if not os.path.exists(vp):
            m = json.load(open(os.path.join(MDIR, f"mem_{ci}.json")))
            page = R.profiles(m["rec_sess"], VIEW_BUDGET)
            json.dump({"profile": page}, open(vp, "w"), ensure_ascii=False, indent=1)
            print("view", ci, len(page), flush=True)
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(one, range(len(data))))


# ---------- answering ----------
ANS_SYS = ("Answer a question about a past conversation between two people, using only the memory provided. "
           "Give a short answer (a few words or one sentence). For time questions give the absolute date or period. "
           "If the memory does not contain the answer, give your best guess.")


def fill(items):
    out, n = [], 0
    for x in items:
        if n + len(x) + 1 > BUDGET:
            break
        out.append(x); n += len(x) + 1
    return "\n".join(out)


def fill_to(items, budget):
    out, n = [], 0
    for x in items:
        if n + len(x) + 1 > budget:
            break
        out.append(x); n += len(x) + 1
    return "\n".join(out)


def context(cond, q, ss, mem):
    turns = {t["id"]: show_turn(t, s["date"]) for s in ss for t in s["turns"]}
    if cond in ("emb", "hyb", "hyb_mq", "view+hyb_mq"):
        recs = mem["rec_sess"]
        docs = [r["text"] for r in recs]
        if cond == "emb":
            order = R.dense_rank(q, docs)
        else:
            queries = R.sub_queries(q) if "mq" in cond else [q]
            bm = BM25(docs)
            order = R.rrf([r for x in queries for r in (bm.rank(x), R.dense_rank(x, docs))])
        lines = ["- " + recs[i]["text"] for i in order]
        if cond.startswith("view"):
            page = mem["profile"]
            return "Profile pages:\n" + page + "\n\nRelevant records:\n" + fill_to(lines, BUDGET - len(page))
        return fill(lines)
    if cond == "full":
        return "\n\n".join(map(show_session, ss))
    if cond in ("once", "roll"):
        return mem[cond]
    if cond == "raw":
        ids = list(turns)
        return fill([turns[ids[i]] for i in BM25([turns[i] for i in ids]).rank(q)])
    recs = mem[cond.split("+")[0]]
    order = BM25([r["text"] for r in recs]).rank(q)
    if "+raw" not in cond:
        return fill(["- " + recs[i]["text"] for i in order])
    items, seen = [], set()
    for i in order:
        items.append("- " + recs[i]["text"])
        for s in recs[i]["sources"]:
            if s not in seen:
                seen.add(s); items.append("    " + turns[s])
    return fill(items)


def answer(data):
    qs = questions(data, ARGS.per_cat)
    mems = {ci: json.load(open(os.path.join(MDIR, f"mem_{ci}.json"))) for ci in {q["conv"] for q in qs}}
    for ci, m in mems.items():
        vp = os.path.join(AOUT, f"view_{ci}.json")
        if os.path.exists(vp):
            m["profile"] = json.load(open(vp))["profile"]
    sss = {ci: sessions(data[ci]) for ci in mems}
    path = os.path.join(AOUT, "answers.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["qid"], r["cond"]) for r in map(json.loads, open(path))}
    jobs = [(q, c) for c in ARGS.conds for q in qs if (q["qid"], c) not in done]
    llm.check_quota()
    def one(job):
        q, c = job
        ctx = context(c, q["question"], sss[q["conv"]], mems[q["conv"]])
        a = llm.call(ANS_SYS, f"Memory:\n{ctx}\n\nQuestion: {q['question']}", tag="ans_" + c)
        return {"qid": q["qid"], "cond": c, "cat": q["category"], "q": q["question"], "gold": str(q["answer"]),
                "pred": a.strip(), "ctx_chars": len(ctx)}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for i, r in enumerate(ex.map(one, jobs)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 199:
                print("answered", i + 1, "/", len(jobs), "quota%", llm.check_quota(), flush=True)


# ---------- judging ----------
JUDGE_SYS = ("You grade answers to questions about a conversation. Given the question, the gold answer and a "
             "generated answer, reply CORRECT if the generated answer refers to the same thing as the gold answer "
             "(be generous with wording; for dates, the same date or period counts even if formatted differently), "
             "otherwise WRONG. Reply with one word.")


def f1(p, g):
    p, g = toks(p), toks(g)
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))


def judge(data):
    rows = [json.loads(l) for l in open(os.path.join(AOUT, "answers.jsonl"))]
    path = os.path.join(AOUT, "judged.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["qid"], r["cond"]) for r in map(json.loads, open(path))}
    rows = [r for r in rows if (r["qid"], r["cond"]) not in done]
    llm.check_quota()
    def one(r):
        v = llm.call(JUDGE_SYS, f"Question: {r['q']}\nGold answer: {r['gold']}\nGenerated answer: {r['pred']}",
                     effort="low", max_tokens=4000, tag="judge")
        return dict(r, correct=int("CORRECT" in v.upper() and "WRONG" not in v.upper()), f1=f1(r["pred"], r["gold"]))
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, rows):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


def report(data):
    keep = {q["qid"] for q in questions(data, ARGS.per_cat)}     # same question set for every condition
    rows = list({(r["qid"], r["cond"]): r for r in map(json.loads, open(os.path.join(AOUT, "judged.jsonl")))
                 if r["qid"] in keep}.values())      # two judge runs may overlap; keep one row per (qid, cond)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["cond"], r["cat"])].append(r); by[(r["cond"], 0)].append(r)
    print(f"{'condition':16s}" + "".join(f"{CAT.get(c, 'all'):>13s}" for c in [0, 1, 2, 3, 4]) + "      F1  ctx_chars")
    for cond in CONDS:
        if (cond, 0) not in by:
            continue
        line = f"{cond:16s}"
        for c in [0, 1, 2, 3, 4]:
            xs = by[(cond, c)]
            line += f"{100 * sum(x['correct'] for x in xs) / len(xs):9.1f} n={len(xs):<3d}" if xs else " " * 13
        a = by[(cond, 0)]
        line += f"{100 * sum(x['f1'] for x in a) / len(a):8.1f}  {sum(x['ctx_chars'] for x in a) / len(a):9.0f}"
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["build", "views", "ranks", "answer", "judge", "report"])
    p.add_argument("--per-cat", type=int, default=50)
    p.add_argument("--par", type=int, default=12)
    p.add_argument("--conds", default=",".join(CONDS))
    p.add_argument("--kinds", default="once,roll,rec_sess,rec_chunk", help="memories to build")
    ARGS = p.parse_args()
    ARGS.conds = ARGS.conds.split(",")
    ARGS.kinds = ARGS.kinds.split(",")
    data = json.load(open(DATA))
    os.makedirs(AOUT, exist_ok=True)
    print("arm", llm.ARM, "port", llm.PORT, "->", os.path.relpath(AOUT, ROOT), flush=True)
    {"build": build, "views": views, "ranks": ranks, "answer": answer, "judge": judge, "report": report}[ARGS.step](data)
