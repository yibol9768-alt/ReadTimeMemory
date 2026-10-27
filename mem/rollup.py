"""WP3, first method: write-time roll-up records on top of Lychee-lite.

WP1/WP2 showed that multi-hop questions fail because the facts of one kind (gifts, trips, classes, pets' tricks...)
are spread over many records from different sessions: a weak reader does not put them together, a strong reader
does not even get all of them, since retrieval brings back one or two. Lychee carries only a disambiguation state
between segments; nothing ever collects the scattered facts.

Here, after encoding, one LLM call per person reads all of that person's records (time-ordered, with source ids) and
writes roll-up records: every recurring kind of fact gathered into one self-contained record listing all items with
dates and sources, and multi-step threads as timelines (start -> milestones -> outcome, with durations). The kinds
are chosen by the model from the records, not from the benchmark questions. Roll-ups join the record pool; retrieval,
budget and reader are unchanged (condition "lychee_roll" vs "lychee"). Roll-ups are not expanded into source turns
(their text already lists the items, and expansion would eat the budget).

Questions: the 80-question sample, plus held-out multi-hop questions (in the 50-per-category sample but not in the
20-per-category one), which were not looked at while designing this.

Usage: python3 -m mem.rollup build|answer|report [--par 6]
"""
import argparse, collections, json, os
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, lychee as Y, retrieve as R

OUT = Y.OUT
MAX_IN = 120_000       # characters of records per call; longer lists are split by time and rolled up per part

ROLL_SYS = (
    "You maintain the long-term memory of a long conversation between two people. You are given every memory "
    "record about one person, in time order, each with its date and source ids. Write roll-up records that gather "
    "facts which are scattered across many records:\n"
    "- collections: any kind of thing that recurs (for example things they own, received or gave, places visited, "
    "activities, classes, hobbies tried or considered, pets and what they do, people they met, foods, books, "
    "games, achievements, health events). One record per kind, listing every item with its date.\n"
    "- threads: any goal, project or situation that unfolds over several sessions, as a timeline from start to "
    "latest state, with dates and the elapsed time between key steps.\n"
    "- counts and changes: how many times something happened, and preferences or plans that changed over time.\n"
    "Only roll up what has at least two records behind it; never invent items. Each record is self-contained (full "
    "names, absolute dates).\n"
    "Output one JSON object per line, nothing else:\n"
    '{"type": "collection|thread|count", "text": "...", "entities": ["..."], "topic": "...", '
    '"time": "YYYY-MM-DD..YYYY-MM-DD", "sources": ["D1:3", ...]}')


def people(conv):
    c = conv["conversation"]
    return [c["speaker_a"], c["speaker_b"]]


def line(r):
    return f"[{r['time'] or 'no date'}] {r['text']} (sources: {', '.join(r['sources'])})"


def roll(ci, data, out=None, sess=None, who_list=None):
    out_dir = out or OUT
    path = os.path.join(out_dir, f"roll_{ci}.json")
    if os.path.exists(path):
        return
    recs = json.load(open(os.path.join(out_dir, f"mem_{ci}.json")))["records"]
    valid = {t["id"] for s in (sess or L.sessions)(data[ci]) for t in s["turns"]}
    out = []
    for who in who_list or people(data[ci]):
        mine = recs if who == "*" else [r for r in recs if who in r["entities"] or who.split()[0] in r["text"]]
        parts, cur = [], []
        for r in mine:                     # records are already in conversation order
            if cur and sum(len(line(x)) for x in cur) + len(line(r)) > MAX_IN:
                parts.append(cur); cur = []
            cur.append(r)
        parts.append(cur)
        for k, part in enumerate(parts):
            body = f"Person: {'the user' if who == '*' else who}\nRecords ({len(part)}):\n" + "\n".join(line(r) for r in part)
            rs, _ = Y.parse(llm.call(ROLL_SYS, body, max_tokens=32000, tag="roll"), valid)
            out += [dict(r, type="rollup:" + (r["type"] or "collection")) for r in rs]
    json.dump({"rollups": out}, open(path, "w"), ensure_ascii=False, indent=1)
    print("rolled", ci, len(out), "roll-ups", flush=True)


def question_set(data):
    base = L.questions(data, 20)
    seen = {q["qid"] for q in base}
    held = [q for q in L.questions(data, 50) if q["category"] == 1 and q["qid"] not in seen]
    return base, held


def build(data):
    base, held = question_set(data)
    convs = sorted({q["conv"] for q in base + held})
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(convs))) as ex:
        list(ex.map(lambda ci: roll(ci, data), convs))


def answer(data):
    base, held = question_set(data)
    qs = base + held
    convs = {q["conv"] for q in qs}
    mems = {ci: json.load(open(os.path.join(OUT, f"mem_{ci}.json")))["records"] for ci in convs}
    rolls = {ci: json.load(open(os.path.join(OUT, f"roll_{ci}.json")))["rollups"] for ci in convs}
    pools = {"lychee": mems, "lychee_roll": {ci: mems[ci] + rolls[ci] for ci in convs}}
    sss = {ci: L.sessions(data[ci]) for ci in convs}
    llm.check_quota()
    with ThreadPoolExecutor(ARGS.par) as ex:
        plans = list(ex.map(lambda q: Y.plan(q["question"]), qs))
    R.precompute_ranks([[r["text"] for r in m] for pool in pools.values() for m in pool.values()],
                       [([r["text"] for r in pools[c][q["conv"]]], x)
                        for c in pools for q, p in zip(qs, plans) for x in p["queries"]])
    path = os.path.join(L.AOUT, "answers.jsonl")
    done = {(r["qid"], r["cond"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(q, p, c) for q, p in zip(qs, plans) for c in pools if (q["qid"], c) not in done]

    def one(job):
        q, p, c = job
        ctx = Y.context(q["question"], p, pools[c][q["conv"]], sss[q["conv"]])
        a = llm.call(L.ANS_SYS, f"Memory:\n{ctx}\n\nQuestion: {q['question']}", tag="ans_lychee")
        return {"qid": q["qid"], "cond": c, "cat": q["category"], "q": q["question"],
                "gold": str(q["answer"]), "pred": a.strip(), "ctx_chars": len(ctx)}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


def report(data):
    base, held = question_set(data)
    rows = {(r["qid"], r["cond"]): r for r in map(json.loads, open(os.path.join(L.AOUT, "judged.jsonl")))}
    for name, qs in (("80-question sample", base), ("held-out multi-hop", held)):
        print(f"\n{name} ({len(qs)} questions)")
        for c in ("lychee", "lychee_roll"):
            by = collections.defaultdict(list)
            for q in qs:
                r = rows.get((q["qid"], c))
                if r:
                    by[0].append(r["correct"]); by[q["category"]].append(r["correct"])
            print(f"  {c:12s}" + "".join(f"  {L.CAT.get(k, 'all')} {100 * sum(v) / len(v):5.1f} n={len(v)}"
                                         for k, v in sorted(by.items())))
        pair = [(rows[(q["qid"], "lychee")]["correct"], rows[(q["qid"], "lychee_roll")]["correct"]) for q in qs
                if (q["qid"], "lychee") in rows and (q["qid"], "lychee_roll") in rows]
        if pair:
            print(f"  paired: roll-up fixes {sum(b > a for a, b in pair)}, breaks {sum(a > b for a, b in pair)}"
                  f" of {len(pair)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["build", "answer", "report"])
    p.add_argument("--par", type=int, default=6)
    ARGS = p.parse_args()
    os.makedirs(L.AOUT, exist_ok=True)
    data = json.load(open(L.DATA))
    {"build": build, "answer": answer, "report": report}[ARGS.step](data)
