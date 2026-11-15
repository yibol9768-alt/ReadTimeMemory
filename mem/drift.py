"""Drift experiment: does a memory that is rewritten from itself forget early facts as sessions pile up, while a
memory rebuilt from an append-only ledger of records does not?

Both arms produce a plain-text memory of at most BUDGET characters with the same instructions (SUM_SYS):
  roll     after every session: old memory + new session -> new memory      (rewrite from itself)
  rebuild  at each checkpoint: all records extracted so far -> memory      (rebuild from the ledger)
Records are extracted once per session (REC_SYS) and never edited. Questions are LoCoMo questions whose evidence
lies entirely in the first third of the sessions; each is asked of both memories at three checkpoints
(after 1/3, 2/3 and all sessions). Retention = share answered correctly. Small sample by design.

Usage: python3 -m mem.drift build|answer|report [--convs 2,4,7] [--per-conv 12]
"""
import argparse, collections, json, os, random, re
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "drift", llm.ARM)


def checkpoints(n):
    return [-(-n // 3), -(-2 * n // 3), n]


def early_questions(conv, n, k, seed=0):
    third = checkpoints(n)[0]
    qs = []
    for j, q in enumerate(conv["qa"]):
        ev = [re.match(r"D(\d+):", e) for e in q.get("evidence") or []]
        if q.get("category") in L.CAT and "answer" in q and ev and all(ev) and all(int(m.group(1)) <= third for m in ev):
            qs.append(dict(q, qid=j))
    random.Random(seed).shuffle(qs)
    return qs[:k]


def build_conv(ci, data):
    path = os.path.join(OUT, f"mem_{ci}.json")
    if os.path.exists(path):
        return json.load(open(path))
    ss = L.sessions(data[ci])
    cps = checkpoints(len(ss))

    def records(s):
        body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(L.show_turn(t, s["date"]) for t in s["turns"])
        return L.parse_records(llm.call(L.REC_SYS, body, tag="drift_rec"), {t["id"] for t in s["turns"]})

    def roll():                       # sequential by nature
        mem, out = "", {}
        for i, s in enumerate(ss, 1):
            mem = L.within_budget(f"Your current memory of the earlier sessions:\n\n{mem or '(empty)'}\n\n"
                                  f"The next session:\n\n{L.show_session(s)}\n\n"
                                  f"Write the updated memory of everything so far (at most {L.BUDGET} characters).",
                                  "drift_roll")
            if i in cps:
                out[i] = mem
        return out

    with ThreadPoolExecutor(2) as ex:             # roll + one records worker per conversation
        roll_f = ex.submit(roll)
        recs = list(ex.map(records, ss))              # recs[i] = records of session i+1
        rebuild = {}
        for k in cps:
            ledger = [r["text"] for rs in recs[:k] for r in rs]
            rebuild[k] = L.within_budget(
                "Memory records extracted so far from the conversation (in time order; each is self-contained):\n\n"
                + "\n".join("- " + t for t in ledger)
                + f"\n\nWrite the memory of everything so far (at most {L.BUDGET} characters).", "drift_rebuild")
        m = {"checkpoints": cps, "roll": roll_f.result(), "rebuild": rebuild,
             "n_records": [len(rs) for rs in recs]}
    m["roll"] = {str(k): v for k, v in m["roll"].items()}
    m["rebuild"] = {str(k): v for k, v in m["rebuild"].items()}
    json.dump(m, open(path, "w"), ensure_ascii=False, indent=1)
    print("built", ci, "checkpoints", cps, "records", sum(m["n_records"]), flush=True)
    return m


def build(data):
    llm.check_quota()
    with ThreadPoolExecutor(len(ARGS.convs)) as ex:
        list(ex.map(lambda ci: build_conv(ci, data), ARGS.convs))


def answer(data):
    path = os.path.join(OUT, "judged.jsonl")
    done = {(r["conv"], r["qid"], r["arm"], r["k"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = []
    for ci in ARGS.convs:
        m = json.load(open(os.path.join(OUT, f"mem_{ci}.json")))
        for q in early_questions(data[ci], len(m["n_records"]), ARGS.per_conv):
            for arm in ("roll", "rebuild"):
                for k in m["checkpoints"]:
                    if (ci, q["qid"], arm, k) not in done:
                        jobs.append((ci, q, arm, k, m[arm][str(k)]))

    def one(job):
        ci, q, arm, k, mem = job
        a = llm.call(L.ANS_SYS, f"Memory:\n{mem}\n\nQuestion: {q['question']}", tag="drift_ans").strip()
        v = llm.call(L.JUDGE_SYS, f"Question: {q['question']}\nGold answer: {q['answer']}\nGenerated answer: {a}",
                     effort="low", max_tokens=4000, tag="drift_judge")
        return {"conv": ci, "qid": q["qid"], "cat": q["category"], "arm": arm, "k": k, "q": q["question"],
                "gold": str(q["answer"]), "pred": a, "mem_chars": len(mem),
                "correct": int("CORRECT" in v.upper() and "WRONG" not in v.upper())}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


def report(data):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "judged.jsonl"))]
    by = collections.defaultdict(list)
    for r in rows:
        pos = json.load(open(os.path.join(OUT, f"mem_{r['conv']}.json")))["checkpoints"].index(r["k"])
        by[(r["arm"], pos)].append(r["correct"])
    print("early-fact retention (% correct), checkpoints after 1/3, 2/3, all sessions")
    for arm in ("roll", "rebuild"):
        print(f"{arm:8s}" + "".join(f"{100 * sum(by[(arm, p)]) / len(by[(arm, p)]):8.1f} n={len(by[(arm, p)]):<3d}"
                                    for p in range(3) if by[(arm, p)]))
    # paired: same question, same checkpoint
    pair = collections.defaultdict(dict)
    for r in rows:
        pair[(r["conv"], r["qid"], r["k"])][r["arm"]] = r["correct"]
    d = [v["rebuild"] - v["roll"] for v in pair.values() if len(v) == 2]
    print(f"rebuild - roll over {len(d)} paired asks: {100 * sum(d) / max(1, len(d)):+.1f} points "
          f"(rebuild-only correct {d.count(1)}, roll-only correct {d.count(-1)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["build", "answer", "report"])
    p.add_argument("--convs", default="2,4,7")
    p.add_argument("--per-conv", type=int, default=12)
    p.add_argument("--par", type=int, default=8)
    p.add_argument("--budget", type=int, default=L.BUDGET, help="memory budget in characters (default 8000)")
    ARGS = p.parse_args()
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    if ARGS.budget != L.BUDGET:      # tighter memory: same prompts with the new limit, separate output directory
        L.SUM_SYS = L.SUM_SYS.replace(str(L.BUDGET), str(ARGS.budget))
        L.BUDGET = ARGS.budget
        OUT = OUT + f"_b{ARGS.budget}"
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(L.DATA))
    {"build": build, "answer": answer, "report": report}[ARGS.step](data)
