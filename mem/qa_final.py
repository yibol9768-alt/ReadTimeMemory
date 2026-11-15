"""LoCoMo's own questions asked of the final-checkpoint memories of selrebuild / atomic (H31 check: does compiled
memory win only on ledger probes, which are built from the same records, or also on the benchmark's questions?)

Memories per conversation: base (rebuild), merge5, atomic (v1), atomic2 (v2) when present. Questions: PER_CONV
LoCoMo questions per conversation from the categories of locomo_exp.CAT (seed 0), same set for every memory.
Answer prompt and judge are locomo_exp's (ANS_SYS, JUDGE_SYS).

Usage: python3 -m mem.qa_final run|report [--convs 0,2,4,5,7,8] [--per-conv 50] [--par 6]
"""
import argparse, collections, json, os, random
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "select", llm.ARM)


def mems(ci):
    d = json.load(open(os.path.join(OUT, f"sel_{ci}.json")))
    out = {"base": d["cands"]["base"]["mem"], "merge5": d["merge"]["merge5"]["mem"]}
    for k in ("atomic", "atomic2"):
        p = os.path.join(OUT, f"{k}_{ci}.json")
        if os.path.exists(p):
            out[k] = json.load(open(p))["mem"]
    return out


def qs_for(conv, ci):
    qs = [dict(q, qid=f"{ci}:{j}") for j, q in enumerate(conv["qa"]) if q.get("category") in L.CAT and "answer" in q]
    random.Random(0).shuffle(qs)
    return qs[:ARGS.per_conv]


def run(data):
    path = os.path.join(OUT, "qa.jsonl")
    done = {(r["qid"], r["mem"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(q, name, m) for ci in ARGS.convs for name, m in mems(ci).items() for q in qs_for(data[ci], ci)
            if (q["qid"], name) not in done]
    llm.check_quota()

    def one(job):
        q, name, m = job
        a = llm.call(L.ANS_SYS, f"Memory:\n{m}\n\nQuestion: {q['question']}", tag="qaf_ans").strip()
        v = llm.call(L.JUDGE_SYS, f"Question: {q['question']}\nGold answer: {q['answer']}\nGenerated answer: {a}",
                     effort="low", max_tokens=4000, tag="qaf_judge")
        return {"qid": q["qid"], "cat": q["category"], "mem": name, "q": q["question"], "gold": str(q["answer"]),
                "pred": a, "correct": int("CORRECT" in v.upper() and "WRONG" not in v.upper())}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
    report(data)


def report(data):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "qa.jsonl"))]
    keep = {q["qid"] for ci in ARGS.convs for q in qs_for(data[ci], ci)}
    rows = [r for r in rows if r["qid"] in keep]
    by, per = collections.defaultdict(list), collections.defaultdict(dict)
    for r in rows:
        by[(r["mem"], 0)].append(r["correct"]); by[(r["mem"], r["cat"])].append(r["correct"])
        per[r["mem"]][r["qid"]] = r["correct"]
    names = sorted({r["mem"] for r in rows}, key=lambda n: ["base", "merge5", "atomic", "atomic2"].index(n))
    print(f"LoCoMo questions on the final memories ({llm.ARM})")
    print(f"{'memory':9s}" + "".join(f"{L.CAT.get(c, 'all'):>13s}" for c in [0, 1, 2, 3, 4]) + "   vs base")
    for n in names:
        b = per["base"]
        w = sum(per[n][q] > b[q] for q in per[n] if q in b); l = sum(per[n][q] < b[q] for q in per[n] if q in b)
        print(f"{n:9s}" + "".join(f"{100 * sum(by[(n, c)]) / len(by[(n, c)]):9.1f} n={len(by[(n, c)]):<3d}"
                                  if by[(n, c)] else f"{'':13s}" for c in [0, 1, 2, 3, 4]) + f"   {w}:{l}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "report"])
    ap.add_argument("--convs", default="0,2,4,5,7,8")
    ap.add_argument("--per-conv", type=int, default=50)
    ap.add_argument("--par", type=int, default=6)
    ARGS = ap.parse_args()
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    data = json.load(open(L.DATA))
    {"run": run, "report": report}[ARGS.step](data)
