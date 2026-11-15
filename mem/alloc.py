"""Where should the strong model go: writing the memory or reading it?  2x2 on Lychee-lite.
  writer  weak = Plus luna (fingerprinted gpt-5.6-luna)  | strong = gpt-6.1-sol on Pro
  reader  the same two models; the reader does the query plan and the answer
Memory format, retrieval, budget and the judge (weak model, low effort, same for all four cells) are fixed.
Small sample: conversations 0, 1, 8 (the shortest) and their questions from the 50-per-category set.

Usage: python3 -m mem.alloc build|answer|report
"""
import argparse, collections, json, os
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, lychee as Y, retrieve as R

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "runs", "alloc")
MODELS = {"weak": ("gpt-6-luna", "18319"), "strong": ("gpt-6.1-sol", "18318")}
MEM_DIR = {"weak": os.path.join(ROOT, "runs", "lychee", "plus56"), "strong": os.path.join(ROOT, "runs", "lychee", "sol")}
CONVS = [0, 1, 8]


def build(data):
    os.makedirs(MEM_DIR["strong"], exist_ok=True)
    seg = Y.segments(data, CONVS)
    m, p = MODELS["strong"]
    with ThreadPoolExecutor(len(CONVS)) as ex:          # the weak memories come from the main Lychee-lite run
        list(ex.map(lambda ci: Y.encode(ci, data, seg[ci], out=MEM_DIR["strong"], model=m, port=p), CONVS))


def answer(data):
    qs = [q for q in L.questions(data, 50) if q["conv"] in CONVS]
    mems = {(w, ci): json.load(open(os.path.join(MEM_DIR[w], f"mem_{ci}.json")))["records"] for w in MEM_DIR for ci in CONVS}
    sss = {ci: L.sessions(data[ci]) for ci in CONVS}
    with ThreadPoolExecutor(8) as ex:
        plans = {r: list(ex.map(lambda q: Y.plan(q["question"], *MODELS[r]), qs)) for r in MODELS}
    pairs = [([x["text"] for x in mems[(w, q["conv"])]], x) for w in MEM_DIR for r in MODELS
             for q, p in zip(qs, plans[r]) for x in p["queries"]]
    R.precompute_ranks([[x["text"] for x in v] for v in mems.values()], pairs)
    path = os.path.join(OUT, "judged.jsonl")
    done = {(r["qid"], r["writer"], r["reader"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(q, w, r, plans[r][i]) for i, q in enumerate(qs) for w in MEM_DIR for r in MODELS
            if (q["qid"], w, r) not in done]

    def one(job):
        q, w, r, p = job
        ctx = Y.context(q["question"], p, mems[(w, q["conv"])], sss[q["conv"]])
        a = llm.call(L.ANS_SYS, f"Memory:\n{ctx}\n\nQuestion: {q['question']}", tag="alloc_ans", model=MODELS[r][0],
                     port=MODELS[r][1]).strip()
        v = llm.call(L.JUDGE_SYS, f"Question: {q['question']}\nGold answer: {q['answer']}\nGenerated answer: {a}",
                     effort="low", max_tokens=4000, tag="alloc_judge", model=MODELS["weak"][0], port=MODELS["weak"][1])
        return {"qid": q["qid"], "cat": q["category"], "writer": w, "reader": r, "q": q["question"],
                "gold": str(q["answer"]), "pred": a, "correct": int("CORRECT" in v.upper() and "WRONG" not in v.upper()),
                "f1": L.f1(a, str(q["answer"]))}
    with ThreadPoolExecutor(8) as ex, open(path, "a") as f:
        for row in ex.map(one, jobs):
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()


def report(data):
    rows = list({(r["qid"], r["writer"], r["reader"]): r for r in map(json.loads, open(os.path.join(OUT, "judged.jsonl")))}.values())
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["writer"], r["reader"])].append(r)
    print(f"{'writer/reader':16s}{'acc':>8s}{'F1':>8s}   n   by category (multi-hop, temporal, open, single)")
    for w in ("weak", "strong"):
        for r in ("weak", "strong"):
            xs = by[(w, r)]
            if not xs:
                continue
            cats = [[x["correct"] for x in xs if x["cat"] == c] for c in (1, 2, 3, 4)]
            print(f"{w + '/' + r:16s}{100 * sum(x['correct'] for x in xs) / len(xs):8.1f}"
                  f"{100 * sum(x['f1'] for x in xs) / len(xs):8.1f}{len(xs):5d}   "
                  + "  ".join(f"{100 * sum(c) / len(c):5.1f}" if c else "   - " for c in cats))
    # paired effects over the same questions
    key = collections.defaultdict(dict)
    for r in rows:
        key[r["qid"]][(r["writer"], r["reader"])] = r["correct"]
    full = [v for v in key.values() if len(v) == 4]
    if full:
        wr = sum(v[("strong", "weak")] + v[("strong", "strong")] - v[("weak", "weak")] - v[("weak", "strong")] for v in full) / (2 * len(full))
        rd = sum(v[("weak", "strong")] + v[("strong", "strong")] - v[("weak", "weak")] - v[("strong", "weak")] for v in full) / (2 * len(full))
        print(f"over {len(full)} questions with all four cells: strong writer +{100 * wr:.1f} points, strong reader +{100 * rd:.1f} points")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["build", "answer", "report"])
    ARGS = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(L.DATA))
    {"build": build, "answer": answer, "report": report}[ARGS.step](data)
