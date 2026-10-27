"""Forgetting curve of a self-rewriting memory (the phenomenon behind WP4/WP5).

The roll memory of mem/drift.py is replayed session by session from the cache (same prompts, no new calls), so we
have the memory after every session. From the ledger (per-session records, also cached) we sample facts; each fact
is turned into a question once, and asked of the memory every STEP sessions from the session it was written in to
the end. Output: for every (fact, lag) whether the memory still answers it, where lag = sessions since the fact
was written (= number of rewrites it has gone through). The report gives retention by lag, which is the curve.

Usage: python3 -m mem.curve build|report [--convs 0,2,4,5,7,8] [--facts 40] [--par 3]
"""
import argparse, collections, json, os, random
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, rehearse as H

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "curve", llm.ARM)
STEP = 2
BATCH = 20


def replay(ss):
    """Memory after each session, from drift.py's roll (cache hits only)."""
    real = llm.urllib.request.urlopen

    def offline(*a, **k):
        raise RuntimeError("cache miss while replaying the roll memory")
    llm.urllib.request.urlopen = offline
    try:
        mem, out = "", []
        for s in ss:
            mem = L.within_budget(f"Your current memory of the earlier sessions:\n\n{mem or '(empty)'}\n\n"
                                  f"The next session:\n\n{L.show_session(s)}\n\n"
                                  f"Write the updated memory of everything so far (at most {L.BUDGET} characters).",
                                  "drift_roll")
            out.append(mem)
        return out
    finally:
        llm.urllib.request.urlopen = real


def build_conv(ci, data):
    path = os.path.join(OUT, f"curve_{ci}.json")
    if os.path.exists(path):
        return
    ss = L.sessions(data[ci])
    mems = replay(ss)

    def records(s):              # drift.py's prompt: cache hits
        body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(L.show_turn(t, s["date"]) for t in s["turns"])
        return L.parse_records(llm.call(L.REC_SYS, body, tag="drift_rec"), {t["id"] for t in s["turns"]})
    recs = [records(s) for s in ss]
    pool = [(i, r) for i, rs in enumerate(recs) for r in rs]
    facts = random.Random(ci).sample(pool, min(ARGS.facts, len(pool)))
    texts = [r["text"] for _, r in facts]
    qs = []
    for b in range(0, len(texts), BATCH):
        part = texts[b:b + BATCH]
        qs += H.by_number(llm.call(H.QGEN_SYS, H.numbered(part), effort="low", max_tokens=4000, tag="cv_qgen"), len(part))
    # ask at every STEP sessions from the writing session to the end (always including the last session)
    asks = collections.defaultdict(list)          # session index -> fact indices
    for f, (i, _) in enumerate(facts):
        for j in sorted(set(range(i, len(ss), STEP)) | {len(ss) - 1}):
            asks[j].append(f)
    rows = []

    def probe(j):
        out = []
        for b in range(0, len(asks[j]), BATCH):
            fs = asks[j][b:b + BATCH]
            ans = H.by_number(llm.call(H.QANS_SYS, f"Memory notes:\n{mems[j]}\n\nQuestions:\n{H.numbered([qs[f] for f in fs])}",
                                       effort="low", max_tokens=4000, tag="cv_qans"), len(fs))
            items = [f"Fact: {texts[f]}\n   Question: {qs[f]}\n   Answer: {a}" for f, a in zip(fs, ans)]
            grade = H.by_number(llm.call(H.QGRADE_SYS, H.numbered(items), effort="low", max_tokens=4000,
                                         tag="cv_qgrade"), len(fs))
            out += [{"fact": f, "written": facts[f][0] + 1, "asked": j + 1, "lag": j - facts[f][0],
                     "kept": int("RIGHT" in g.upper())} for f, g in zip(fs, grade)]
        return out
    with ThreadPoolExecutor(2) as ex:
        for part in ex.map(probe, sorted(asks)):
            rows += part
    json.dump({"facts": texts, "questions": qs, "n_sessions": len(ss), "rows": rows}, open(path, "w"),
              ensure_ascii=False, indent=1)
    print("curve", ci, len(facts), "facts", len(rows), "probes", flush=True)


def build(data):
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(lambda ci: build_conv(ci, data), ARGS.convs))


def report(data):
    by, first = collections.defaultdict(list), collections.defaultdict(list)
    for ci in ARGS.convs:
        d = json.load(open(os.path.join(OUT, f"curve_{ci}.json")))
        for r in d["rows"]:
            by[min(r["lag"] // 3 * 3, 24)].append(r["kept"])
            if r["lag"] == 0:
                first[ci].append(r["kept"])
    print("retention of ledger facts in the rolling memory, by sessions since the fact was written")
    for lag in sorted(by):
        xs = by[lag]
        print(f"  lag {lag:2d}-{lag + 2:<2d}{'+' if lag == 24 else ' '} {100 * sum(xs) / len(xs):5.1f}%  n={len(xs)}")
    print("  written-and-kept right away (lag 0):",
          ", ".join(f"conv {c}: {100 * sum(v) / len(v):.0f}%" for c, v in first.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["build", "report"])
    p.add_argument("--convs", default="0,2,4,5,7,8")
    p.add_argument("--facts", type=int, default=40)
    p.add_argument("--par", type=int, default=3)
    ARGS = p.parse_args()
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(L.DATA))
    {"build": build, "report": report}[ARGS.step](data)
