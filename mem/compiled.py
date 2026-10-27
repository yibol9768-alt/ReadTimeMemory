"""H32: compiled memory over time (the dynamic version of H31).

After every session the new ledger records are compressed into short notes (one per record, at most T characters,
grouped under the session's date, atomic.notes2) and appended; old notes are never rewritten from themselves.
When the notes no longer fit the budget, the level drops: T = SLACK * (budget - headers) / n_records, and EVERY note
is recompressed at the new T from its ledger record (not from its old note), so a fact is only ever derived from
the ledger, at most O(log n) times. Records shorter than T are kept verbatim (no call).

Measured like rehearse tavg (time-averaged retention of the EVAL_FACTS ledger facts already written, every STEP
sessions; same facts and questions), paired with rehearse's roll / prebuild rows of the same model.

Usage: python3 -m mem.compiled build|report [--convs 0,2,4,5,7,8] [--par 3]
"""
import argparse, collections, json, os, random
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, rehearse as H, atomic as A

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "compiled", llm.ARM)
TAVG = os.path.join(os.path.dirname(__file__), "..", "runs", "rehearse", llm.ARM, "tavg.jsonl")
STEP, SLACK, TMAX = 2, 0.8, 400


def session_block(s, rs, t, tag):
    """Notes of one session at level t; records already short enough are kept verbatim."""
    if not rs:
        return ""
    if all(len(r["text"]) <= t for r in rs):
        return f"[{s['date']}]\n" + "\n".join(r["text"] for r in rs)
    return A.notes2([s], [rs], t, tag)


def build_conv(ci, data):
    path = os.path.join(OUT, f"mem_{ci}.json")
    if os.path.exists(path):
        return json.load(open(path))
    ss = L.sessions(data[ci])
    recs = [L.parse_records(llm.call(L.REC_SYS, f"Session {s['n']}, date {s['date']}:\n" + "\n".join(
        L.show_turn(t, s["date"]) for t in s["turns"]), tag="drift_rec"), {t["id"] for t in s["turns"]}) for s in ss]
    t, blocks, mems, levels = TMAX, [], {}, []
    for i, (s, rs) in enumerate(zip(ss, recs), 1):
        blocks.append(session_block(s, rs, t, "cm_new"))
        if len("\n".join(b for b in blocks if b)) > L.BUDGET:            # level drop: recompress all from the ledger
            n = sum(len(r) for r in recs[:i])
            heads = sum(len(x["date"]) + 3 for x, r in zip(ss[:i], recs[:i]) if r)
            t = max(10, int(SLACK * (L.BUDGET - heads) / n) - 1)
            with ThreadPoolExecutor(4) as ex:
                blocks = list(ex.map(lambda j: session_block(ss[j], recs[j], t, "cm_level"), range(i)))
            levels.append({"after": i, "t": t, "records": n})
        mems[str(i)] = "\n".join(b for b in blocks if b)[:L.BUDGET]
    m = {"mems": mems, "levels": levels, "n_records": [len(r) for r in recs]}
    json.dump(m, open(path, "w"), ensure_ascii=False, indent=1)
    print("compiled", ci, "levels", [(x["after"], x["t"]) for x in levels],
          "final chars", len(mems[str(len(ss))]), flush=True)
    return m


def probe_conv(ci, data):
    path = os.path.join(OUT, "tavg.jsonl")
    done = {(r["conv"], r["after"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    m = build_conv(ci, data)
    ss = L.sessions(data[ci])
    recs = [L.parse_records(llm.call(L.REC_SYS, f"Session {s['n']}, date {s['date']}:\n" + "\n".join(
        L.show_turn(t, s["date"]) for t in s["turns"]), tag="drift_rec"), {t["id"] for t in s["turns"]}) for s in ss]
    pairs = [(si, r) for si, rs in enumerate(recs, 1) for r in rs]
    picked = random.Random(f"eval-{ci}").sample(pairs, min(H.EVAL_FACTS, len(pairs)))
    facts, born = [r for _, r in picked], [si for si, _ in picked]
    qs = H.probes_for(facts)
    rows = []
    for i in range(STEP, len(ss) + 1, STEP):
        if (ci, i) in done:
            continue
        idx = [j for j in range(len(facts)) if born[j] <= i]
        bad = set()
        for b in range(0, len(idx), 20):
            part = idx[b:b + 20]
            bad |= {part[j] for j in H.ask(m["mems"][str(i)], [facts[x] for x in part], [qs[x] for x in part])}
        rows.append({"conv": ci, "arm": "compiled", "after": i, "facts": idx, "kept": [int(j not in bad) for j in idx],
                     "chars": len(m["mems"][str(i)])})
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("compiled tavg", ci, len(rows), "probes", flush=True)


def build(data):
    llm.check_quota()
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(lambda ci: probe_conv(ci, data), ARGS.convs))
    report(data)


def report(data):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "tavg.jsonl"))]
    if os.path.exists(TAVG):
        rows += [json.loads(l) for l in open(TAVG)]
    rows = [r for r in rows if r["conv"] in ARGS.convs]
    convs_c = {r["conv"] for r in rows if r["arm"] == "compiled"}
    by, per = collections.defaultdict(list), collections.defaultdict(dict)
    for r in rows:
        if r["conv"] in convs_c:
            by[r["arm"]] += r["kept"]
            for f, k in zip(r["facts"], r["kept"]):
                per[r["arm"]][(r["conv"], r["after"], f)] = k
    print(f"time-averaged retention ({llm.ARM}), conversations {sorted(convs_c)}")
    for a in sorted(by, key=lambda a: sum(by[a]) / len(by[a])):
        c = per["compiled"]
        w = sum(c[k] > v for k, v in per[a].items() if k in c); l = sum(c[k] < v for k, v in per[a].items() if k in c)
        print(f"  {a:14s} {100 * sum(by[a]) / len(by[a]):5.1f}  n={len(by[a])}   compiled vs it {w}:{l}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "report"])
    ap.add_argument("--convs", default="0,2,4,5,7,8")
    ap.add_argument("--par", type=int, default=3)
    ARGS = ap.parse_args()
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    data = json.load(open(L.DATA))
    {"build": build, "report": report}[ARGS.step](data)
