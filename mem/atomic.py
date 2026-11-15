"""H31 static test: compiled memory. Instead of asking the model for a summary of the whole ledger (it underfills,
drops different facts every time, H28/H29), compress every ledger record on its own into a one-line note of at most
T = budget / n_records characters (batches of BATCH records per call) and pack all notes. Each note is derived from
its record only, so later updates never rewrite old text.

Evaluated like selrebuild (final checkpoint, the EVAL_FACTS ledger facts of seed eval-{ci}); compared with the
selrebuild candidates of the same conversation.

v2 (--v 2): notes grouped under one "[date]" header per session (no date in each note), budget per note
T = (budget - headers) / n_records; lines over T are asked again (shorter) instead of being cut.

Usage: python3 -m mem.atomic build|report [--v 2] [--convs 0,2,4,5,7,8] [--par 3]
"""
import argparse, json, os, random
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, rehearse as H, selrebuild as S

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "select", llm.ARM)
BATCH = 25
ATOM_SYS = ("You compress memory records into ultra-short notes. For each numbered record write one line: the number, "
            "a period, then a note of at most {t} characters that keeps the key detail someone could later ask about "
            "(who, what, when, and the specific name, number, place or object). Use first names, YYYY-MM-DD or YYYY-MM "
            "dates, symbols and abbreviations; drop filler and feelings unless they are the point. Output only the "
            "numbered lines.")


def notes(recs, t, tag):
    texts = [r["text"] for r in recs]
    out = []
    for b in range(0, len(texts), BATCH):
        part = texts[b:b + BATCH]
        lines = H.by_number(llm.call(ATOM_SYS.format(t=t), H.numbered(part), tag=tag), len(part))
        out += [(x or y)[:t] for x, y in zip(lines, part)]
    return out


ATOM2_SYS = ("You compress memory records into ultra-short notes. All records below come from one session dated "
             "{date}, so do not repeat that date. For each numbered record write one line: the number, a period, then "
             "a note of at most {t} characters that keeps the key detail someone could later ask about (who, what, "
             "and the specific name, number, place or object; other dates only if they differ from {date}). Use "
             "first names, symbols and abbreviations; drop filler and feelings unless they are the point. Output "
             "only the numbered lines.")


def notes2(ss, recs, t, tag):
    """v2: per session, one header and short notes; overlong notes are rewritten once, cut only after that."""
    blocks = []
    for s, rs in zip(ss, recs):
        if not rs:
            continue
        texts = [r["text"] for r in rs]
        lines = []
        for b in range(0, len(texts), BATCH):
            part = texts[b:b + BATCH]
            lines += [x or y for x, y in zip(H.by_number(llm.call(ATOM2_SYS.format(t=t, date=s["date"]),
                                                                  H.numbered(part), tag=tag), len(part)), part)]
        long = [i for i, x in enumerate(lines) if len(x) > t]
        if long:
            redo = H.by_number(llm.call(ATOM2_SYS.format(t=int(t * 0.8), date=s["date"]),
                                        H.numbered([lines[i] for i in long]), tag=tag + "_redo"), len(long))
            for i, x in zip(long, redo):
                if x and len(x) < len(lines[i]):
                    lines[i] = x
        blocks.append(f"[{s['date']}]\n" + "\n".join(x[:t] for x in lines))
    return "\n".join(blocks)


def build_conv(ci, data):
    path = os.path.join(OUT, f"atomic{'' if ARGS.v == 1 else ARGS.v}_{ci}.json")
    if os.path.exists(path):
        return
    ss = L.sessions(data[ci])
    recs = [L.parse_records(llm.call(L.REC_SYS, f"Session {s['n']}, date {s['date']}:\n" + "\n".join(
        L.show_turn(t, s["date"]) for t in s["turns"]), tag="drift_rec"), {t["id"] for t in s["turns"]}) for s in ss]
    pairs = [(si, r) for si, rs in enumerate(recs, 1) for r in rs]
    ev = [r for _, r in random.Random(f"eval-{ci}").sample(pairs, min(H.EVAL_FACTS, len(pairs)))]
    ev_q = H.probes_for(ev)
    flat = [r for rs in recs for r in rs]
    if ARGS.v == 1:
        t = L.BUDGET // len(flat) - 1                       # one newline per note
        mem = "\n".join(notes(flat, t, "atomic"))[:L.BUDGET]
    else:
        heads = sum(len(s["date"]) + 3 for s, rs in zip(ss, recs) if rs)
        t = (L.BUDGET - heads) // len(flat) - 1
        mem = notes2(ss, recs, t, "atomic2")[:L.BUDGET]
    res = {"t": t, "n": len(flat), "mem": mem, "chars": len(mem), "eval": S.score(mem, ev, ev_q)}
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"atomic{ARGS.v}", ci, "T", t, "notes", len(flat), "chars", len(mem), "eval", sum(res["eval"]), flush=True)


def build(data):
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(lambda ci: build_conv(ci, data), ARGS.convs))
    report(data)


def report(data):
    tot = {}
    for ci in ARGS.convs:
        a = os.path.join(OUT, f"atomic_{ci}.json")
        if not os.path.exists(a):
            continue
        d = json.load(open(os.path.join(OUT, f"sel_{ci}.json")))
        rows = {"base": d["cands"]["base"]["eval"], "merge5": d["merge"]["merge5"]["eval"],
                "atomic": json.load(open(a))["eval"]}
        a2 = os.path.join(OUT, f"atomic2_{ci}.json")
        if os.path.exists(a2):
            rows["atomic2"] = json.load(open(a2))["eval"]
        for k in ("chunk2", "chunk4", "chunk8"):
            if k in d.get("chunk", {}):
                rows[k] = d["chunk"][k]["eval"]
        for k, v in rows.items():
            tot.setdefault(k, []).append(v)
    for k, rows in tot.items():
        xs = [x for r in rows for x in r]
        b = [x for r in tot["base"][:len(rows)] for x in r]
        print(f"  {k:8s} {100 * sum(xs) / len(xs):5.1f}  vs base {sum(x > y for x, y in zip(xs, b))}:"
              f"{sum(y > x for x, y in zip(xs, b))}  n={len(xs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "report"])
    ap.add_argument("--convs", default="0,2,4,5,7,8")
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--v", type=int, default=1)
    ARGS = ap.parse_args()
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    data = json.load(open(L.DATA))
    {"build": build, "report": report}[ARGS.step](data)
