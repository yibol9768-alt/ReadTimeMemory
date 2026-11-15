"""H34: anchored records. Lychee-lite records that also keep the user's own words.

Error analysis of the BEAM-100K dev baselines (PROGRESS 10-15 20:55, 22:45): the record pipeline loses most of its
points at write time, and H33 showed that handing the raw source turns back to the reader hurts (-9.7). Looking at
the losses, records paraphrase away exactly what the rubric checks: the number or time as stated ("3 PM", "3 years
now"), the user's own framing, both sides of a contradiction. Here the encoder writes the same records (same
segments, same state chaining, same prompt otherwise) plus, per record, a short verbatim quote of the user's (or
assistant's) words it rests on. Retrieval ranks on the record text only; the quote is only shown to the reader.

Ablation from one encoding: anc (records + quotes) vs anc_noq (same records, quotes hidden), both through the
lychee context (8K, mem/beam.py lyx with lychee's settings). anc_noq vs lychee measures re-encoding noise.

Usage: V3_BEAM_SPLIT=100k|lme|1m python3 -m mem.anchor build [--par 6]
"""
import argparse, json, os, shutil
from concurrent.futures import ThreadPoolExecutor
from . import llm, lychee as Y, beam as B

QUOTE_MAX = 240
ENC_SYS = Y.ENC_SYS.replace(
    'Output one JSON object per line, nothing else:\n',
    "Also give, for each record, a quote: the shortest verbatim span (at most 200 characters, copied exactly from "
    "one turn) that the record rests on, preferably the user's own words; keep numbers, times, durations and "
    "names exactly as said there (e.g. \"I've been with Douglas for 3 years now\").\n"
    'Output one JSON object per line, nothing else:\n').replace(
    '"sources": ["D1:3", ...]}', '"sources": ["D1:3", ...], "quote": "..."}')
assert ENC_SYS != Y.ENC_SYS and '"quote"' in ENC_SYS


def parse(text, valid):
    recs, state = Y.parse(text, valid)
    quotes = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("text"):
                quotes.append(str(r.get("quote") or "")[:QUOTE_MAX])
    for r, q in zip(recs, quotes):            # Y.parse keeps the same lines in the same order
        r["quote"] = q
    return recs, state


def encode(ci, convs, starts, out):
    path = os.path.join(out, f"mem_{ci}.json")
    if os.path.exists(path):
        return
    flat = [(s, t) for s in convs[ci]["sessions"] for t in s["turns"]]
    bounds = starts + [len(flat)]
    state, recs = "(empty)", []
    for a, b in zip(bounds, bounds[1:]):
        piece = flat[a:b]
        s = piece[0][0]
        body = (f"Disambiguation state:\n{state}\n\nNext segment (session {s['n']}, date {s['date']}):\n"
                + "\n".join(Y.show(t, s["date"]) for _, t in piece))
        rs, st = parse(llm.call(ENC_SYS, body, tag="anc_enc"), {t["id"] for _, t in piece})
        recs += rs
        state = st or state
    json.dump({"records": recs, "segments": len(bounds) - 1}, open(path, "w"), ensure_ascii=False, indent=1)
    print("anchored", ci, "segments", len(bounds) - 1, "records", len(recs),
          "with quote", sum(bool(r["quote"]) for r in recs), flush=True)


def build(_):
    convs = B.load()
    cis = B.picked(convs)
    out = os.path.join(B.OUT, "anchor"); os.makedirs(out, exist_ok=True)
    seg_src = os.path.join(B.OUT, "lychee", "segments.json")     # identical segmentation to lychee
    if not os.path.exists(os.path.join(out, "segments.json")):
        shutil.copy(seg_src, os.path.join(out, "segments.json"))
    seg = json.load(open(os.path.join(out, "segments.json")))
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(cis))) as ex:
        list(ex.map(lambda ci: encode(ci, convs, seg[str(ci)], out), cis))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build"])
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--convs", type=int, default=6)
    ARGS = ap.parse_args()
    B.ARGS = ARGS
    {"build": build}[ARGS.step](None)
