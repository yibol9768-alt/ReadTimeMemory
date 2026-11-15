"""Error analysis of the BEAM-100K dev baselines (PROTOCOL.md step 3): where does each method lose its points?

For every judged answer (mem/beam.py judged.jsonl) and every rubric item that did not get full credit, ask whether the
information that item needs is present in the context the reader was given:
  reader     present in the context, but the answer missed it      -> the reader did not use what it had
  context    absent from the context                               -> lost before reading (write or retrieval)
For lychee (records are on disk) a context miss is split further by asking whether any record holds the information:
  retrieval  some record has it, but it was not retrieved
  write      no record has it (never written, or written too lossily)
For the full-context condition every miss is a reader miss by construction (the whole conversation is shown).
Presence is judged by luna (effort low) on the item text against the context, in pieces of PIECE characters.

Also reports, per question, the best score any method reached (oracle over methods): the head-room for a method
that combines what each baseline does well.

Usage: python3 -m mem.erran run|report [--conds lychee,hindsight,rag,lightmem,full] [--par 6]
"""
import argparse, collections, json, os, re
from concurrent.futures import ThreadPoolExecutor
from . import llm, beam as B, locomo_exp as L

OUT = B.OUT                      # follows V3_BEAM_SPLIT (100k / 1m / lme)
PIECE = 60000
PRESENT_SYS = ("You check whether a piece of text contains a given piece of information. You get a requirement "
               "(something a correct reply must contain or do) and a text. Reply PRESENT if the text states the "
               "information the requirement needs (wording may differ), PARTIAL if it states part of it, ABSENT "
               "otherwise. Reply with one word.")


def present(item, text):
    """PRESENT / PARTIAL / ABSENT of a rubric item in a (possibly long) text, piece by piece."""
    best = "ABSENT"
    for i in range(0, max(1, len(text)), PIECE):
        v = llm.call(PRESENT_SYS, f"Requirement: {item}\n\nText:\n{text[i:i + PIECE]}", effort="low",
                     max_tokens=2000, tag="erran_present").upper()
        if "PRESENT" in v and "ABSENT" not in v.replace("PRESENT", ""):
            return "PRESENT"
        if "PARTIAL" in v:
            best = "PARTIAL"
    return best


def run(_):
    ans = {(r["qid"], r["cond"]): r for r in map(json.loads, open(os.path.join(OUT, "answers.jsonl")))}
    judged = {(r["qid"], r["cond"]): r for r in map(json.loads, open(os.path.join(OUT, "judged.jsonl")))}
    ctx = {}
    for c in ARGS.conds:
        f = os.path.join(OUT, "ext", c, "ctx.jsonl")
        if os.path.exists(f):
            ctx.update({(r["qid"], c): r["context"] for r in map(json.loads, open(f))})
    recs = {}
    ldir = os.path.join(OUT, "lychee")
    for f in os.listdir(ldir):
        m = re.match(r"mem_(\d+)\.json$", f)
        if m:
            recs[int(m.group(1))] = "\n".join(r["text"] for r in json.load(open(os.path.join(ldir, f)))["records"])
    path = os.path.join(OUT, "erran.jsonl")
    done = {(r["qid"], r["cond"], r["item"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = []
    for (qid, c), j in judged.items():
        if c not in ARGS.conds:
            continue
        for item, s in zip(j["rubric"], j["scores"]):
            if s < 1 and (qid, c, item) not in done:
                jobs.append((qid, c, item, s))
    llm.check_quota()
    print("items to check", len(jobs), flush=True)

    def one(job):
        qid, c, item, s = job
        if c == "full":
            where = "reader"
        else:
            text = ctx.get((qid, c)) or ans[(qid, c)].get("context") or ""
            if not text:                         # beam.py answers do not store the context: rebuild it is costly,
                text = ""                        # so conditions without a saved context are analysed by store only
            p = present(item, text) if text else "UNKNOWN"
            if p == "PRESENT":
                where = "reader"
            elif c == "lychee":
                ci = int(re.match(r"b(\d+)-", qid).group(1))
                where = "retrieval" if present(item, recs[ci]) == "PRESENT" else "write"
            else:
                where = "context" if p != "UNKNOWN" else "unknown"
        return {"qid": qid, "cond": c, "ability": judged[(qid, c)]["ability"], "item": item, "score": s, "where": where}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
    report(_)


def report(_):
    judged = [r for r in map(json.loads, open(os.path.join(OUT, "judged.jsonl"))) if r["cond"] in ARGS.conds]
    rows = [json.loads(l) for l in open(os.path.join(OUT, "erran.jsonl"))]
    lost = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        n = len(next(j["rubric"] for j in judged if j["qid"] == r["qid"] and j["cond"] == r["cond"])) or 1
        lost[r["cond"]][r["where"]] += (1 - r["score"]) / n          # points lost on this question, as a share
    nq = collections.Counter(j["cond"] for j in judged)
    print("points lost per question (out of 100), by where the information was lost")
    for c in ARGS.conds:
        tot = sum(lost[c].values())
        print(f"  {c:10s} lost {100 * tot / max(1, nq[c]):5.1f}  " +
              "  ".join(f"{k} {100 * v / max(1, nq[c]):5.1f}" for k, v in lost[c].most_common()))
    by = collections.defaultdict(dict)
    for j in judged:
        by[j["qid"]][j["cond"]] = j["score"]
    qs = [q for q, v in by.items() if all(c in v for c in ARGS.conds)]
    print(f"\noracle over methods ({len(qs)} questions with all of {ARGS.conds}):")
    for c in ARGS.conds:
        print(f"  {c:10s} {100 * sum(by[q][c] for q in qs) / len(qs):5.1f}")
    print(f"  best-of   {100 * sum(max(by[q].values()) for q in qs) / len(qs):5.1f}")
    mem = [c for c in ARGS.conds if c != "full"]
    print(f"  best-of without full {100 * sum(max(by[q][c] for c in mem) for q in qs) / len(qs):5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "report"])
    ap.add_argument("--conds", default="lychee,hindsight,rag,lightmem,full")
    ap.add_argument("--par", type=int, default=6)
    ARGS = ap.parse_args()
    ARGS.conds = ARGS.conds.split(",")
    {"run": run, "report": report}[ARGS.step](None)
