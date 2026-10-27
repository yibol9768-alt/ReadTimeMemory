"""CCR (docs/refine/FINAL_PROPOSAL.md): records keyed by mutable proposition, never adjudicated at write time.

Write side. Same segmentation and state chaining as Lychee-lite (mem/lychee.py encode), with two changes to the
encoder prompt: every record names the proposition(s) it states a value for (slot: reuse an id from the candidate
table, or create one with a canonical question that does not contain the answer), and the encoder must not decide
which of two inconsistent statements is right - both become records.
Candidate table per segment: top 20 existing slots by embedding similarity of their question to the segment text
(H20 qwen3-embedding-8b via retrieve.precompute_ranks) + the 10 most recently used, de-duplicated.

Output: runs/<split>/<arm>/ccr/mem_<ci>.json {"records": [... "slots": ["S0001", ...]], "slots": {id: question}}

Usage: V3_BEAM_SPLIT=100k python3 -m mem.ccr build [--par 6] | audit
"""
import argparse, collections, json, os, re, shutil, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
from . import llm, lychee as Y, beam as B, retrieve as R

TOP_SIM, RECENT, MAX_SLOTS = 20, 10, 2
ENC_SYS = Y.ENC_SYS.replace(
    'Output one JSON object per line, nothing else:\n',
    "Never decide which of two inconsistent statements is right, and never drop a statement because it conflicts "
    "with an earlier one: if the user says something that contradicts or revises what was said before, write it as "
    "its own record, as said (the disambiguation state is only for who/what names refer to, not for which claim "
    "is true).\n"
    "Slots: every record states a value for one or two mutable propositions - a question about the user's world "
    "that has one answer at a time but can change or be stated inconsistently (e.g. \"Has Darryl ever placed an "
    "online sneaker order?\", \"When is Darryl's Foot Locker visit on Main Street?\", \"Who is Crystal's will "
    "executor?\"). The question names the entity, the attribute and, when needed, which event instance; it never "
    "contains the answer. You get a table of existing slots: reuse an id when the record answers the same question "
    "(the same event instance), otherwise create a new one by giving its question. Give at most two slots per "
    "record; if a record answers more independent questions, split it into several records.\n"
    'Output one JSON object per line, nothing else:\n').replace(
    '"sources": ["D1:3", ...]}',
    '"sources": ["D1:3", ...], "slots": ["S0007" or {"new": "question?"}, ...]}')
assert ENC_SYS != Y.ENC_SYS and '"slots"' in ENC_SYS


def parse(text, valid, registry, recent):
    recs, state = Y.parse(text, valid)
    raw = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("text"):
                raw.append(r.get("slots") or [])
    by_q = {q.strip().lower(): sid for sid, q in registry.items()}
    for r, ss in zip(recs, raw):           # Y.parse keeps the same lines in the same order
        ids = []
        for s in ss[:MAX_SLOTS] if isinstance(ss, list) else []:
            if isinstance(s, str) and s.strip() in registry:
                ids.append(s.strip())
            elif isinstance(s, dict) and str(s.get("new", "")).strip():
                q = str(s["new"]).strip()
                sid = by_q.get(q.lower())
                if not sid:
                    sid = f"S{len(registry) + 1:04d}"
                    registry[sid] = q; by_q[q.lower()] = sid
                ids.append(sid)
        r["slots"] = list(dict.fromkeys(ids))
        for sid in r["slots"]:
            if sid in recent:
                recent.remove(sid)
            recent.append(sid)
    return recs, state


_LOCK = threading.Lock()     # one remote ranking job at a time (the H20 side keeps one vector cache file)


def rank(qs, query):
    """Same remote ranker as retrieve.precompute_ranks, but not written to runs/rank_cache.json (other runs use it)."""
    job = json.dumps({"docs": {"g": qs}, "queries": [["g", query]]}).encode()
    with _LOCK:
        for attempt in range(3):
            try:
                out = subprocess.run(R.SSH + ["python3 ~/v3_emb/remote_rank.py"], input=job, capture_output=True,
                                     check=True, timeout=600)
                return json.loads(out.stdout)["g\t" + query]
            except Exception:
                if attempt == 2:
                    raise


def candidates(registry, recent, seg_text):
    ids = list(registry)
    out = []
    if ids:
        out = [ids[j] for j in rank([registry[i] for i in ids], seg_text[:3000])[:TOP_SIM]]
    out += [s for s in reversed(recent[-RECENT:]) if s not in out]
    return out


def encode(ci, convs, starts, out):
    path = os.path.join(out, f"mem_{ci}.json")
    if os.path.exists(path):
        return
    flat = [(s, t) for s in convs[ci]["sessions"] for t in s["turns"]]
    bounds = starts + [len(flat)]
    state, recs, registry, recent = "(empty)", [], {}, []
    for a, b in zip(bounds, bounds[1:]):
        piece = flat[a:b]
        s = piece[0][0]
        seg = "\n".join(Y.show(t, s["date"]) for _, t in piece)
        cand = candidates(registry, recent, seg)
        table = "\n".join(f"{sid}: {registry[sid]}" for sid in cand) or "(none yet)"
        body = (f"Disambiguation state:\n{state}\n\nExisting slots (candidates):\n{table}\n\n"
                f"Next segment (session {s['n']}, date {s['date']}):\n{seg}")
        rs, st = parse(llm.call(ENC_SYS, body, tag="ccr_enc"), {t["id"] for _, t in piece}, registry, recent)
        recs += rs
        state = st or state
    json.dump({"records": recs, "slots": registry, "segments": len(bounds) - 1}, open(path, "w"),
              ensure_ascii=False, indent=1)
    sizes = collections.Counter(s for r in recs for s in r["slots"])
    print("ccr", ci, "records", len(recs), "slots", len(registry), "unkeyed", sum(not r["slots"] for r in recs),
          "max chain", max(sizes.values() or [0]), flush=True)


def build(_):
    convs = B.load()
    cis = B.picked(convs)
    out = os.path.join(B.OUT, "ccr"); os.makedirs(out, exist_ok=True)
    if not os.path.exists(os.path.join(out, "segments.json")):           # identical segmentation to lychee
        shutil.copy(os.path.join(B.OUT, "lychee", "segments.json"), os.path.join(out, "segments.json"))
    seg = json.load(open(os.path.join(out, "segments.json")))
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(cis))) as ex:
        list(ex.map(lambda ci: encode(ci, convs, seg[str(ci)], out), cis))


def audit(_):
    """Chain-size distribution; for the contradiction / knowledge-update questions, print the slots whose records
    mention the question's key words, for the manual split check (FINAL_PROPOSAL: slot audit)."""
    convs = B.load()
    out = os.path.join(B.OUT, "ccr")
    for ci in B.picked(convs):
        m = json.load(open(os.path.join(out, f"mem_{ci}.json")))
        sizes = collections.Counter(s for r in m["records"] for s in r["slots"])
        dist = collections.Counter(min(v, 10) for v in sizes.values())
        print(f"conv {ci}: {len(m['records'])} records, {len(m['slots'])} slots, chain sizes (10=10+) "
              f"{sorted(dist.items())}, largest {sizes.most_common(3)}")


AUD_SYS = ("You audit a memory. You get a question about a long conversation, the grading points a correct reply "
           "must contain, and a numbered list of memory records. Find the records that state the facts the grading "
           "points rely on. Output one JSON object, nothing else: {\"sides\": [{\"fact\": \"short\", "
           "\"records\": [record numbers]}, ...]} with one entry per distinct fact (for a contradiction: one entry "
           "per conflicting statement; for an update: one for the old value, one for the new value). Use an empty "
           "list when no record states that fact.")


def audit2(_):
    """Split check on the contradiction and knowledge-update dev questions: are both sides written, and do they
    share a slot? (luna finds the records for each side; slots are then compared in code)."""
    convs = B.load()
    qs = [q for q in B.questions(convs) if q["ability"] in ("contradiction_resolution", "knowledge_update")]
    out = os.path.join(B.OUT, "ccr")
    led = {ci: json.load(open(os.path.join(out, f"mem_{ci}.json"))) for ci in B.picked(convs)}
    def one(q):
        recs = led[q["conv"]]["records"]
        listing = "\n".join(f"{k}. {r['text']}" for k, r in enumerate(recs))
        v = llm.call(AUD_SYS, f"Question: {q['question']}\nGrading points:\n" + "\n".join(q["rubric"]) +
                     f"\n\nRecords:\n{listing}", effort="low", max_tokens=4000, tag="ccr_audit")
        try:
            sides = json.loads(v[v.index("{"):v.rindex("}") + 1])["sides"]
        except Exception:
            sides = []
        sl = [set(s for k in sd.get("records", []) if isinstance(k, int) and 0 <= k < len(recs)
                  for s in recs[k]["slots"]) for sd in sides]
        written = [bool(sd.get("records")) for sd in sides]
        shared = len(sl) >= 2 and bool(set.intersection(*sl)) if all(written) and len(sl) >= 2 else False
        return {"qid": q["qid"], "ability": q["ability"], "n_sides": len(sides), "all_written": all(written),
                "shared_slot": shared, "sides": [dict(sd, slots=sorted(x)) for sd, x in zip(sides, sl)]}
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, qs))
    json.dump(rows, open(os.path.join(out, "audit2.json"), "w"), ensure_ascii=False, indent=1)
    for ab in ("contradiction_resolution", "knowledge_update"):
        rs = [r for r in rows if r["ability"] == ab]
        print(f"{ab}: {len(rs)} questions, all sides written {sum(r['all_written'] for r in rs)}, "
              f"sides share a slot {sum(r['shared_slot'] for r in rs)}")
    for r in rows:
        print(r["qid"], "written" if r["all_written"] else "MISSING", "shared" if r["shared_slot"] else "split",
              [(sd["fact"][:50], sd["slots"]) for sd in r["sides"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "audit", "audit2"])
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--convs", type=int, default=6)
    ARGS = ap.parse_args()
    B.ARGS = ARGS
    {"build": build, "audit": audit, "audit2": audit2}[ARGS.step](None)
