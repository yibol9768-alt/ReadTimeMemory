"""Register and freeze the never-used held-out benchmarks (2026-11-02, user approved WORLD_V2_PLAN). Reads ids and metadata
only; prints no question text and runs no model. Nothing about these sets may be used for any choice until the final run.
  REALTALK: every question with evidence (10 real 21-day chats; raw files checked against the GitHub tree sha1).
  ConvoMem (multi-message part, converted): stratified by question type, 600 questions, seed 20261102, excluding the three
  questions inspected by hand on 10-31 and every question whose evidence lines were not all matched.
usage: python3 -I hold_freeze.py OUT.json"""
import collections, glob, hashlib, json, os, random, sys
B = os.path.expanduser("~/data/rtm/bench")
out = {"created": "2026-11-02 EST", "rule": __doc__, "realtalk": {}, "convomem": {}}
fs = sorted(glob.glob(f"{B}/realtalk/raw/data/*.json")); assert len(fs) == 10, fs
rt = []
for f in fs:
    d = json.load(open(f)); name = os.path.basename(f)[:-5]
    out["realtalk"].setdefault("md5", {})[name] = hashlib.md5(open(f, "rb").read()).hexdigest()
    for k, q in enumerate(d["qa"]):
        if q.get("evidence"): rt.append({"id": f"rt_{name}_{k}", "category": q.get("category"), "n_evidence": len(q["evidence"])})
out["realtalk"]["questions"] = rt; out["realtalk"]["n"] = len(rt)
out["realtalk"]["by_category"] = dict(collections.Counter(str(x["category"]) for x in rt))
INSPECTED = {"v202_changing_evidence_0", "v425_user_evidence_1", "v60_assistant_facts_evidence_4"}
cm = json.load(open(f"{B}/convomem/convomem_g.json")); ok = []; drop = collections.Counter()
for q in cm:
    qid = q["question_id"]
    if qid in INSPECTED: drop["inspected"] += 1; continue
    m, n = q.get("n_evidence_lines_matched"), q.get("n_evidence_lines")
    if m is None or n is None or m < n or not q.get("n_src"): drop["evidence not fully matched"] += 1; continue
    ok.append((qid, q.get("question_type", "?"), q.get("n_src")))
by = collections.defaultdict(list)
for x in ok: by[x[1]].append(x)
rng = random.Random(20261102); N = 600; tot = len(ok); pick = []
for t in sorted(by):
    k = round(N * len(by[t]) / tot); L = sorted(by[t]); rng.shuffle(L); pick += L[:k]
pick = sorted(pick)
out["convomem"] = {"source": "convomem_g.json (converted 10-31, BENCH_NEW_2026-10-31.md)", "md5_first_64MB": hashlib.md5(open(f"{B}/convomem/convomem_g.json", "rb").read(64 << 20)).hexdigest(),
                   "size_bytes": os.path.getsize(f"{B}/convomem/convomem_g.json"), "eligible": tot, "dropped": dict(drop), "n": len(pick),
                   "questions": [{"id": a, "type": b, "n_src": c} for a, b, c in pick], "by_type": dict(collections.Counter(b for _, b, _ in pick))}
json.dump(out, open(sys.argv[1], "w"), indent=0)
print("REALTALK", out["realtalk"]["n"], out["realtalk"]["by_category"]); print("ConvoMem eligible", tot, "dropped", dict(drop), "frozen", len(pick), out["convomem"]["by_type"])
print("freeze md5", hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest())
