"""Inner-validation recall area for checkpoint and r selection (SPEC_STAGE23 v3.2 3.2, 6).
Area = mean over budgets 1k, 2k, 4k, 8k, 16k (log-spaced, equal weight) of evidence recall; EverMem uses gold sessions.
Benchmarks of the fold are weighted equally. Every inner question of every listed benchmark must be present with all five
budgets exactly once, else INVALID (exit 2). The ctx files come from infer.py with K1_BUDGETS=1000,2000,4000,8000,16000
on the inner question files written by stage_qfiles.py (q_inner_<bench>.json).
usage: python3 inner_eval.py TAG beam,lme,loc,ev,ep      (reads ctx_<TAG>_inner_<bench>.jsonl)"""
import hashlib, json, statistics as st, sys
sys.path.insert(0, ".")
from k1 import rounds_of
B = [1000, 2000, 4000, 8000, 16000]
# Canonical identity hash of the frozen H20 INNERSEL_STAGE23.json (read-only check,
# 2026-11-03). Hash sorted IDs per benchmark, not whitespace or JSON key order.
INNERSEL_IDENTITY_SHA256 = "735cf4551717acd1dcd21939cf21a58e9592d67b9cd170aec7c77a1896e9c8e3"


def selection_hash(selection):
    canonical = {b: sorted(ids) for b, ids in selection.items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_selection(frozen, selection):
    if set(selection) != set(frozen): raise ValueError("INNERSEL benchmark identities differ from frozen inner set")
    for b, ids in selection.items():
        if len(ids) != len(set(ids)): raise ValueError(f"INNERSEL {b}: duplicate question IDs")
        if len(ids) > min(100, len(frozen[b]["inner"])): raise ValueError(f"INNERSEL {b}: per-benchmark cap exceeded")
        if not set(ids) <= set(frozen[b]["inner"]): raise ValueError(f"INNERSEL {b}: IDs outside frozen inner set")
    if selection_hash(selection) != INNERSEL_IDENTITY_SHA256:
        raise ValueError("INNERSEL frozen identity hash mismatch")
    return {b: {"inner": selection[b]} for b in selection}


def smoke_ids(ids, bench):
    """Fixed subset of INNERSEL, independent of training seed and model/arm."""
    import random
    chosen = sorted(ids)
    random.Random(f"20261103-smoke-{bench}").shuffle(chosen)
    return set(chosen[:10])


def inner_questions(bench, frozen, selmode=False, smoke=False):
    if smoke and not selmode: raise ValueError("smoke requires INNERSEL (--sel)")
    path = f"q_innersel_{bench}.json" if selmode else f"q_inner_{bench}.json"
    questions = json.load(open(path)); ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)): raise ValueError(f"{bench}: duplicate question IDs in question file")
    if set(ids) != set(frozen[bench]["inner"]): raise ValueError(f"{bench}: inner question identity mismatch")
    if smoke:
        chosen = smoke_ids(ids, bench)
        questions = [q for q in questions if q["question_id"] in chosen]
    return questions


def evaluate(tag, benches, selmode=False, smoke=False):
    fr = json.load(open("DATA_FREEZE_STAGE23.json"))["bench"]
    if selmode:
        fr = validate_selection(fr, json.load(open("INNERSEL_STAGE23.json")))
    per = {}
    for b in benches:
        questions = inner_questions(b, fr, selmode, smoke)
        Q = {q["question_id"]: q for q in questions}
        want = set(Q)
        gs = {}
        if b == "ev":
            for qid, q in Q.items(): R = rounds_of(q); gs[qid] = ({r[0] for r in R if r[3]}, R)
        if set(Q) != want: print(f"INVALID {b}: q_inner_{b}.json does not match the frozen inner list"); sys.exit(2)
        nogold = {qid for qid, q in Q.items() if not any(r[3] for r in rounds_of(q))}   # decided from the question file, not from ctx rows
        seen, val = {}, {}
        for r in map(json.loads, open(f"ctx_{tag}_inner_{b}.jsonl")):
            k = (r["qid"], r["budget"]); seen[k] = seen.get(k, 0) + 1
            if r["qid"] not in want or r["budget"] not in B: print(f"INVALID {b}: unexpected row {k}"); sys.exit(2)
            if r["qid"] in nogold: continue
            if not r["gold_total"]: print(f"INVALID {b}: gold_total 0 for a question with gold {k}"); sys.exit(2)
            if b == "ev":
                g, R = gs[r["qid"]]; v = len({R[i][0] for i in r["keep"]} & g) / len(g)
            else: v = r["gold_in"] / r["gold_total"]
            val[k] = v
        dup = [k for k, n in seen.items() if n != 1]; miss = [(q, x) for q in want for x in B if (q, x) not in seen]
        if dup or miss: print(f"INVALID {b}: duplicates {len(dup)} missing {len(miss)}"); sys.exit(2)
        qs = sorted(want - nogold)
        per[b] = st.mean(st.mean(val[(q, x)] for x in B) for q in qs)
    return {"tag": tag, "per_bench": {k: round(100 * v, 2) for k, v in per.items()}, "area": round(100 * st.mean(per.values()), 3)}


if __name__ == "__main__":
    try:
        print(json.dumps(evaluate(sys.argv[1], sys.argv[2].split(","), "--sel" in sys.argv[3:], "--smoke" in sys.argv[3:])))
    except (ValueError, KeyError) as exc:
        print(f"INVALID: {exc}"); sys.exit(2)
