"""Data inventory and content dedup (audit 11-02, item 4), CPU only.
For each candidate training source and each final set: number of questions, histories, user turns; share of the
source's distinct user turns (normalized, >= 30 chars) that also occur in the final set; exact question-text overlap.
usage: python3 dedup.py > dedup_out.txt"""
import json, re, sys, collections, hashlib, os

norm = lambda t: re.sub(r"\s+", " ", re.sub(r"->->\s*[\d,\sN/A]*", "", t or "")).strip().lower()


def load(path, keep=None):
    qs = json.load(open(path))
    if keep: qs = [q for q in qs if keep(q["question_id"])]
    turns, hist, ques = set(), set(), set()
    for q in qs:
        h = hashlib.md5(json.dumps(q.get("haystack_dates", [])).encode() + str(len(q["haystack_sessions"])).encode()).hexdigest()
        hist.add(h); ques.add(norm(q["question"]))
        for s in q["haystack_sessions"]:
            for t in s:
                if t["role"] == "user":
                    x = norm(t["content"])
                    if len(x) >= 30: turns.add(hashlib.md5(x.encode()).hexdigest())
    return dict(n=len(qs), hist=len(hist), turns=turns, ques=ques)


def conv(q): return q.split("_")[0]
def bconv(lo, hi): return lambda qid: (m := re.match(r"b(\d+)$", conv(qid))) is not None and lo <= int(m.group(1)) <= hi


S = json.load(open("splits.json"))
lme_final, lme_dev = set(S["lme"]["final"]), set(S["lme"]["dev"]) | set(S["lme"]["curriculum"])
LME = "../lme/longmemeval_s_cleaned.json"; EVD = "../bench/evermem/evermem_g.t0{}.json"
lmeid = lambda ids: (lambda i: i in ids or i.replace("_abs", "") in ids)
SRC = {   # candidate training sources (all non-final data)
    "BEAM-500K = dev h1-35": ("beam500k_g.json", None),
    "BEAM-100K = dev c1-20": ("beam100k_g.json", None),
    "BEAM-1M b1-15 (curriculum + demoted dev)": ("beam1m_dev_g.json", None),
    "BEAM-1M b6-15 file": ("beam1m_test_g.json", None),
    "LME dev+curriculum (357)": (LME, lmeid(lme_dev)),
    "EpBench-200": ("../bench/epbench/epbench200_g.json", None),
    "EverMem t01": (EVD.format(1), None), "EverMem t02": (EVD.format(2), None), "EverMem t03": (EVD.format(3), None),
    "LoCoMo conv 0-4": ("q_loc150.json", None),
}
FIN = {
    "BEAM final 1M b16-35": ("beam1m_final_g.json", None),
    "LME final (138)": (LME, lmeid(lme_final)),
    "EpBench-2000": ("../bench/epbench/epbench_g.json", None),
    "EverMem t04": (EVD.format(4), None),
}
F = {}
for k, (p, f) in FIN.items():
    try: F[k] = load(p, f); print(f"final {k}: {F[k]['n']} questions, {F[k]['hist']} histories, {len(F[k]['turns'])} user turns", flush=True)
    except Exception as e: print(f"final {k}: cannot load {p}: {e!r}"[:200], flush=True)
print()
for k, (p, f) in SRC.items():
    try: s = load(p, f)
    except Exception as e: print(f"source {k}: cannot load {p}: {e!r}"[:200], flush=True); continue
    print(f"source {k}: {s['n']} questions, {s['hist']} histories, {len(s['turns'])} user turns", flush=True)
    for fk, fv in F.items():
        ov = len(s["turns"] & fv["turns"]); oq = len(s["ques"] & fv["ques"])
        if ov or oq: print(f"    vs {fk}: shared user turns {ov} ({100*ov/max(1,len(s['turns'])):.1f}% of source), identical questions {oq}")
    print(flush=True)
