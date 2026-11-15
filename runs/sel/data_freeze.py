"""DATA_FREEZE for SPEC_STAGE23 v3.2 (3.1, 3.2): exact-ID trainable and inner-validation lists per benchmark, plus md5 of
every source file. Inner validation is drawn by history (BEAM conversations, LME question families, a LoCoMo conversation,
EverMem topic t02); EpBench has one book, so its inner split is by question (disclosed). Seed 20261102.
LME exclusions: same family (X / X_abs) as any final question, and any question whose history contains an evidence turn of
a revised-final question (DATA_INVENTORY 11-02). Runs no model. usage: python3 data_freeze.py > DATA_FREEZE_STAGE23.json"""
import hashlib, json, random, re
rng = random.Random(20261102)
norm = lambda t: re.sub(r"\s+", " ", (t or "")).strip().lower()
hsh = lambda t: hashlib.md5(norm(t).encode()).hexdigest()
md5f = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
conv = lambda qid: qid.split("_")[0]
F = {"beam500k": "beam500k_g.json", "beam100k": "beam100k_g.json", "beam1m_b6_15": "beam1m_test_g.json", "lme": "../lme/longmemeval_s_cleaned.json",
     "loc": "q_loc_dev_all.json", "ev_t01": "../bench/evermem/evermem_g.t01.json", "ev_t02": "../bench/evermem/evermem_g.t02.json",
     "ep200": "../bench/epbench/epbench200_g.json", "splits": "splits.json", "final_revised": "final_revised_ids.json"}
F.update({"fin_beam": "beam1m_final_g.json", "fin_ep": "../bench/epbench/epbench_g.json", "fin_ev": "../bench/evermem/evermem_g.t04.json", "fin_loc": "q_loc_final150.json"})
out = {"seed": 20261102, "md5": {k: md5f(v) for k, v in F.items()}, "files": F, "bench": {}}
# gold evidence turns (any role, has_answer) of every final set, revised lists where they exist (SPEC 3.1, audit 11-02)
FR = json.load(open(F["final_revised"]))
def gold_turns(qs, keep=None):
    g = set()
    for q in qs:
        if keep is not None and q["question_id"] not in keep: continue
        for sess in q["haystack_sessions"]:
            for t in sess:
                if t.get("has_answer"): g.add(hsh(t["content"]))
    return g
FINEV = set()
FINEV |= gold_turns(json.load(open(F["fin_beam"])))
FINEV |= gold_turns(json.load(open(F["lme"])), set(FR["lme_final_revised"]))
FINEV |= gold_turns(json.load(open(F["fin_ep"])), set(FR["ep_final_revised"]))
FINEV |= gold_turns(json.load(open(F["fin_ev"])))
FINEV |= gold_turns(json.load(open(F["fin_loc"])))
def touches_final(q): return any(hsh(t["content"]) in FINEV for sess in q["haystack_sessions"] for t in sess)
QALL = {}
for k in ("beam500k", "beam100k", "beam1m_b6_15", "loc", "ev_t01", "ev_t02", "ep200"):
    for q in json.load(open(F[k])): QALL[q["question_id"]] = q
for q in json.load(open(F["lme"])):
    if q["question_id"] in set(json.load(open(F["splits"]))["lme"]["dev"]): QALL[q["question_id"]] = q
DROP = {i for i, q in QALL.items() if touches_final(q)}
S = json.load(open(F["splits"]))


def by_hist(ids, key, frac):
    hs = sorted({key(i) for i in ids}); rng.shuffle(hs); k = max(1, round(frac * len(hs))); held = set(hs[:k])
    return sorted(i for i in ids if key(i) not in held), sorted(i for i in ids if key(i) in held)


# BEAM: 20% of conversations per size
b = {"train": [], "inner": []}
for name in ("beam500k", "beam100k", "beam1m_b6_15"):
    ids = [q["question_id"] for q in json.load(open(F[name]))]
    tr, inn = by_hist(ids, conv, 0.2); b["train"] += tr; b["inner"] += inn
out["bench"]["beam"] = b
# LME
L = {q["question_id"]: q for q in json.load(open(F["lme"]))}
fam = lambda i: i.replace("_abs", "")
fin_rev = json.load(open(F["final_revised"]))["lme_final_revised"]; finfam = {fam(i) for i in S["lme"]["final"]}
def ans_turns(q):
    sid = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    return {hsh(t["content"]) for s in q["answer_session_ids"] if s in sid for t in sid[s] if t["role"] == "user"}
finev = set().union(*(ans_turns(L[i]) for i in fin_rev))
dev = [i for i in S["lme"]["dev"] if i in L]
c1 = [i for i in dev if fam(i) not in finfam]
c2 = [i for i in c1 if not ({hsh(t["content"]) for s in L[i]["haystack_sessions"] for t in s if t["role"] == "user"} & finev)]
tr, inn = by_hist(c2, fam, 0.15)
out["bench"]["lme"] = {"train": tr, "inner": inn, "excluded_same_family_as_final": len(dev) - len(c1), "excluded_final_evidence_in_history": len(c1) - len(c2)}
# LoCoMo: one conversation held out
ids = [q["question_id"] for q in json.load(open(F["loc"]))]
cs = sorted({conv(i) for i in ids}); held = rng.choice(cs)
out["bench"]["loc"] = {"train": sorted(i for i in ids if conv(i) != held), "inner": sorted(i for i in ids if conv(i) == held), "inner_conversation": held}
# EverMem: t01 train, t02 inner
out["bench"]["ev"] = {"train": sorted(q["question_id"] for q in json.load(open(F["ev_t01"]))), "inner": sorted(q["question_id"] for q in json.load(open(F["ev_t02"])))}
# EpBench: one book, 15% of questions
ids = sorted(q["question_id"] for q in json.load(open(F["ep200"]))); rng.shuffle(ids); k = round(0.15 * len(ids))
out["bench"]["ep"] = {"train": sorted(ids[k:]), "inner": sorted(ids[:k]), "note": "single book: inner split by question (same history)"}
for k_, v in out["bench"].items():
    # every benchmark (LME included, on top of its stricter answer-session rule) goes through the all-final check
    v["excluded_any_final_gold_turn_in_history"] = sum(i in DROP for i in v["train"] + v["inner"])
    v["train"] = [i for i in v["train"] if i not in DROP]; v["inner"] = [i for i in v["inner"] if i not in DROP]
    v["n_train"], v["n_inner"] = len(v["train"]), len(v["inner"])
out["final_gold_turns_checked"] = len(FINEV)
print(json.dumps(out, indent=0))
