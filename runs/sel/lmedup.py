import json, re
S = json.load(open("splits.json")); fin = set(S["lme"]["final"]); dev = set(S["lme"]["dev"]) | set(S["lme"]["curriculum"])
q = json.load(open("../lme/longmemeval_s_cleaned.json"))
n = lambda t: re.sub(r"\s+", " ", t).strip().lower(); base = lambda i: i.replace("_abs", "")
in_ = lambda i, ids: i in ids or i.replace("_abs", "") in ids
F, D = {}, {}
for x in q:
    i = x["question_id"]
    if in_(i, fin): F.setdefault(n(x["question"]), []).append(x)
    if in_(i, dev): D.setdefault(n(x["question"]), []).append(x)
for t in sorted(set(F) & set(D)):
    f, d = F[t][0], D[t][0]
    print(f["question_id"], d["question_id"], "| same answer:", str(f["answer"]) == str(d["answer"]),
          "| shared answer sessions:", len(set(f["answer_session_ids"]) & set(d["answer_session_ids"])), "|", t[:70])
