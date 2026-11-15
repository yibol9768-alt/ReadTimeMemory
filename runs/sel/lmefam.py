import json, re, hashlib
norm = lambda t: re.sub(r"\s+", " ", (t or "")).strip().lower()
L = {q["question_id"]: q for q in json.load(open("../lme/longmemeval_s_cleaned.json"))}
S = json.load(open("splits.json"))["lme"]; dev = set(S["dev"]) | set(S["curriculum"])
def ans_text(q):
    sid = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    return {hashlib.md5(norm(t["content"]).encode()).hexdigest() for s in q["answer_session_ids"] if s in sid for t in sid[s] if t["role"] == "user"}
def all_text(q): return {hashlib.md5(norm(t["content"]).encode()).hexdigest() for s in q["haystack_sessions"] for t in s if t["role"] == "user"}
n = 0
for i in S["final"]:
    f = i.replace("_abs", "")
    tw = [j for j in (f, f + "_abs") if j != i and j in dev and j in L]
    for j in tw:
        a_fin, a_dev = ans_text(L[i]), ans_text(L[j])
        print(i, "<->", j, "| final evidence turns", len(a_fin), "| shared with dev evidence turns", len(a_fin & a_dev), "| final evidence turns anywhere in dev history", len(a_fin & all_text(L[j])))
        n += 1
print("pairs", n)
keep = json.load(open("final_revised_ids.json"))["lme_final_revised"]
devall = set()
for j in dev:
    if j in L: devall |= all_text(L[j])
hit = []
for i in keep:
    a = ans_text(L[i])
    if a & devall: hit.append((i, len(a & devall), len(a)))
print("kept final questions whose evidence turns appear anywhere in dev/curriculum histories:", len(hit), "of", len(keep))
for x in hit[:10]: print("  ", x)
finev = set()
for i in keep: finev |= ans_text(L[i])
trainable = [j for j in S["dev"] if j in L and j.replace("_abs", "") not in {k.replace("_abs", "") for k in S["final"]}]
bad = [j for j in trainable if all_text(L[j]) & finev]
print(f"LME trainable dev questions {len(trainable)}; containing any kept-final evidence turn {len(bad)}; left {len(trainable) - len(bad)}")
