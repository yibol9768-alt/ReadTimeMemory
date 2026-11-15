"""Final-set revision lists (SPEC_STAGE23 v3.2 3.4). Computes id lists only; runs no model and reads no result file."""
import json, re, hashlib, random, collections
norm = lambda t: re.sub(r"\s+", " ", (t or "")).strip().lower()
h = lambda t: hashlib.md5(norm(t).encode()).hexdigest()
out = {}
# ---- LME: exact ids ----
S = json.load(open("splits.json"))["lme"]; fin, dev, cur = set(S["final"]), set(S["dev"]), set(S["curriculum"])
L = {q["question_id"]: q for q in json.load(open("../lme/longmemeval_s_cleaned.json"))}
fam = lambda i: i.replace("_abs", "")
devfam = {fam(i) for i in dev | cur}
devsess = {s for i in dev | cur if i in L for s in L[i]["answer_session_ids"]}
drop_fam = sorted(i for i in fin if fam(i) in devfam)
drop_sess = sorted(i for i in fin if i in L and set(L[i]["answer_session_ids"]) & devsess)
keep = sorted(i for i in fin if i not in set(drop_fam) | set(drop_sess))
missing = sorted(i for i in fin if i not in L)
print(f"LME final {len(fin)} ids ({len(missing)} not in data); same family as dev/curriculum {len(drop_fam)}; share an answer session {len(drop_sess)}; union dropped {len(fin) - len(keep)}; kept {len(keep)} (abstention among kept {sum('_abs' in i for i in keep)})")
out["lme_final_revised"] = keep
# ---- EpBench ----
dev_ep = json.load(open("../bench/epbench/epbench200_g.json")); fin_ep = json.load(open("../bench/epbench/epbench_g.json"))
devch = {h(t["content"]) for s in dev_ep[0]["haystack_sessions"] for t in s}
devtpl = {(q["question_type"], q.get("cue")) for q in dev_ep}
book = fin_ep[0]["haystack_sessions"]; chash = [h(t["content"]) for s in book for t in s]
print(f"EpBench: dev chapters {len(devch)}, final book chapters {len(chash)}, of which in dev book {sum(c in devch for c in chash)}")
def ev(q):
    x = q.get("evidence_chapters"); x = json.loads(x) if isinstance(x, str) else x
    return [int(c) for c in (x or [])]
cnt = collections.Counter(); keep_ep = []
for q in fin_ep:
    e = ev(q)
    if not e: cnt["no evidence chapter"] += 1; continue
    if any(c - 1 < 0 or c - 1 >= len(chash) for c in e): cnt["evidence index out of range"] += 1; continue
    if any(chash[c - 1] in devch for c in e): cnt["evidence in dev book"] += 1; continue
    if (q["question_type"], q.get("cue")) in devtpl: cnt["same template+cue as a dev question"] += 1; continue
    keep_ep.append(q)
print("EpBench final filtering:", dict(cnt), "eligible", len(keep_ep))
by = collections.defaultdict(list)
for q in keep_ep: by[q["question_type"]].append(q["question_id"])
rng = random.Random(20261101); tot = len(keep_ep); pick = []
if tot <= 150: pick = [q["question_id"] for q in keep_ep]
else:
    for t in sorted(by):   # stratified by question type, proportional, largest remainder
        rng.shuffle(by[t])
    quota = {t: 150 * len(v) / tot for t, v in by.items()}; base = {t: int(x) for t, x in quota.items()}
    for t in sorted(quota, key=lambda t: -(quota[t] - base[t]))[: 150 - sum(base.values())]: base[t] += 1
    for t in sorted(by): pick += by[t][: base[t]]
print("EpBench final sample", len(pick), "by type", collections.Counter(q.split("_")[0] for q in pick) if False else {t: min(len(by[t]), 0) for t in []})
out["ep_final_revised"] = sorted(pick)
json.dump(out, open("final_revised_ids.json", "w"), indent=0)
print("wrote final_revised_ids.json")
