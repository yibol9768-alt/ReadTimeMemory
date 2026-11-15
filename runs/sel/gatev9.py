"""D3 v9 pre-training gate (spec v8 item 4), on the actual training inputs tr_v9_e / tr_v9_fix. usage: python3 gatev9.py
Exposure: share of sampled positives with carrier long, E minus fixed; pooled >= +15 points and >= 3 of 4 world pairs positive.
Retention: a positive whose full round text holds its ledger value must still hold it after doc_str (3000 characters) and the training token cap (--max-d 1100), on the training tokenizer.
Positives with no literal value (paraphrased) are counted and reported as unresolved, never dropped silently. Need >= 95% of resolved long positives retained in E. exit 0 PASS, 3 FAIL."""
import collections, json, re, sys
sys.path.insert(0, "."); from k1 import rounds_of, ntok
from fmt import doc_str, split_round, round_prev
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("ckpt/w20"); W = [998, 999, 1000, 1001]; MAXD = 1100; END = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def has(v, text):
    """literal value, or at least 80% of its words of 3+ letters (the model may rephrase the value)"""
    text = text.lower()
    if v in text: return True
    w = re.findall(r"[a-z0-9$%.]{3,}", v)
    return bool(w) and sum(x in text for x in w) / len(w) >= .8
def maps(tag, facts_dir):
    M = {}
    for q in json.load(open(f"env_{tag}.json")):
        w = int(q["question_id"].split("_")[0][1:]); R = rounds_of(q); prev = round_prev(q); d = {}; i = 0
        for s in q["haystack_sessions"]:
            j = 0
            while j < len(s):
                n = 2 if j + 1 < len(s) and s[j]["role"] == "user" and s[j + 1]["role"] == "assistant" else 1
                d[doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i])] = (s[j].get("carrier") or "dialog", str(s[j].get("stage", "")).split("#")[0], R[i][2]); i += 1; j += n
        M[q["question_id"]] = (w, d)
    F = {w: {f["id"]: f["value"].lower() for f in json.load(open(f"{facts_dir}/w{w:05d}.json"))["facts"]} for w in W}
    return M, F
def scan(tag, facts_dir):
    M, F = maps(tag, facts_dir); C = collections.defaultdict(collections.Counter); B = collections.defaultdict(lambda: [0, 0]); bad = unres = 0
    for x in map(json.loads, open(f"tr_v9_{'fix' if tag == 'ws7x' else 'e'}.jsonl")):
        w, d = M[x["qid"]]; doc = x["docs"][x["label"]]
        if doc not in d: bad += 1; continue
        car, fid, full = d[doc]; C[w][car] += 1
        if car != "long": continue
        v = F[w].get(fid, "")
        if not v or not has(v, full): unres += 1; continue
        body = doc[: -len(END)] if doc.endswith(END) else doc
        ids = tok(body, add_special_tokens=False)["input_ids"][:MAXD]; b = min(ntok(full) // 250, 4); B[b][0] += 1; B[b][1] += has(v, tok.decode(ids))
    if bad: print("INVALID", tag, bad, "positives not found"); sys.exit(2)
    return C, B, unres
X, bx, ux = scan("ws7x", "ws7x"); E, be, ue = scan("ws7e", "ws7e")
sh = lambda C, ws: sum(C[w]["long"] for w in ws) / max(1, sum(sum(C[w].values()) for w in ws))
diff = {w: 100 * (sh(E, [w]) - sh(X, [w])) for w in W}; pooled = 100 * (sh(E, W) - sh(X, W))
print("sampled long share fixed", round(100 * sh(X, W), 1), "E", round(100 * sh(E, W), 1), "pooled diff", round(pooled, 1), "per pair", {k: round(v, 1) for k, v in diff.items()})
ok = pooled >= 15 and sum(v > 0 for v in diff.values()) >= 3
for nm, r, u in (("fixed", bx, ux), ("E", be, ue)): print(nm, "retained by length bin (x250 tok)", {k: f"{v[1]}/{v[0]}" for k, v in sorted(r.items())}, "unresolved", u)
a = sum(v[0] for v in be.values()); b = sum(v[1] for v in be.values()); keep = b / max(1, a); print("E resolved-only retention", round(100 * keep, 1), "resolved", a, "unresolved", ue); ok = ok and a > 0 and keep >= .95
if ue > .2 * (a + ue): print("RETENTION UNCERTIFIED: over 20% of long positives could not be matched to their value; the gate covers resolved positives only")
print("GATEV9", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 3)
