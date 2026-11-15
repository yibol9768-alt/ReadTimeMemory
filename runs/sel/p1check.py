"""Acceptance check for one infer.py output before it is published (Codex audit 11-02).
Passes only if: every expected qid is present exactly once per budget, budgets == K1_BUDGETS, and each row's 'rank'
is exactly the set of the first P pool rounds of that question.
usage: python3 p1check.py CTX_TMP QFILE POOLDIR LIMIT [P]   -> exit 0 and prints OK, else exit 1"""
import collections, json, os, sys
ctxp, qf, pd, lim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); P = int(sys.argv[5]) if len(sys.argv) > 5 else 600
B = sorted(int(x) for x in os.environ["K1_BUDGETS"].split(","))
qs = json.load(open(qf)); qs = qs[: lim or None]; want = {q["question_id"] for q in qs}
seen = collections.Counter(); bad = []
for l in open(ctxp):
    r = json.loads(l); seen[(r["qid"], r["budget"])] += 1
    pool = json.load(open(os.path.join(pd, r["qid"] + ".json")))["fused"][:P]
    if sorted(r["rank"]) != sorted(pool) or len(r["rank"]) != len(set(r["rank"])): bad.append(("rank", r["qid"], r["budget"]))
dup = [k for k, v in seen.items() if v != 1]
miss = [(q, b) for q in want for b in B if (q, b) not in seen]
extra = [k for k in seen if k[0] not in want or k[1] not in B]
if dup or miss or extra or bad:
    print("FAIL", ctxp, "dup", len(dup), "missing", len(miss), "extra", len(extra), "bad rank", len(bad), (dup + miss + extra + bad)[:5]); sys.exit(1)
print("OK", ctxp, len(want), "questions x", len(B), "budgets")
