"""SPEC_STAGE23 v3.2 code check: mkdata --teacher-only output must not depend on gold. Perturb has_answer flags in the
questions and the gold/stage fields of the pools, rerun with the same seed, require byte-identical output.
usage (in ~/data/rtm/sel): python3 test_teacher_only.py q_lmes150.json pool_lmes150 20"""
import json, os, random, subprocess, sys, hashlib, tempfile, shutil
qf, pd, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
qs = json.load(open(qf))[:n]; tmp = tempfile.mkdtemp(prefix="tto_")
def run(qpath, pdir, out):
    subprocess.run([sys.executable, "mkdata.py", qpath, pdir, out, "--teacher-only", "--n-groups", "60", "--seed", "7"], check=True, capture_output=True)
    return hashlib.md5(open(out, "rb").read()).hexdigest()
json.dump(qs, open(f"{tmp}/q.json", "w"))
h1 = run(f"{tmp}/q.json", pd, f"{tmp}/a.jsonl")
rng = random.Random(1)
for q in qs:
    for s in q["haystack_sessions"]:
        for t in s: t["has_answer"] = rng.random() < 0.3
json.dump(qs, open(f"{tmp}/q2.json", "w"))
os.makedirs(f"{tmp}/pool"); 
for q in qs:
    p = json.load(open(os.path.join(pd, q["question_id"] + ".json")))
    p["gold"] = [rng.random() < 0.3 for _ in p.get("gold", [])]; p["stage"] = [f"x{rng.randint(0, 9)}" for _ in p.get("stage", [])]
    json.dump(p, open(f"{tmp}/pool/{q['question_id']}.json", "w"))
h2 = run(f"{tmp}/q2.json", f"{tmp}/pool", f"{tmp}/b.jsonl")
rows = [json.loads(l) for l in open(f"{tmp}/a.jsonl")]
ok = h1 == h2 and len(rows) == 60 and all(r["nolabel"] and len(r["docs"]) == 16 and len(set(r["docs"])) == 16 and all(t is not None for t in r["teacher"]) for r in rows)
sys.path.insert(0, "."); from k1 import rounds_of; from mkdata import rounds_text
same = all([r[:3] for r in rounds_of(q)] == rounds_text(q) for q in json.load(open(f"{tmp}/q.json")))
ok = ok and same
print("rounds_text equals rounds_of without labels:", same, "|", "identical after perturbing gold:", h1 == h2, "| groups", len(rows), "| all 16 distinct scored docs:", ok)
shutil.rmtree(tmp); sys.exit(0 if ok else 1)
