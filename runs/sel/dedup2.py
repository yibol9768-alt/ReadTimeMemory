import json, re, hashlib
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"->->\s*[\d,\sN/A]*", "", t or "")).strip().lower()
def load(p):
    qs = json.load(open(p)); T, Q = set(), set()
    for q in qs:
        Q.add(norm(q["question"]))
        for s in q["haystack_sessions"]:
            for t in s:
                if t["role"] == "user":
                    x = norm(t["content"])
                    if len(x) >= 30: T.add(hashlib.md5(x.encode()).hexdigest())
    return len(qs), T, Q
pairs = [("LoCoMo dev conv 0-4 (all)", "q_loc_dev_all.json", "LoCoMo final conv 5-9", "q_loc_final150.json"),
         ("EverMem t01", "../bench/evermem/evermem_g.t01.json", "EverMem t05 (reserve)", "../bench/evermem/evermem_g.t05.json"),
         ("EverMem t02", "../bench/evermem/evermem_g.t02.json", "EverMem t05 (reserve)", "../bench/evermem/evermem_g.t05.json")]
for a, pa, b, pb in pairs:
    na, ta, qa = load(pa); nb, tb, qb = load(pb)
    print(f"{a} ({na} q, {len(ta)} turns) vs {b} ({nb} q, {len(tb)} turns): shared turns {len(ta & tb)}, identical questions {len(qa & qb)}", flush=True)
