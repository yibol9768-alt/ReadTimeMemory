"""Regression for the 11-01 packing fix: python3 test_infer.py OLD_INFER.py. Uses mocked scores, no server. Exit 1 on failure."""
import hashlib, importlib.util, json, os, random, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
new = load(os.path.join(here, "infer.py"), "inf_new"); old = load(sys.argv[1], "inf_old")
def mock(path, payload):
    q = payload["query"]; res = []
    for k, d in enumerate(payload["documents"]):
        v = int(hashlib.md5((q + d).encode()).hexdigest()[:6], 16) / 2**24
        res.append({"index": k, "relevance_score": round(v, 2)})   # coarse, so ties happen
    return {"results": res}
r = random.Random(0); WORD = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
def turn(n): return " ".join(r.choice(WORD) for _ in range(n))
S = []
for s in range(30):
    T = []
    for t in range(4):
        n = r.choice([20, 80, 300, 700, 3000])   # 3000 words is oversized for the 1k budget
        T += [{"role": "user", "content": turn(n), **({"has_answer": True} if r.random() < .1 else {})}, {"role": "assistant", "content": turn(r.choice([10, 60, 200]))}]
    S.append(T)
Q = [{"question_id": "t0", "question": "what", "question_date": "2025/01/01", "question_type": "x", "answer": "a", "haystack_sessions": S, "haystack_dates": [f"2024/01/{1 + i % 28:02d}" for i in range(30)]}]
d = tempfile.mkdtemp(); os.chdir(d); json.dump(Q, open("q.json", "w")); os.mkdir("p")
R = new.rounds_of(Q[0]); F = list(range(len(R))); r.shuffle(F); json.dump({"fused": F}, open("p/t0.json", "w"))
def run(mod, budgets, tag, *extra):
    mod.BUDGETS = budgets; mod._post = mock; sys.argv = ["infer", "q.json", "--port", "1", "--pooldir", "p", "--tag", tag, "--par", "1", *extra]
    if os.path.exists(f"ctx_{tag}.jsonl"): os.remove(f"ctx_{tag}.jsonl")
    out = sys.stdout; sys.stdout = open(os.devnull, "w"); mod.main(); sys.stdout = out
    return {x["budget"]: x for x in map(json.loads, open(f"ctx_{tag}.jsonl"))}
bad = []
def ck(name, c):
    print("ok  " if c else "FAIL", name); bad.append(name) if not c else None
C0 = ("--cond", "0")
single = {b: run(new, [b], f"s{b}", *C0)[b]["keep"] for b in (1000, 2000, 4000)}
m = run(new, [1000, 2000, 4000], "m", *C0); rev = run(new, [4000, 2000, 1000], "r", *C0)
ck("multi equals single, every budget", all(m[b]["keep"] == single[b] for b in single))
ck("budget order does not matter", all(m[b]["keep"] == rev[b]["keep"] for b in single))
ck("old and new single budget equal", all(run(old, [b], f"o{b}", *C0)[b]["keep"] == single[b] for b in single))
om = run(old, [1000, 2000, 4000], "om", *C0); print("     old multi differed at", [b for b in (1000, 2000) if om[b]["keep"] != single[b]])
ck("largest budget equal old multi", om[4000]["keep"] == m[4000]["keep"])
ck("ties stable across runs", run(new, [1000], "t", *C0)[1000]["keep"] == single[1000])
ck("budget respected", all(sum(new.ntok(R[i][2]) for i in m[b]["keep"]) <= b for b in single))
ck("oversized never kept at 1k", all(new.ntok(R[i][2]) <= 1000 for i in m[1000]["keep"]))
ck("expansion path as before", run(new, [1000, 4000], "x", "--cond", "0", "--xexp", "1")[1000]["keep"] == run(old, [1000, 4000], "ox", "--cond", "0", "--xexp", "1")[1000]["keep"])
ck("conditional single budget as before", run(new, [1000], "c", "--cond", "1", "--M", "8")[1000]["keep"] == run(old, [1000], "oc", "--cond", "1", "--M", "8")[1000]["keep"])
# known-bad case from the audit: score order lengths 3500, 600 (gold), 500. Old multi-budget loses the gold at 1k, must show as a difference (negative control)
def rd(mark, tok):
    n = 1
    while True:
        Qx = [{"question_id": "t1", "question": "what", "question_date": "2025/01/01", "question_type": "x", "answer": "a", "haystack_sessions": [[{"role": "user", "content": mark + " " + " ".join(["word"] * n)}, {"role": "assistant", "content": "ok"}]], "haystack_dates": ["2024/01/01"]}]
        if new.rounds_of(Qx[0]) and new.ntok(new.rounds_of(Qx[0])[0][2]) >= tok: return Qx[0]["haystack_sessions"][0]
        n += 10
sess = [rd("AAA", 3500), rd("BBB", 600), rd("CCC", 470)]; sess[1][0]["has_answer"] = True
Q2 = [{"question_id": "t1", "question": "what", "question_date": "2025/01/01", "question_type": "x", "answer": "a", "haystack_sessions": sess, "haystack_dates": ["2024/01/01", "2024/01/02", "2024/01/03"]}]
json.dump(Q2, open("q.json", "w")); json.dump({"fused": [0, 1, 2]}, open("p/t1.json", "w"))
SC = {"AAA": .9, "BBB": .8, "CCC": .7}
def mock2(path, payload): return {"results": [{"index": k, "relevance_score": next(v for m, v in SC.items() if m in d)} for k, d in enumerate(payload["documents"])]}
mock = mock2
n1 = run(new, [1000, 4000], "n2", *C0)[1000]["keep"]; o1 = run(old, [1000, 4000], "o2", *C0)[1000]["keep"]; s1 = run(new, [1000], "s2", *C0)[1000]["keep"]
print("     new", n1, "old", o1, "single", s1)
ck("new multi keeps gold round 1 at 1k", 1 in n1 and n1 == s1)
ck("old multi really had the bug (negative control)", 1 not in o1)
sys.exit(1 if bad else 0)
