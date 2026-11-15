"""tests of the EverMem multiple-choice letter rule (a6judge.mc_letter), then a dry run over every stored EverMem answer"""
import collections, json, os, sys
sys.path.insert(0, ".")
import a6judge as J
q = {"question": "Pick one.\nOptions:\nA. " + "alpha " * 10 + "\nB. " + "beta " * 10 + "\nC. " + "gamma " * 10 + "\nD. " + "delta " * 10, "answer": "D. delta"}
CASES = [("D", "D"), ("D.", "D"), ("**D**", "D"), ("Answer: **D**\nbecause", "D"), ("The answer is D.", "D"), ("Final answer: C", "C"),
         ("A woman asked about it.", None), ("The answer is a woman.", None), ("Answer: A; correction: B", None),
         ("D. Unlike option B, this keeps the logs.", "D"), ("I don't know.", None), ("I pick " + ("delta " * 10).strip(), "D"),
         ("<think>maybe B</think>C.", "C"), ("Option C is best.", "C"),
         ("Answer: **A**\n\n- **B** is not suitable.\n- **C** is off.", "A"), ("D.\n\nOption A is incorrect.", "D"),
         ("**Answer: D**\n\nTherefore, option **D** is correct; **A** is wrong.", "D"), ("Answer: C ... final answer: B", None),
         ("The correct answer is B.", "B"), ("Options **A** and **D** are wrong.", None),
         ("Option B is incorrect; I cannot determine the answer.", None), ("Answer: A; correction: option B.", None), ("Option B is not right.", None)]
bad = [(t, e, J.mc_letter(q, t)) for t, e in CASES if J.mc_letter(q, t) != e]
assert J.mc_gold(q) == "D" and J.mc_gold({"question": "x", "answer": "D. y"}) is None, "mc_gold"
assert not bad, f"letter rule failures: {bad}"
print("unit cases PASS", len(CASES))
if os.path.exists("q_ev150.json"):
    Q = {x["question_id"]: x for x in json.load(open("q_ev150.json"))}; mc = {i for i in Q if J.mc_gold(Q[i])}
    rows = [r for r in map(json.loads, open("out/A6_part1.jsonl")) if r["stratum"] == "ev" and "error" not in r and r["qid"] in mc]
    c = collections.Counter("parsed" if J.mc_letter(Q[r["qid"]], r["answer"]) else "to LLM" for r in rows); print("MC questions", len(mc), "answers", dict(c))
    for r in rows:
        if not J.mc_letter(Q[r["qid"]], r["answer"]): print("  to LLM:", r["m"], r["qid"], repr(r["answer"][:160]))
