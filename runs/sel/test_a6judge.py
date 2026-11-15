"""offline tests of a6judge parsing: stub chat(), no model. run from runs/sel with S1 stubs."""
import sys, types, json, os
sys.argv = ["x"]
S1 = sys.argv  # placeholder
here = os.path.dirname(os.path.abspath(__file__))
# stub modules a6judge imports lazily
qa = types.ModuleType("qa"); qa.lst = lambda x: x if isinstance(x, list) else json.loads(x); sys.modules["qa"] = qa
judge = types.ModuleType("judge"); judge.ev = types.SimpleNamespace(get_anscheck_prompt=lambda *a, **k: "P"); sys.modules["judge"] = judge
os.makedirs(os.path.expanduser("~/data/rtm/s1"), exist_ok=True) if False else None
import importlib.util
src = open(sys.path[0] + "/../../../../a6judge.py").read() if False else open(os.environ["A6J"]).read()
src = src.replace('BEAM_JP = open(os.path.join(S1, "beam_judge_prompt.txt")).read()', 'BEAM_JP = "<rubric_item> <llm_response>"')
m = types.ModuleType("a6j"); exec(compile(src, "a6judge.py", "exec"), m.__dict__)
out = []
def run(s, q, resp, ans="A"):
    m.chat = lambda *a, **k: resp
    try: r = m.score(0, s, q, ans); out.append((s, repr(resp)[:40], "ok", r[0]))
    except m.JudgeFail as e: out.append((s, repr(resp)[:40], "FAIL", None))
beam = {"question_id": "h1_x", "rubric": ["item"]}
run("hard", beam, '{"score": 1}'); run("hard", beam, '{"score": 0.5}'); run("hard", beam, '{"score": 1e3}'); run("hard", beam, '{"score": 10}'); run("hard", beam, 'nothing')
lme = {"question_id": "a", "question_type": "multi-session", "question": "q", "answer": "x"}
run("lme", lme, "Yes."); run("lme", lme, "no"); run("lme", lme, "Unable to evaluate this response.")
ep = {"question_id": "p", "question_type": "Entities|all", "answer": "A; B", "answer_items": "['A', 'B']"}
run("ep", ep, '{"identified_items_in_AI_answer": ["A","B"], "matching_score": [{"A": 1}, {"B": 1}], "explanation": ""}')
run("ep", ep, '{"identified_items_in_AI_answer": ["A"], "matching_score": "10", "explanation": ""}')
run("ep", ep, '{"identified_items_in_AI_answer": [], "matching_score": [{"A": 0}, {"B": 0}], "explanation": ""}')
ev = {"question_id": "e", "question": "q", "answer": "x", "options": "None"}
run("ev", ev, '{"label": "CORRECT"}'); run("ev", ev, 'The answer is INCORRECT.'); run("ev", ev, 'garbage')
loc = {"question_id": "loc0_1", "question_type": "single-hop", "answer": "She got hurt"}
m.chat = None; print("loc f1 gold", m.loc_score(loc, "She got hurt"), "wrong", m.loc_score(loc, "I don't know."))
for o in out: print(o)
