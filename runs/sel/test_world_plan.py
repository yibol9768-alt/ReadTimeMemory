"""regression: world.plan with --exclude-from off must equal the 11-02 backup bit for bit (stub LLM, no GPU);
with it on, the project and every person name must be new."""
import argparse, importlib.util, json, re, sys
sys.path.insert(0, ".")
def load(path, name):
    from importlib.machinery import SourceFileLoader
    sp = importlib.util.spec_from_loader(name, SourceFileLoader(name, path)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
class Stub:
    def __init__(self, *a): pass
    def chat(self, p, **k):
        if p.startswith("A person works for months"): return [json.dumps({"entities": [f"E{i}" for i in range(14)], "categories": [f"cat {i}" for i in range(8)]})]
        ids = re.findall(r"^(f\d+) \|", p, re.M)
        return [json.dumps([{"id": i, "attr": f"a{i}", "value": f"v{i}", "statement": f"When I ask about x, {i}"} for i in ids])]
new, old = load("world.py", "wnew"), load("world.py.bak-1102", "wold")
for m in (new, old): m.Pool = Stub
base = dict(q27="x", gem="x", oss="x", rounds=60, projects="/home/centuser/data/rtm/env/projects.json", bg="/home/centuser/data/rtm/env/bg", par=1, fu=.15)
for knobs in ({}, {"p_group": .4, "p_long": .3, "p_dated": .15, "p_greet": .1, "p_sib": .3}):
    for i in (10, 990, 2000):
        a_new = argparse.Namespace(**base, knobs=knobs, exclude_from="", team_q=0); a_old = argparse.Namespace(**base, knobs=knobs)
        assert json.dumps(new.plan(i, a_new), sort_keys=True) == json.dumps(old.plan(i, a_old), sort_keys=True), (i, knobs)
print("flag off: identical to backup for 6 cases")
a = argparse.Namespace(**base, out="ws8", knobs={"p_group": .4, "p_long": .3, "p_dated": .15, "p_greet": .1, "p_sib": .3}, exclude_from="ws1,ws3,ws4,ws5p,ws6,ws6e,ws7", team_q=0)
used, N = new.excluded(a)
for i in range(2000, 2012):
    w = new.plan(i, a); ppl = {w["name"], *w.get("team", [])}
    assert w["project"] not in used and not (ppl & N) and w["name"] not in w.get("team", []), i
print("flag on: 12 worlds, new projects and names; used projects", len(used), "used names", len(N))
