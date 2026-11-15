"""D4 §7/§12.5 summary on H4. area = mean recall over 1k,2k,4k,8k,16k. Selection: H4s all-question area (ties -> earlier step).
usage: python3 d4/d4eval.py ARM:TAGPREFIX:STEPS ...   e.g. C:D4C:100,200,...,600   (ctx_<TAGPREFIX>s<step>_h4.jsonl)"""
import json, statistics as st, sys
BUD = [1000, 2000, 4000, 8000, 16000]
Q = {q["question_id"]: q for q in json.load(open("env_h4.json"))}
def grp(q): return any("group" in c for c in q.get("carriers", []))
def area(tag):
    E = {}
    for r in map(json.loads, open(f"ctx_{tag}.jsonl")):
        if r["gold_total"]: E[(r["qid"], r["budget"])] = r["gold_in"] / r["gold_total"]
    out = {}
    for half, par in (("H4s", 0), ("H4j", 1)):
        qs = [k for k in Q if int(k.split("_")[0][1:]) % 2 == par and (k, 1000) in E]
        miss = [k for k in Q if int(k.split("_")[0][1:]) % 2 == par and (k, 1000) not in E]
        a = {k: st.mean(E[(k, b)] for b in BUD) for k in qs}
        out[half] = dict(all=100 * st.mean(a.values()), group=100 * st.mean(a[k] for k in qs if grp(Q[k])), n=len(qs), miss=len(miss))
    return out
res = {}
for spec in sys.argv[1:]:
    arm, pre, steps = spec.split(":"); R = {int(s): area(f"{pre}s{s}_h4") for s in steps.split(",")}
    best = max(sorted(R), key=lambda s: (round(R[s]["H4s"]["all"], 6), -s)); last = max(R)
    res[arm] = dict(sel=best, sel_r=R[best], end=last, end_r=R[last], all=R)
    print(f"{arm:5s} " + " ".join(f"s{s}:{R[s]['H4s']['all']:.1f}" for s in sorted(R)) + f" | sel s{best}: H4j all {R[best]['H4j']['all']:.1f} group {R[best]['H4j']['group']:.1f} | end s{last}: H4j all {R[last]['H4j']['all']:.1f} group {R[last]['H4j']['group']:.1f}")
json.dump(res, open("d4/d4eval_last.json", "w"), indent=1)
def d(a, b, k, w="sel"): return res[a][f"{w}_r"]["H4j"][k] - res[b][f"{w}_r"]["H4j"][k]
if "P0P1" in res and "C" in res:
    for w in ("sel", "end"):
        g, al = d("P0P1", "C", "group", w), d("P0P1", "C", "all", w); line = f"[{w}] P0P1-C group {g:+.1f} all {al:+.1f} -> crit1 {'PASS' if g >= 3 and al >= -1 else 'FAIL'}"
        if "P0" in res: p = d("P0P1", "P0", "all", w); line += f" | P0P1-P0 all {p:+.1f} -> crit2 {'PASS' if p >= 1 else 'FAIL'} | P0-C group {d('P0','C','group',w):+.1f} all {d('P0','C','all',w):+.1f}"
        print(line)
