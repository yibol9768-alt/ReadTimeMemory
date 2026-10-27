# LME dev: per reader arm, mean over available gens of note3_deep vs other conds on shared qids; paired bootstrap over questions.
import json, os, random, sys
arms = sys.argv[1:] or ["pro6", "pro_gpt-6.1-sol", "ds41_1017"]
for a in arms:
    G = []
    for g in ["", "_g2", "_g3"]:
        f = f"runs/{os.environ.get('LME_DIR', 'lme')}/{a}/judged{g}.jsonl"
        if os.path.exists(f):
            S = {}
            for l in open(f):
                r = json.loads(l); S.setdefault(r["cond"], {})[r["qid"]] = r["score"]
            G.append(S)
    for other in ["lychee", "hindsight"]:
        gs = [S for S in G if "note3_deep" in S and other in S]
        if not gs: continue
        q = sorted(set.intersection(*[set(S["note3_deep"]) & set(S[other]) for S in gs]))
        d = {k: sum(S["note3_deep"][k] - S[other][k] for S in gs) / len(gs) for k in q}
        m = lambda c: 100 * sum(S[c][k] for S in gs for k in q) / len(q) / len(gs)
        rng = random.Random(0); bs = sorted(100 * sum(d[rng.choice(q)] for _ in q) / len(q) for _ in range(5000))
        print(f"{a:16s} gens={len(gs)} n={len(q)} note3_deep {m('note3_deep'):.1f} {other} {m(other):.1f} "
              f"diff {100*sum(d.values())/len(q):+.1f} [{bs[125]:+.1f},{bs[4875]:+.1f}] win:loss {sum(v>0 for v in d.values())}:{sum(v<0 for v in d.values())}")
