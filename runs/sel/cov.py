"""Evidence metrics for ctx_<TAG>.jsonl files (uses the 'keep' round indices written by k1.pack).
  env questions : stage coverage = |stages of kept rounds ∩ gold_stages| / |gold_stages|, all-covered rate
  BEAM / LME    : round recall = gold_in / gold_total; group coverage when gold_groups (nested source ids) exist
Also: `python3 cov.py rr8 QFILE TAG` builds the 8B-reranker baseline ctx straight from the pools (no model calls).

usage: python3 cov.py report QFILE TAG[,TAG...] [--pooldir pool]
       python3 cov.py rr8 QFILE TAG [--T 300]"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of, pack, BUDGETS


def round_turn_ids(q):
    out = []
    for sess in q["haystack_sessions"]:
        i = 0
        while i < len(sess):
            ids = [sess[i].get("id")]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant":
                ids.append(sess[i + 1].get("id")); i += 1
            i += 1; out.append(ids)
    return out


def main():
    cmd, qfile, tags = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    pooldir = sys.argv[sys.argv.index("--pooldir") + 1] if "--pooldir" in sys.argv else "pool"
    T = int(sys.argv[sys.argv.index("--T") + 1]) if "--T" in sys.argv else 600
    XEXP = "--xexp" in sys.argv
    qs = {q["question_id"]: q for q in json.load(open(qfile))}
    if cmd == "rr8":
        with open(f"ctx_{tags[0]}.jsonl", "w") as f:
            for qid, q in qs.items():
                pp = os.path.join(pooldir, qid + ".json")
                if not os.path.exists(pp): continue
                pool = json.load(open(pp)); R = rounds_of(q)
                rr = {int(k): v for k, v in pool["rr8"].items()}
                top = sorted(pool["fused"][:T], key=lambda i: -rr.get(i, -1))
                if XEXP:   # exchange expansion: each chosen round is followed by its short follow-ups
                    from infer import followups
                    seen, ex = set(), []
                    for i in top:
                        for j in [i] + followups(R, i, None):
                            if j not in seen: seen.add(j); ex.append(j)
                    top = ex
                order = top + [i for i in pool["fused"][T:] if i not in set(top)]
                order += [i for i in range(len(R)) if i not in set(order)]
                for B in BUDGETS: f.write(json.dumps({**pack(q, R, order, B), "hist": tags[0]}) + "\n")
        print("wrote", f"ctx_{tags[0]}.jsonl"); return
    for tag in tags:
        rows = [json.loads(l) for l in open(f"ctx_{tag}.jsonl")]
        agg = collections.defaultdict(list)
        for r in rows:
            q = qs.get(r["qid"])
            if q is None or "keep" not in r: continue
            B, t = r["budget"], q["question_type"]
            if "gold_stages" in q:
                pool = json.load(open(os.path.join(pooldir, r["qid"] + ".json")))
                got = {pool["stage"][i] for i in r["keep"]} & set(q["gold_stages"])
                c = len(got) / len(q["gold_stages"])
                agg[(B, t)].append((c, float(c == 1.0)))
            elif r["gold_total"]:
                rec = r["gold_in"] / r["gold_total"]
                g = q.get("gold_groups") or []
                if g:
                    tid = round_turn_ids(q); kept = {x for i in r["keep"] for x in tid[i]}
                    c = sum(any(x in kept for x in grp) for grp in g) / len(g)
                else:
                    c = rec
                agg[(B, t)].append((c, rec))
        print(f"== {tag}  (cov = stage/group coverage; 2nd = all-covered rate (env) or round recall (real))")
        for B in BUDGETS:
            allv = [v for (b, t), vs in agg.items() if b == B for v in vs]
            if not allv: continue
            line = f"  {B:6d} all n={len(allv):4d} cov {100 * sum(x[0] for x in allv) / len(allv):5.1f} / {100 * sum(x[1] for x in allv) / len(allv):5.1f} |"
            for (b, t), vs in sorted(agg.items()):
                if b == B: line += f" {t[:12]} {100 * sum(x[0] for x in vs) / len(vs):5.1f}"
            print(line)


if __name__ == "__main__":
    main()
