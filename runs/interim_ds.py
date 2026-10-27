import json, os, collections
for S in ("beam1m", "beam"):
    P = f"runs/{S}/ds41_1017/"; route = json.load(open(P + "route.json"))
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in ("", "_g2", "_g3"):
        f = P + f"judged{g}.jsonl"
        if not os.path.exists(f): continue
        seen = set()
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if (r["cond"], r["qid"]) in seen: continue
            seen.add((r["cond"], r["qid"])); d[r["cond"]][r["qid"]].append(r["score"])
    m = lambda c, Q: 100 * sum(sum(d[c][q]) / len(d[c][q]) for q in Q) / len(Q)
    print(S, {c: sum(len(v) for v in d[c].values()) for c in d})
    for R, cs in (("POINT", ["note3_ly~m0", "note3_deep~m0", "lychee_cf~m0", "cf_hindsight~m0", "hindsight~m0"]),
                  ("AREA", ["note3_deep~m0", "lychee_b60~m0", "hindsight~m0"])):
        cs = [c for c in cs if c in d]
        Q = [q for q in route if route[q] == R and all(q in d[c] for c in cs)]
        if len(Q) >= 10: print(f"  {R} n={len(Q)}", {c: round(m(c, Q), 1) for c in cs})
