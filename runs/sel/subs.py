"""Draw a fixed number of training groups (D3_SPEC v2). usage: python3 subs.py IN.jsonl OUT.jsonl N [--weights type=w,...] [--hard LOSSFILE] [--seed 0]
--weights: sampling weight per question type (default 1); --hard: take the N groups with the highest loss in LOSSFILE (lines: index loss)."""
import argparse, json, random
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("N", type=int); ap.add_argument("--weights", default=""); ap.add_argument("--hard", default=""); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); L = open(a.inp).readlines()
if len(L) < a.N: raise SystemExit(f"INVALID: only {len(L)} groups, need {a.N}")
if a.hard:
    loss = {int(i): float(v) for i, v in (x.split() for x in open(a.hard))}
    if len(loss) != len(L): raise SystemExit(f"INVALID: {len(loss)} losses for {len(L)} groups")
    pick = sorted(range(len(L)), key=lambda i: -loss[i])[: a.N]
else:
    W = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)}
    w = [W.get(json.loads(x)["type"], 1.0) for x in L]; r = random.Random(a.seed)
    keys = sorted(range(len(L)), key=lambda i: -(r.random() ** (1.0 / w[i])))   # weighted sampling without replacement (Efraimidis-Spirakis)
    pick = keys[: a.N]
r = random.Random(a.seed + 1); r.shuffle(pick)
open(a.out, "w").writelines(L[i] for i in pick); print("wrote", len(pick), "of", len(L))
