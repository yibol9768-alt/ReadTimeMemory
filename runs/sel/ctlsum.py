"""D4 §5/§12.4 filter controls: per judge YES rate; POS >= 0.90, HARD <= 0.10, FAR <= 0.05; both judges must pass. INVALID counts as not YES."""
import json, sys, collections
V = [json.loads(l) for l in open(sys.argv[1])]; V = [r for r in V if r.get("version") == "fnfilter-v1.1-2026-11-03"]
TH = {"POS": (">=", .90), "HARD": ("<=", .10), "FAR": ("<=", .05)}; ok = True
for c, (op, t) in TH.items():
    R = [r for r in V if r["key"].startswith(c + "|")]; err = sum("error" in r for r in R); R = [r for r in R if "error" not in r]
    line = f"{c:4s} n={len(R)} err={err}"
    for j in ("q27", "oss"):
        y = sum(r[j] == "YES" for r in R) / max(1, len(R)); inv = sum(r[j] == "INVALID" for r in R)
        p = y >= t if op == ">=" else y <= t; ok &= p and len(R) == 40
        line += f" | {j} YES {y:.3f} INVALID {inv} {'PASS' if p else 'FAIL'}"
    both = sum(r["q27"] == "YES" and r["oss"] == "YES" for r in R) / max(1, len(R)); line += f" | both YES {both:.3f}"
    print(line)
    for r in R:
        bad = (c == "POS" and (r["q27"] != "YES" or r["oss"] != "YES")) or (c != "POS" and (r["q27"] == "YES" or r["oss"] == "YES"))
        if bad: print("   ", r["key"], r["q27"], r["oss"])
print("CONTROLS", "PASS" if ok else "FAIL")
