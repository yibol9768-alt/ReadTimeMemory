"""Fix budgets for EpBench and EverMem from untrained baselines only (11-01). Largest budget with baseline recall <= 80%. Writes budgets.json."""
import json, statistics as st
from ssum import score
out = {}
for n in ("ep150", "ev150"):
    A = {t: score(t + "_" + n, n) for t in ("A0", "A0O")}; ok = []
    for b in (1000, 2000, 4000, 16000):
        m = max(st.mean(v for (q, x), v in A[t].items() if x == b) for t in A); print(n, b, round(100 * m, 1))
        if m <= .8: ok.append(b)
    out[n] = max(ok) if ok else None   # no unsaturated budget: that benchmark is invalid, never a silent fallback
json.dump(out, open("budgets.json", "w")); print(out)
