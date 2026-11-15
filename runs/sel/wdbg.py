"""Debug one aggregation and one direct question through the sufficiency gate; print every step."""
import json, sys, random
sys.argv = sys.argv[:1]
import world as W
from gen import Pool
Q = json.load(open("wt/q00001.json")); rp = Pool("Qwen3.8-27B-FP8@18301")
for q in [next(x for x in Q if x["question_type"] == "aggregation"), next(x for x in Q if x["question_type"] == "direct")]:
    R = [(d, s[j:j + 2]) for s, d in zip(q["haystack_sessions"], q["haystack_dates"]) for j in range(0, len(s), 2)]
    keys = set(q["gold_stages"]); gold = [x for x in R if x[1][0]["stage"] in keys]
    print("Q:", q["question"], "| keys", len(keys), "gold rounds", len(gold))
    ctx = "\n".join(f"({d}) User: {t[0]['content']}\nAssistant: {t[1]['content'][:600]}" for d, t in gold)
    print("CTX head:", ctx[:400].replace("\n", " / "))
    t, m = rp.chat(W.READ.format(ctx=ctx, qdate=q["question_date"], q=q["question"]), max_tokens=500, temperature=0)
    print("MODEL", m, "RESP len", len(t), ":", t[:900].replace("\n", " / "))
    for x in q["rubric"][:4]:
        y = x.replace("LLM response should contain: ", ""); j = rp.chat(W.JUDGE.format(q=q["question"], r=t[:3000], x=y), max_tokens=8, temperature=0)[0]
        print("  RUB", y[:130], "->", repr(j))
