"""Are the top non-gold rounds real negatives? (11-01) 27B judges the 10 best-scored non-gold, non-support rounds of 60 multi-evidence training questions.
False-negative rate under 10%: hard negatives are kept (hard-skip 0); otherwise skip the top 10 as before. Writes hs.txt.
usage: python3 fnchk.py QFILE POOLDIR IDS"""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.expanduser("~/data/rtm/env"))
from k1 import rounds_of
from fmt import split_round
from gen import Pool
from mkdata import kinds_of
J = """Question about a user's past chats: {q}
One past message from the user: {m}
Does this message contain something that belongs in the answer? Answer YES or NO."""
qf, pd, ids = sys.argv[1:4]; keep = set(open(ids).read().split()); rng = random.Random(0)
Q = [q for q in json.load(open(qf)) if q["question_id"] in keep and len(q["gold_stages"]) > 1]; rng.shuffle(Q); Q = Q[:60]; p = Pool("Qwen3.8-27B-FP8@18303")
def one(q):
    P = json.load(open(f"{pd}/{q['question_id']}.json")); R = rounds_of(q); sup = set(q.get("support_stages", [])); st = P["stage"]; rr = {int(k): v for k, v in P["rr8"].items()}
    ks = kinds_of(q); F = set(P["fused"][:600]); G = set(q["gold_stages"])   # same negative set as mkdata --ignore-kinds repeat,fu
    neg = [i for i in sorted(rr, key=lambda i: -rr[i]) if i in F and st[i] not in G and st[i] not in sup and not (i < len(ks) and ks[i] in ("repeat", "fu"))][:10]
    hit = [i for i in neg if "YES" in p.chat(J.format(q=q["question"], m=split_round(R[i][2])[0][:1500]), max_tokens=8, temperature=0)[0].upper()]
    for i in hit[:1]: print("YES:", q["question"][:90], "=>", split_round(R[i][2])[0][:130].replace("\n", " "), flush=True)
    pos = [i for i in rr if P["gold"][i]][:5]
    ph = sum("YES" in p.chat(J.format(q=q["question"], m=split_round(R[i][2])[0][:1500]), max_tokens=8, temperature=0)[0].upper() for i in pos)
    return len(hit), len(neg), ph, len(pos)
with ThreadPoolExecutor(12) as ex: res = list(ex.map(one, Q))
y, n = sum(r[0] for r in res), sum(r[1] for r in res); py, pn = sum(r[2] for r in res), sum(r[3] for r in res)
rate = y / n if n else 1.0; rec = py / pn if pn else 0.0; adj = rate / rec if rec else 1.0
open("hs.txt", "w").write("0" if rec >= .7 and adj < .1 else "10")   # the judge must recognise real evidence before its NO is believed; else keep the old skip
print("false negatives", y, "of", n, round(rate, 3), "| gold recognised", py, "of", pn, round(rec, 3), "| adjusted", round(adj, 3), "hard-skip", open("hs.txt").read())
