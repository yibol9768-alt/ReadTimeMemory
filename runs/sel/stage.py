"""BEAM plan labels for every round, and point coverage (10-31, audit item 1). CPU only.
A round's user message carries index "N,M" = batch N, bullet M of conversation_plan; follow-ups inherit the last index.
Column = plan label before ':' ; point = (batch, bullet). Point coverage counts a gold point as covered when ANY round of
that point is selected, so a sibling follow-up is not penalised.
usage: alfenv python stage.py PARQUET QFILE IDFILE POOLDIR --out beam_stage.json"""
import argparse, collections, json, os, re, statistics as st, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of

ap = argparse.ArgumentParser(); ap.add_argument("parquet"); ap.add_argument("qfile"); ap.add_argument("ids"); ap.add_argument("pooldir")
ap.add_argument("--out", required=True); ap.add_argument("--types", default="event_ordering,summarization"); a = ap.parse_args()


def plan_of(text):
    out, b, m = {}, 0, 0
    for l in text.splitlines():
        h = re.match(r"\s*BATCH\s+(\d+)", l)
        if h: b, m = int(h.group(1)), 0; continue
        if l.strip().startswith("•"):
            m += 1
            lab = re.sub(r"\*", "", l.strip().lstrip("•").strip()).split(":")
            col = lab[0].strip()
            out[(b, m)] = (col, lab[1].strip() if col.endswith("Labels") and len(lab) > 1 else "")
    return out


D = pd.read_parquet(a.parquet); Plans = {str(r["conversation_id"]): plan_of(r["conversation_plan"]) for _, r in D.iterrows()}
keep = set(open(a.ids).read().split()); T = a.types.split(",")
Q = [q for q in json.load(open(a.qfile)) if q["question_id"] in keep and q["question_type"] in T and q["n_src"]]


def labels(q):
    """per round (batch, bullet) aligned with rounds_of: walks turns the same way (user + following assistant = one round)"""
    out, cur = [], None
    for s in q["haystack_sessions"]:
        i = 0
        while i < len(s):
            t = s[i]
            if t["role"] == "user":
                m = re.search(r"->->\s*(\d+)\s*,\s*(\d+)", t["content"]) or (re.match(r"(\d+),(\d+)", t.get("index") or ""))
                if m: cur = (int(m.group(1)), int(m.group(2)))
                if i + 1 < len(s) and s[i + 1]["role"] == "assistant": i += 1
            i += 1; out.append(cur)
    return out


stat = collections.defaultdict(lambda: collections.defaultdict(list)); dump = {}; miss = 0
for q in Q:
    qid = q["question_id"]; conv = qid.split("_")[0].lstrip("h"); R = rounds_of(q); L = labels(q)
    if len(L) != len(R): print("ALIGN", qid, len(L), len(R)); continue
    pl = Plans.get(conv, {}); p = json.load(open(os.path.join(a.pooldir, qid + ".json"))); g = p["gold"]
    rr = {int(k): v for k, v in p["rr8"].items()}; order = sorted(p["fused"], key=lambda i: -rr.get(i, -1))
    gi = [i for i in range(len(g)) if g[i]]; col = [pl.get(x, ("?", ""))[0] if x else "?" for x in L]
    miss += sum(c == "?" for c in col)
    cc = collections.Counter(col[i] for i in gi); top, n = cc.most_common(1)[0]
    d = stat[q["question_type"]]
    d["gold rounds"].append(len(gi)); d["top column share"].append(100 * n / len(gi)); d["columns"].append(len(cc))
    pts = {L[i] for i in gi}; d["gold points"].append(len(pts))
    same = [i for i in range(len(g)) if col[i] == top]; d["rounds in top column"].append(len(same))
    d["gold share of top column"].append(100 * n / len(same))
    ptR = [i for i in range(len(g)) if L[i] in pts]; d["rounds on gold points"].append(len(ptR)); d["gold share of those"].append(100 * len(gi) / max(1, len(ptR)))
    for k in (4, 16, 64):
        top_k = order[:k]
        d[f"round recall@{k}"].append(100 * sum(1 for i in top_k if g[i]) / len(gi))
        d[f"point cover@{k}"].append(100 * len({L[i] for i in top_k} & pts) / len(pts))
        d[f"top column hit@{k}"].append(100 * sum(1 for i in top_k if col[i] == top) / k)
    dump[qid] = {"point": [list(x) if x else None for x in L], "column": col, "top": top, "ignore": [i for i in range(len(g)) if col[i] == top and not g[i]]}
json.dump(dump, open(a.out, "w")); print("questions", len(dump), "rounds with no label", miss)
for t, d in stat.items():
    print("==", t, "n", len(d["gold rounds"]))
    for k, v in d.items(): print(f"  {k:28s} mean {st.mean(v):6.1f}  median {st.median(v):6.1f}")
