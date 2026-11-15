"""Convert BEAM (1M / 10M parquet) into LongMemEval-style question objects so k1.py builders work unchanged.
Each probing question -> {question_id, question_type=ability, question, answer, question_date, haystack_sessions,
haystack_dates}; a turn gets has_answer=True if its id is in the question's source_chat_ids (nested lists/dicts flattened).
Questions without source ids (abstention) are kept with no gold (completeness is computed only where gold exists).
usage: python beam2lme.py PARQUET[,PARQUET] CONV_IDS(comma or 'all') OUT.json"""
import ast, json, sys
import pandas as pd

def flat(x):
    if isinstance(x, dict): return [i for v in x.values() for i in flat(v)]
    if isinstance(x, (list, tuple)): return [i for v in x for i in flat(v)]
    try: return [int(x)]
    except Exception: return []

def lit(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return json.loads(x)
    return x

files, ids, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
PFX = sys.argv[4] if len(sys.argv) > 4 else "b"   # use a distinct prefix per BEAM size: conversation ids restart at 1
df = pd.concat([pd.read_parquet(f) for f in files])
want = None if ids == "all" else set(ids.split(","))
qs = []
for _, r in df.iterrows():
    cid = str(r["conversation_id"])
    if want and cid not in want: continue
    chat = lit(r["chat"]); pq = lit(r["probing_questions"])
    sessions, dates = [], []
    for s in chat:
        s = list(s); sessions.append(s); dates.append(str(s[0].get("time_anchor", "")) if s else "")
    last = next((d for d in reversed(dates) if d), "")
    for ab, lst in pq.items():
        for j, x in enumerate(lst):
            raw = x.get("source_chat_ids", [])
            src = set(flat(raw))
            groups = [flat(g) for g in raw] if isinstance(raw, (list, tuple)) and raw and all(isinstance(g, (list, tuple)) for g in raw) else []
            hs = [[{**t, "has_answer": int(t.get("id", -1)) in src if str(t.get("id", "")).lstrip("-").isdigit() else False} for t in s] for s in sessions]
            qs.append({"question_id": f"{PFX}{cid}_{ab}_{j}", "question_type": ab, "question": x["question"],
                       "answer": str(x.get("answer") or x.get("ideal_answer") or x.get("ideal_response") or x.get("ideal_summary") or ""),
                       "question_date": last, "haystack_sessions": hs, "haystack_dates": dates, "n_src": len(src), "gold_groups": groups,
                       "rubric": [str(i) for i in (lit(x.get("rubric")) if x.get("rubric") else [])]})
json.dump(qs, open(out, "w"))
print(len(qs), "questions", sum(q["n_src"] > 0 for q in qs), "with gold")
