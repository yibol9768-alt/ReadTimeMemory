"""LoCoMo to the question format used here (11-01). Every utterance is one round, role user, "Name: text" (with the image caption when present), like a group chat. Gold = turns named in `evidence`. Category 5 (adversarial, no answer) left out.
Conversations 0-4 are development, 5-9 are kept for the final test. Writes q_loc150.json: 150 development questions, seed 20261101. usage: python3 locomo.py locomo10.json
11-02: optional args LO HI OUT N (e.g. 0 5 q_loc_dev_all.json 0 = all of conversations 0-4; 5 10 q_loc_final.json 150 = final sample); defaults reproduce q_loc150.json exactly."""
import json, random, re, sys
D = json.load(open(sys.argv[1])); NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}; out = []; bad = 0
LO, HI, OUT, N = (int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])) if len(sys.argv) > 5 else (0, 5, "q_loc150.json", 150)
for ci, x in enumerate(D[LO:HI], start=LO):
    c = x["conversation"]; ks = sorted((int(m.group(1)) for k in c for m in [re.fullmatch(r"session_(\d+)", k)] if m)); S = []; dates = []
    for k in ks:
        dates.append(c[f"session_{k}_date_time"]); S.append([dict(role="user", content=f"{t['speaker']}: {t['text']}" + (f" [shared image: {t['blip_caption']}]" if t.get("blip_caption") else ""), id=t["dia_id"]) for t in c[f"session_{k}"]])
    ids = {t["id"] for s in S for t in s}
    for qi, q in enumerate(x["qa"]):
        if q.get("category") == 5 or "answer" not in q: continue
        ev = {e.strip() for e in q.get("evidence", []) if e.strip() in ids}
        if not ev: bad += 1; continue
        H = [[dict(t, **({"has_answer": True} if t["id"] in ev else {})) for t in s] for s in S]
        out.append(dict(question_id=f"loc{ci}_{qi}", question=q["question"], question_date=dates[-1], question_type=NAME[q["category"]], answer=str(q["answer"]), haystack_sessions=H, haystack_dates=dates, carriers=["group"]))
r = random.Random(20261101); r.shuffle(out); s = out[:N] if N else out; json.dump(s, open(OUT, "w")); import collections
print("usable", len(out), "dropped no evidence", bad, "sample", len(s), collections.Counter(q["question_type"] for q in s))
