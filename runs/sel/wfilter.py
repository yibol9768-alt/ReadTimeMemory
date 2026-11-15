"""Keep questions that pass the gates; write env_ws1.json, world split id files. usage: python3 wfilter.py DIR W1,W2,.. HOLDOUT OUTTAG"""
import collections, json, os, sys
d, ws, hold, tag = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4]
all_, tr, dv, c = [], [], [], collections.Counter()
for i in ws:
    G = {g["qid"]: g for g in json.load(open(f"{d}/g{i:05d}.json"))}
    for q in json.load(open(f"{d}/q{i:05d}.json")):
        g = G[q["question_id"]]; multi = g["k"] > 1; c["all_" + q["question_type"]] += 1
        ok = g["blind"] < 0.5 and g["suff"] >= (0.6 if multi else 1.0) and (not multi or g.get("del", 1) < g["suff"])
        if q["question_type"] == "ordering" and ok: c["ordjudge_" + str(g.get("order"))] += 1   # judge passed only 8% on 11-01 and was never piloted: reported, not a gate
        if ok: all_.append(q); (dv if i == hold else tr).append(q["question_id"]); c["kept_" + q["question_type"]] += 1
json.dump(all_, open(f"env_{tag}.json", "w")); open(f"{tag}_tr.txt", "w").write("\n".join(tr)); open(f"{tag}_dv.txt", "w").write("\n".join(dv))
print(collections.Counter((q["question_type"], q.get("scope")) for q in all_).most_common(), collections.Counter(q.get("tilt") for q in all_))
print(sorted(c.items()), "train", len(tr), "holdout", len(dv))
