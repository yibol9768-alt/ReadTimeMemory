"""Benchmark subsets for the main criterion (10-31): 150 questions, stratified by type, seed 0, only questions with gold.
EpBench-200 drops zero-gold questions. EverMemBench is scored at session level later. Writes q_ep150.json, q_ev150.json and the untouched-rest id lists."""
import collections, json, random
rng = random.Random(0)
for tag, fn, skip in (("ep150", "/home/centuser/data/rtm/bench/epbench/epbench200_g.json", lambda q: False), ("ev150", "/home/centuser/data/rtm/bench/evermem/evermem_g.t01.json", lambda q: False)):
    Q = [q for q in json.load(open(fn)) if not skip(q) and any(t.get("has_answer") in (True, 1, "True") for s in q["haystack_sessions"] for t in s)]
    by = collections.defaultdict(list)
    for q in Q: by[q["question_type"]].append(q)
    pick = []; ks = sorted(by)
    for k in ks: rng.shuffle(by[k])
    while len(pick) < 150 and any(by.values()):
        for k in ks:
            if by[k] and len(pick) < 150: pick.append(by[k].pop())
    for q in pick: q["question_date"] = q["question_date"] or "unknown"
    json.dump(pick, open(f"q_{tag}.json", "w")); print(tag, len(Q), "->", len(pick), collections.Counter(q["question_type"] for q in pick).most_common(6))
