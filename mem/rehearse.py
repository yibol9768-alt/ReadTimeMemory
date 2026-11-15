"""WP5: can a rolling memory stop drifting by testing itself against the ledger?

WP4 showed that a memory rewritten from itself after every session (roll) loses early facts, while one rebuilt from
the append-only ledger of records (rebuild) keeps more of them. Rebuilding re-reads the whole ledger each time, which
grows without bound. Here the memory stays rolling, and every EVERY sessions it is checked against a small sample of
the ledger, the way the testing effect works in people:

  restudy  K records sampled from the ledger so far are shown again in the next rewrite ("re-read")
  quiz     the same K records are turned into questions, the current memory alone answers them, the answers are
           graded against the records, and only the facts the memory got wrong are shown in the next rewrite

Both arms see the same sample at the same sessions (paired); quiz adds 3 small calls per check and shows fewer
facts, so if it beats restudy, targeting what was forgotten matters, not just seeing old facts again.
Same sessions, budget, rewrite prompt, early-fact questions and checkpoints as mem/drift.py; roll and rebuild come
from drift's judged.jsonl, so the report puts all four side by side.

Usage: python3 -m mem.rehearse build|answer|report [--convs 0,2,4,5,7,8] [--par 3]
"""
import argparse, collections, json, math, os, random, re
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, drift as D

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "rehearse", llm.ARM)
EVERY, K = 3, 12
ARMS = ["restudy", "quiz"]
ALL_ARMS = ["restudy", "quiz", "selfquiz", "leitner", "pin", "prebuild", "trigger"]
# trigger:  tests are used only as a cheap drift monitor, never for repair: every EVERY sessions K ledger records
#           are probed; if the miss rate exceeds TRIGGER (default 0.4), the memory is rebuilt from the whole ledger
#           at the next session. Repair is zero-sum under a fixed budget (10-15 ledger-probe result), rebuilding is not.
# pin:      like quiz, but failed facts go into a pinned section that is never rewritten (PIN characters, oldest
#           evicted first); the rolling notes get BUDGET - PIN characters, so the total budget is unchanged
PIN = 2000
EVAL_FACTS = 80
EVERY_SESSION = False      # set by the tavg step: keep the memory after every session, not only at checkpoints
TRIGGER = 0.4
# prebuild: roll, but every EVERY sessions the memory is rebuilt from the whole ledger instead (periodic rebuild);
#           the baseline that spends tokens on re-reading everything rather than on tests
# selfquiz: probes written from the memory itself (MemDream-style)
# leitner:  every ledger record gets a probe when it is written; each check tests the K most due probes
#           (Leitner boxes: a passed probe moves up a box and waits 2^box checks, a failed one goes back to box 0)

QGEN_SYS = ("For each numbered memory record below, write one short question whose answer is the key detail of that "
            "record (a name, date, place, number, object or choice). Output one line per record: '<number>. <question>'.")
QANS_SYS = ("Answer each numbered question using only the memory notes provided. If the notes do not contain the "
            "answer, write 'unknown'. Output one line per question: '<number>. <answer>'.")
QSELF_SYS = ("Below are your own memory notes. Write {k} short question-answer pairs that test details in them (names, "
             "dates, places, numbers, choices), spread over the whole notes. Output one line per pair: "
             "'<number>. <question> || <answer>'.")
QGRADE_SYS = ("For each numbered item you get a reference fact, a question about it and an answer. Reply with one line "
              "per item: '<number>. RIGHT' if the answer matches the fact, otherwise '<number>. WRONG'.")


def numbered(lines):
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(lines))


def by_number(text, n):
    out = [""] * n
    for line in text.splitlines():
        head, _, rest = line.strip().partition(".")
        if head.strip().isdigit() and 1 <= int(head) <= n:
            out[int(head) - 1] = rest.strip()
    return out


def quiz(mem, recs):
    """Indices of the sampled records the current memory gets wrong."""
    texts = [r["text"] for r in recs]
    qs = by_number(llm.call(QGEN_SYS, numbered(texts), effort="low", max_tokens=4000, tag="rh_qgen"), len(texts))
    ans = by_number(llm.call(QANS_SYS, f"Memory notes:\n{mem}\n\nQuestions:\n{numbered(qs)}", effort="low",
                             max_tokens=4000, tag="rh_qans"), len(texts))
    items = [f"Fact: {t}\n   Question: {q}\n   Answer: {a}" for t, q, a in zip(texts, qs, ans)]
    grade = by_number(llm.call(QGRADE_SYS, numbered(items), effort="low", max_tokens=4000, tag="rh_qgrade"), len(texts))
    return [i for i, g in enumerate(grade) if "RIGHT" not in g.upper()]


def selfquiz(mem, k):
    """MemDream-style check: probes and their answers come from the memory itself, so a fact the memory has already
    lost can never be probed. Failed probes' answers are shown again in the next rewrite, as plain facts."""
    pairs = by_number(llm.call(QSELF_SYS.format(k=k), f"Memory notes:\n{mem}", effort="low", max_tokens=4000,
                               tag="rh_sgen"), k)
    pairs = [p.split("||") for p in pairs if "||" in p]
    qs, golds = [q.strip() for q, _ in pairs], [a.strip() for _, a in pairs]
    ans = by_number(llm.call(QANS_SYS, f"Memory notes:\n{mem}\n\nQuestions:\n{numbered(qs)}", effort="low",
                             max_tokens=4000, tag="rh_qans"), len(qs))
    items = [f"Fact: {q} -> {g}\n   Question: {q}\n   Answer: {a}" for q, g, a in zip(qs, golds, ans)]
    grade = by_number(llm.call(QGRADE_SYS, numbered(items), effort="low", max_tokens=4000, tag="rh_qgrade"), len(qs))
    return [{"text": f"{q} {g}"} for q, g, v in zip(qs, golds, grade) if "RIGHT" not in v.upper()]


def ask(mem, recs, qs):
    """Indices of the probes (pre-written questions about recs) that the memory gets wrong."""
    texts = [r["text"] for r in recs]
    ans = by_number(llm.call(QANS_SYS, f"Memory notes:\n{mem}\n\nQuestions:\n{numbered(qs)}", effort="low",
                             max_tokens=4000, tag="rh_qans"), len(qs))
    items = [f"Fact: {t}\n   Question: {q}\n   Answer: {a}" for t, q, a in zip(texts, qs, ans)]
    grade = by_number(llm.call(QGRADE_SYS, numbered(items), effort="low", max_tokens=4000, tag="rh_qgrade"), len(qs))
    return [i for i, g in enumerate(grade) if "RIGHT" not in g.upper()]


def probes_for(rs):
    """One question per record, written once when the record enters the ledger."""
    out = []
    for b in range(0, len(rs), 20):
        part = [r["text"] for r in rs[b:b + 20]]
        out += by_number(llm.call(QGEN_SYS, numbered(part), effort="low", max_tokens=4000, tag="rh_qgen"), len(part))
    return out


def rewrite(mem, s, extra):
    note = ""
    if extra:
        note = ("\n\nFacts from earlier sessions to keep in the memory (restore any that are missing):\n"
                + "\n".join("- " + r["text"] for r in extra))
    return L.within_budget(f"Your current memory of the earlier sessions:\n\n{mem or '(empty)'}\n\n"
                           f"The next session:\n\n{L.show_session(s)}{note}\n\n"
                           f"Write the updated memory of everything so far (at most {L.BUDGET} characters).",
                           "rh_roll")


def tag():
    """Name suffix for non-default settings (K, EVERY), so variants keep their own files and arm names."""
    return "" if (ARGS.k, ARGS.every) == (12, 3) else f"_k{ARGS.k}e{ARGS.every}"


def rewrite_to(mem, s, budget):
    """The roll rewrite with a smaller character limit (same instruction otherwise); cut after three tries."""
    prompt = (f"Your current memory of the earlier sessions:\n\n{mem or '(empty)'}\n\n"
              f"The next session:\n\n{L.show_session(s)}\n\n"
              f"Write the updated memory of everything so far (at most {budget} characters).")
    sys_ = L.SUM_SYS.replace(str(L.BUDGET), str(budget))
    text = llm.call(sys_, prompt, tag="rh_pin_roll")
    for i in range(3):
        if len(text) <= budget:
            break
        text = llm.call(sys_, prompt + f"\n\nYour previous attempt was {len(text)} characters, over the limit. "
                        f"Write it again in at most {budget - 1000 * (i + 1)} characters.", tag="rh_pin_retry")
    return text[:budget]


def mem_path(ci, arms):
    """restudy and quiz share mem_{ci}.json (first run); any other arm set or K/EVERY gets its own file."""
    if arms == ARMS and not tag():
        return os.path.join(OUT, f"mem_{ci}.json")
    return os.path.join(OUT, f"mem_{ci}.{'+'.join(arms)}{tag()}.json")


def build_conv(ci, data, write=True, arms=None):
    arms = arms or ARGS.arms
    path = mem_path(ci, arms)
    if write and os.path.exists(path):
        return
    ss = L.sessions(data[ci])
    cps = D.checkpoints(len(ss))

    def records(s):              # same prompt as drift.py, so these come from the cache
        body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(L.show_turn(t, s["date"]) for t in s["turns"])
        return L.parse_records(llm.call(L.REC_SYS, body, tag="drift_rec"), {t["id"] for t in s["turns"]})
    with ThreadPoolExecutor(2) as ex:
        recs = list(ex.map(records, ss))

    def run(arm):
        mem, out, extra, log = "", {}, [], []
        bank, box, due, check = [], {}, {}, 0          # leitner state: bank[j] = (record, question)
        pinned = []                                     # pin arm: restored facts, never rewritten
        cost = {"rewrite": 0, "test": 0, "rebuild": 0}  # characters sent to the model (input), per kind of call
        rebuild_next, rebuilds = False, []              # trigger arm
        for i, s in enumerate(ss, 1):
            if arm == "leitner":
                bank += [(r, q) for r, q in zip(recs[i - 1], probes_for(recs[i - 1])) if q]
            if (arm == "prebuild" and i % ARGS.every == 0) or (arm == "trigger" and rebuild_next):
                ledger = [r["text"] for rs in recs[:i] for r in rs]
                body = ("Memory records extracted so far from the conversation (in time order; each is self-contained):\n\n"
                        + "\n".join("- " + t for t in ledger)
                        + f"\n\nWrite the memory of everything so far (at most {L.BUDGET} characters).")
                cost["rebuild"] += len(body)
                mem = L.within_budget(body, "drift_rebuild")
            elif arm == "pin":
                cost["rewrite"] += len(mem) + len(L.show_session(s))
                mem = rewrite_to(mem, s, L.BUDGET - PIN)
                for r in extra:
                    if r["text"] not in pinned:
                        pinned.append(r["text"])
                while sum(len(t) + 3 for t in pinned) > PIN:
                    pinned.pop(0)
            else:
                cost["rewrite"] += len(mem) + len(L.show_session(s)) + sum(len(r["text"]) for r in extra)
                mem = rewrite(mem, s, extra)
            extra = []
            if arm == "trigger":
                if rebuild_next:
                    rebuilds.append(i)
                rebuild_next = False
            if i % ARGS.every == 0 and i < len(ss) and arm != "prebuild":
                ledger = [r for rs in recs[:i] for r in rs]
                sample = random.Random(f"{ci}-{i}").sample(ledger, min(ARGS.k, len(ledger)))
                if arm == "restudy":
                    extra = sample
                elif arm == "selfquiz":
                    extra = selfquiz(mem, ARGS.k)
                elif arm == "leitner":
                    check += 1
                    ready = [j for j in range(len(bank)) if due.get(j, 0) <= check]
                    ready.sort(key=lambda j: (box.get(j, 0), j))   # lowest box first, then oldest
                    pick = ready[:ARGS.k]
                    bad = set(ask(mem, [bank[j][0] for j in pick], [bank[j][1] for j in pick]))
                    for n, j in enumerate(pick):
                        box[j] = 0 if n in bad else box.get(j, 0) + 1
                        due[j] = check + (1 if n in bad else 2 ** box[j])
                    extra = [bank[pick[n]][0] for n in sorted(bad)]
                elif arm == "trigger":
                    miss = len(quiz(mem, sample)) / max(1, len(sample))
                    rebuild_next = miss > ARGS.trigger
                    extra = []
                elif arm == "pin":
                    full = mem + "\n\nPinned facts:\n" + "\n".join("- " + t for t in pinned)
                    extra = [sample[j] for j in quiz(full, sample)]
                else:
                    extra = [sample[j] for j in quiz(mem, sample)]
                n_test = len(bank) if arm == "leitner" else ARGS.k
                cost["test"] += 2 * len(mem) + 300 * min(ARGS.k, n_test)   # answer call sees the memory; grading is small
                log.append({"after": i, "shown": len(extra), "facts": [r["text"] for r in extra]})
            if i in cps or EVERY_SESSION:
                out[str(i)] = (mem + "\n\nPinned facts:\n" + "\n".join("- " + t for t in pinned)) if arm == "pin" else mem
        return {"mem": out, "log": log, "cost": cost, "rebuilds": rebuilds}
    with ThreadPoolExecutor(2) as ex:
        res = dict(zip(arms, ex.map(run, arms)))
    if not write:
        return res
    json.dump({"checkpoints": cps, **res}, open(path, "w"), ensure_ascii=False, indent=1)
    print("built", ci, {a: sum(x["shown"] for x in res[a]["log"]) for a in arms}, "facts shown", flush=True)


def durability(data):
    """H19: are facts that the quiz arm restored still in the final memory? Replays the quiz arm from the cache
    (no rewrite calls) to recover which facts were restored and when, then asks them of the final quiz memory and
    of the final roll memory (drift.py) for comparison."""
    real = llm.urllib.request.urlopen
    out = []
    for ci in ARGS.convs:
        def offline(*a, **k):
            raise RuntimeError("cache miss while replaying the quiz arm")
        llm.urllib.request.urlopen = offline
        try:
            res = build_conv(ci, data, write=False, arms=["quiz"])["quiz"]
        finally:
            llm.urllib.request.urlopen = real
        last = max(res["mem"], key=int)
        roll = json.load(open(os.path.join(D.OUT, f"mem_{ci}.json")))["roll"][last]
        seen, facts = set(), []
        for e in res["log"]:
            for t in e["facts"]:
                if t not in seen:
                    seen.add(t); facts.append((e["after"], {"text": t}))
        recs = [r for _, r in facts]
        qs = probes_for(recs)
        for name, mem in (("quiz_final", res["mem"][last]), ("roll_final", roll)):
            bad = set()
            for b in range(0, len(recs), 20):
                bad |= {b + j for j in ask(mem, recs[b:b + 20], qs[b:b + 20])}
            for n, (after, r) in enumerate(facts):
                out.append({"conv": ci, "mem": name, "restored_after": after, "last": int(last), "kept": int(n not in bad)})
        print("durability", ci, len(facts), "restored facts", flush=True)
    json.dump(out, open(os.path.join(OUT, "durability.json"), "w"), indent=1)
    by = collections.defaultdict(list)
    for r in out:
        by[(r["mem"], "all")].append(r["kept"])
        gap = r["last"] - r["restored_after"]
        by[(r["mem"], "restored <=6 sessions before end" if gap <= 6 else "restored >6 sessions before end")].append(r["kept"])
    for k in sorted(by):
        print(f"  {k[0]:11s} {k[1]:34s} kept {100 * sum(by[k]) / len(by[k]):5.1f}%  n={len(by[k])}")


def lprobe(data):
    """Main metric with more power than the 12 LoCoMo questions: EVAL_FACTS ledger records per conversation
    (sampled with a seed unrelated to maintenance), one question each, asked of every arm's memory at every
    checkpoint. Arms: roll/rebuild from drift.py plus every rehearse memory file present."""
    import glob
    path = os.path.join(OUT, "lprobe.jsonl")
    done = {(r["conv"], r["arm"], r["k"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    for ci in ARGS.convs:
        ss = L.sessions(data[ci])

        def records(s):
            body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(L.show_turn(t, s["date"]) for t in s["turns"])
            return L.parse_records(llm.call(L.REC_SYS, body, tag="drift_rec"), {t["id"] for t in s["turns"]})
        recs = [r for s in ss for r in records(s)]
        facts = random.Random(f"eval-{ci}").sample(recs, min(EVAL_FACTS, len(recs)))
        qs = probes_for(facts)
        mems = {}
        dm = json.load(open(os.path.join(D.OUT, f"mem_{ci}.json")))
        mems.update({("roll", k): v for k, v in dm["roll"].items()})
        mems.update({("rebuild", k): v for k, v in dm["rebuild"].items()})
        for f in glob.glob(os.path.join(OUT, f"mem_{ci}.*json")) + [os.path.join(OUT, f"mem_{ci}.json")]:
            if not os.path.exists(f):
                continue
            m = json.load(open(f))
            suffix = re.search(r"(_k\d+e\d+)\.json$", f)
            for arm in ALL_ARMS:
                if arm in m:
                    mems.update({(arm + (suffix.group(1) if suffix else ""), k): v for k, v in m[arm]["mem"].items()})
        jobs = [(a, k, v) for (a, k), v in mems.items() if (ci, a, int(k)) not in done]

        def one(job):
            a, k, mem = job
            bad = set()
            for b in range(0, len(facts), 20):
                bad |= {b + j for j in ask(mem, facts[b:b + 20], qs[b:b + 20])}
            return {"conv": ci, "arm": a, "k": int(k), "kept": [int(j not in bad) for j in range(len(facts))]}
        with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as fo:
            for r in ex.map(one, jobs):
                fo.write(json.dumps(r) + "\n"); fo.flush()
        print("lprobe", ci, len(jobs), "memories", flush=True)
    rows = [json.loads(l) for l in open(path)]
    cps = {ci: json.load(open(os.path.join(D.OUT, f"mem_{ci}.json")))["checkpoints"] for ci in ARGS.convs}
    by, fin = collections.defaultdict(list), collections.defaultdict(dict)
    for r in rows:
        if r["conv"] not in cps or r["k"] not in cps[r["conv"]]:
            continue
        pos = cps[r["conv"]].index(r["k"])
        by[(r["arm"], pos)] += r["kept"]
        if pos == 2:
            fin[r["arm"]][r["conv"]] = r["kept"]
    print(f"ledger-probe retention ({EVAL_FACTS} facts per conversation) after 1/3, 2/3, all sessions")
    arms = sorted({a for a, _ in by}, key=lambda a: sum(by[(a, 2)]) / max(1, len(by[(a, 2)])))
    for a in arms:
        print(f"  {a:16s}" + "".join(f"{100 * sum(by[(a, p)]) / len(by[(a, p)]):7.1f}" if by[(a, p)] else "      -"
                                    for p in range(3)) + f"   n={len(by[(a, 2)])}")

    def sign(x, y):
        n, k = x + y, min(x, y)
        return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    for a in arms:
        if a == "roll" or "roll" not in fin:
            continue
        x = y = 0
        for ci, v in fin[a].items():
            if ci in fin["roll"]:
                x += sum(p > q for p, q in zip(v, fin["roll"][ci])); y += sum(q > p for p, q in zip(v, fin["roll"][ci]))
        print(f"  final {a} vs roll: {a}-only {x}, roll-only {y}, p={sign(x, y):.4f}")


def tavg(data):
    """Time-averaged retention: memory quality at any moment a user might ask, not only right after a rebuild.
    Every arm's memory is replayed from the cache after each session (no new rewrite/rebuild calls), and the
    EVAL_FACTS ledger facts (same sample as lprobe) that were already written are asked every STEP sessions."""
    global EVERY_SESSION
    from . import curve as C
    STEP = 2
    path = os.path.join(OUT, "tavg.jsonl")
    done = {(r["conv"], r["arm"], r["after"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    real = llm.urllib.request.urlopen
    for ci in ARGS.convs:
        ss = L.sessions(data[ci])

        def records(s):
            body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(L.show_turn(t, s["date"]) for t in s["turns"])
            return L.parse_records(llm.call(L.REC_SYS, body, tag="drift_rec"), {t["id"] for t in s["turns"]})
        pairs = [(si, r) for si, s in enumerate(ss, 1) for r in records(s)]
        picked = random.Random(f"eval-{ci}").sample(pairs, min(EVAL_FACTS, len(pairs)))
        facts, born = [r for _, r in picked], [si for si, _ in picked]
        qs = probes_for(facts)

        misses = [0]

        def offline(*a, **k):      # a rare miss (e.g. an over-budget retry) is generated live and counted
            misses[0] += "/v1/responses" in getattr(a[0], "full_url", "")   # not the Pro usage guard
            return real(*a, **k)
        llm.urllib.request.urlopen = offline
        EVERY_SESSION = True
        try:
            mems = {"roll": {str(i): m for i, m in enumerate(C.replay(ss), 1)}}
            for arms, every in ARGS.tavg_arms:
                ARGS.every = every
                for a, v in build_conv(ci, data, write=False, arms=[arms]).items():
                    mems[a + ("" if every == 3 else f"_e{every}")] = v["mem"]
        finally:
            llm.urllib.request.urlopen = real
            EVERY_SESSION = False
            ARGS.every = EVERY
        jobs = [(a, i) for a in mems for i in range(STEP, len(ss) + 1, STEP) if (ci, a, i) not in done]

        def one(job):
            a, i = job
            idx = [j for j in range(len(facts)) if born[j] <= i]
            bad = set()
            for b in range(0, len(idx), 20):
                part = idx[b:b + 20]
                bad |= {part[j] for j in ask(mems[a][str(i)], [facts[x] for x in part], [qs[x] for x in part])}
            return {"conv": ci, "arm": a, "after": i, "facts": idx, "kept": [int(j not in bad) for j in idx]}
        with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as fo:
            for r in ex.map(one, jobs):
                fo.write(json.dumps(r) + "\n"); fo.flush()
        print("tavg", ci, len(jobs), "probes", misses[0], "live calls in replay", flush=True)
    rows = [json.loads(l) for l in open(path)]
    by, per = collections.defaultdict(list), collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["conv"] in ARGS.convs:
            by[r["arm"]] += r["kept"]
            for f, k in zip(r["facts"], r["kept"]):
                per[r["arm"]][(r["conv"], r["after"], f)] = k
    print("time-averaged retention of already-written ledger facts (every 2 sessions)")
    for a in sorted(by, key=lambda a: sum(by[a]) / len(by[a])):
        x = y = 0
        for key, k in per[a].items():
            if key in per["roll"]:
                x += k > per["roll"][key]; y += per["roll"][key] > k
        print(f"  {a:14s} {100 * sum(by[a]) / len(by[a]):6.1f}  n={len(by[a])}  vs roll {x}:{y}")


def build(data):
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(lambda ci: build_conv(ci, data), ARGS.convs))


def answer(data):
    path = os.path.join(OUT, "judged.jsonl")
    done = {(r["conv"], r["qid"], r["arm"], r["k"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = []
    for ci in ARGS.convs:
        m = json.load(open(mem_path(ci, ARGS.arms)))
        for q in D.early_questions(data[ci], m["checkpoints"][-1], ARGS.per_conv):
            for arm in ARGS.arms:
                for k in m["checkpoints"]:
                    if (ci, q["qid"], arm + tag(), k) not in done:
                        jobs.append((ci, q, arm + tag(), k, m[arm]["mem"][str(k)]))

    def one(job):                # same answer and judge prompts as drift.py
        ci, q, arm, k, mem = job
        a = llm.call(L.ANS_SYS, f"Memory:\n{mem}\n\nQuestion: {q['question']}", tag="drift_ans").strip()
        v = llm.call(L.JUDGE_SYS, f"Question: {q['question']}\nGold answer: {q['answer']}\nGenerated answer: {a}",
                     effort="low", max_tokens=4000, tag="drift_judge")
        return {"conv": ci, "qid": q["qid"], "cat": q["category"], "arm": arm, "k": k, "q": q["question"],
                "gold": str(q["answer"]), "pred": a, "mem_chars": len(mem),
                "correct": int("CORRECT" in v.upper() and "WRONG" not in v.upper())}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


def report(data):
    rows = [json.loads(l) for l in open(os.path.join(D.OUT, "judged.jsonl"))]
    rows += [json.loads(l) for l in open(os.path.join(OUT, "judged.jsonl"))]
    rows = [r for r in rows if r["conv"] in ARGS.convs]
    cps = {ci: json.load(open(os.path.join(D.OUT, f"mem_{ci}.json")))["checkpoints"] for ci in ARGS.convs}
    by, cell = collections.defaultdict(list), {}
    for r in rows:
        pos = cps[r["conv"]].index(r["k"])
        by[(r["arm"], pos)].append(r["correct"]); cell[(r["conv"], r["qid"], pos, r["arm"])] = r["correct"]
    print("early-fact retention (% correct) after 1/3, 2/3, all sessions")
    order = ["roll", "selfquiz", "restudy", "quiz", "leitner", "pin", "rebuild"]
    for arm in order + sorted({a for (_, _, _, a) in cell} - set(order)):
        print(f"{arm:8s}" + "".join(f"{100 * sum(by[(arm, p)]) / len(by[(arm, p)]):8.1f} n={len(by[(arm, p)]):<3d}"
                                    for p in range(3) if by[(arm, p)]))

    def sign(a, b):
        n, k = a + b, min(a, b)
        return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    pairs = [("quiz", "roll"), ("restudy", "roll"), ("quiz", "restudy"), ("rebuild", "quiz"),
             ("selfquiz", "roll"), ("quiz", "selfquiz"), ("leitner", "quiz"), ("leitner", "roll"),
             ("pin", "quiz"), ("pin", "roll"), ("rebuild", "pin")]
    pairs += [(a, "roll") for a in sorted({a for (_, _, _, a) in cell} - set(order))]
    for x, y in pairs:
        keys = [(c, q, p) for (c, q, p, a) in cell if a == x and (c, q, p, y) in cell and p == 2]
        a = sum(cell[k + (x,)] > cell[k + (y,)] for k in keys); b = sum(cell[k + (y,)] > cell[k + (x,)] for k in keys)
        print(f"final checkpoint {x} vs {y}: {x}-only {a}, {y}-only {b}, sign test p={sign(a, b):.3f} (n={len(keys)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["build", "answer", "report", "durability", "lprobe", "tavg"])
    p.add_argument("--convs", default="0,2,4,5,7,8")
    p.add_argument("--per-conv", type=int, default=12)
    p.add_argument("--par", type=int, default=3)
    p.add_argument("--arms", default=",".join(ARMS), help="restudy,quiz (default), selfquiz or leitner")
    p.add_argument("--k", type=int, default=K, help="probes per check")
    p.add_argument("--every", type=int, default=EVERY, help="sessions between checks")
    p.add_argument("--budget", type=int, default=L.BUDGET, help="memory budget in characters (default 8000)")
    p.add_argument("--trigger", type=float, default=TRIGGER, help="trigger arm: miss rate that triggers a rebuild")
    p.add_argument("--tavg-arms", default="quiz:3,leitner:3,trigger:3,prebuild:3,prebuild:6,prebuild:12",
                   help="tavg: arm:every pairs to replay")
    ARGS = p.parse_args()
    ARGS.arms = ARGS.arms.split(",")
    ARGS.tavg_arms = [(x.split(":")[0], int(x.split(":")[1])) for x in ARGS.tavg_arms.split(",")]
    if ARGS.budget != L.BUDGET:      # same prompts with the new limit; own directories here and for drift's roll/rebuild
        L.SUM_SYS = L.SUM_SYS.replace(str(L.BUDGET), str(ARGS.budget))
        L.BUDGET = ARGS.budget
        OUT = OUT + f"_b{ARGS.budget}"
        D.OUT = D.OUT + f"_b{ARGS.budget}"
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(L.DATA))
    {"build": build, "answer": answer, "report": report, "durability": durability, "lprobe": lprobe, "tavg": tavg}[ARGS.step](data)
