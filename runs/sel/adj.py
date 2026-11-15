"""Evidence adjudication (11-02, stage-1 failure study, fix 2): the EverMem and LoCoMo gold marks are coarse (EverMem marks
message ranges that include greetings; some LoCoMo marks point at the question line, not the answer line). Two independent
judges (gpt-oss-120b, gemma-4-31b-it) decide for each (question, round) whether the round contains information needed to
answer the question. A round is "evidence" only if BOTH say YES. Candidates per question: every gold round plus the top-20
rounds (packing order at the answer budget) of E0, A0, RR8 and MR4 -- the same set for every model.
Outputs (out/ADJ_<b>.jsonl, one row per question): core = gold rounds judged evidence, filler = gold rounds judged not,
alt = non-gold rounds judged evidence. A secondary metric only; the frozen gold and metrics stay as they are.
Controls (controls): POS = LongMemEval gold rounds (marked per message); NEG_far = a round of a different conversation;
NEG_near = a non-gold round of the SAME conversation from a session without any gold (same domain, harder). PASS needs, for
each judge, POS >= 0.90, NEG_far <= 0.05, NEG_near <= 0.10, and no missing verdicts; a formal run refuses to start without a
PASS record (out/ADJ_controls_PASS.json for this VERSION). Any missing verdict makes the formal run exit 1 without output.
Gold rounds are split into core (both YES), filler (both NO) and split (judges disagree); alt = non-gold, both YES.
usage: python3 adj.py ev|loc|controls --oss PORTS --gem PORTS [--workers 96]"""
import argparse, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
from k1 import rounds_of
from a6judge import chat, JudgeFail
PROMPT = """You check evidence for a question about past conversations.
Question (asked on {qdate}): {q}
Reference answer: {a}

Conversation round ({date}):
{r}

Does this round contain information that is needed to answer the question -- it states the answer or part of it, or a fact the answer depends on? A greeting, an acknowledgement, or a message that is only on the same topic does not count.
Answer YES or NO only."""
VERSION = "adj-v1-2026-11-02"; MAXC = 3000
CFG = {"ev": ("q_ev150.json", "ev150", 4000), "loc": ("q_loc150.json", "loc150", 2000)}
MODELS = ["E0", "A0", "RR8", "MR4"]


def verdict(model, ports, item):
    q, date, text = item; port = random.choice(ports)
    t = chat(model, port, None, PROMPT.format(qdate=q["question_date"], q=q["question"][:3000], a=str(q["answer"])[:600], date=date, r=text[:MAXC]), 1500)
    u = t.strip().upper(); y, n = bool(re.search(r"\bYES\b", u)), bool(re.search(r"\bNO\b", u))
    if y == n: raise JudgeFail(f"no clear verdict: {t[-120:]!r}")
    return y


def run(jobs, a, out):
    """jobs: list of (key, q, date, text); writes {key: {"oss": bool, "gem": bool}} rows to out (resumable)"""
    done = {}
    try:
        for l in open(out):
            r = json.loads(l)
            if r.get("version") == VERSION: done[r["key"]] = r
    except FileNotFoundError: pass
    todo = [j for j in jobs if j[0] not in done]; print("todo", len(todo), "of", len(jobs), flush=True)
    f = open(out, "a"); ff = open(out + ".fail", "a"); lk = threading.Lock(); fails = [0]
    def one(j):
        k, q, date, text = j
        try: r = {"key": k, "version": VERSION, "oss": verdict("gpt-oss-120b", a.oss.split(","), (q, date, text)), "gem": verdict("gemma-4-31b-it", a.gem.split(","), (q, date, text))}
        except Exception as e:
            with lk: fails[0] += 1; ff.write(json.dumps({"key": k, "version": VERSION, "error": repr(e)[:500]}) + "\n"); ff.flush()
            return
        with lk: f.write(json.dumps(r) + "\n"); f.flush(); done[k] = r
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, todo))
    print("failures", fails[0], flush=True); return done


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--oss", required=True); ap.add_argument("--gem", required=True)
    ap.add_argument("--workers", type=int, default=96); a = ap.parse_args()
    if a.what == "controls":
        L = json.load(open("q_lmes150.json")); rng = random.Random(0); jobs = []; lab = {}
        for n, q in enumerate(rng.sample(L, 40)):
            R = rounds_of(q); g = [i for i, r in enumerate(R) if r[3]]
            if not g: continue
            i = rng.choice(g); jobs.append((f"POS|{q['question_id']}|{i}", q, R[i][1], R[i][2])); lab[jobs[-1][0]] = "POS"
            o = L[(L.index(q) + 7) % len(L)]; RO = rounds_of(o); j = rng.randrange(len(RO))   # a round of another conversation
            jobs.append((f"NEGFAR|{q['question_id']}|{o['question_id']}|{j}", q, RO[j][1], RO[j][2])); lab[jobs[-1][0]] = "NEG_far"
            gs = {R[x][0] for x in g}; near = [x for x, r in enumerate(R) if r[0] not in gs]
            if near:
                j = rng.choice(near); jobs.append((f"NEGNEAR|{q['question_id']}|{j}", q, R[j][1], R[j][2])); lab[jobs[-1][0]] = "NEG_near"
        D = run(jobs, a, "out/ADJ_controls.jsonl"); ok = len(D.keys() & lab.keys()) == len(lab); res = {}
        lim = {"POS": (.90, None), "NEG_far": (None, .05), "NEG_near": (None, .10)}
        for jn in ("oss", "gem"):
            for c, (lo, hi) in lim.items():
                v = [D[k][jn] for k in lab if lab[k] == c and k in D]; r = sum(v) / max(1, len(v))
                good = len(v) >= 25 and (lo is None or r >= lo) and (hi is None or r <= hi)   # every control class needs >= 25 verdicts
                ok &= good; res[f"{jn}_{c}"] = [round(r, 3), len(v)]; print(f"controls {jn} {c}: YES rate {r:.2f} (n={len(v)})  {'ok' if good else 'FAIL'}")
        print("missing verdicts", len(lab) - len(D.keys() & lab.keys())); print("ADJ CONTROLS", "PASS" if ok else "FAIL")
        if ok: json.dump({"version": VERSION, **res}, open("out/ADJ_controls_PASS.json", "w"))
        sys.exit(0 if ok else 1)
    try: assert json.load(open("out/ADJ_controls_PASS.json"))["version"] == VERSION
    except Exception: sys.exit("FAIL no controls PASS record for this VERSION: run `adj.py controls` first")
    qf, tag, B = CFG[a.what]; Q = {q["question_id"]: q for q in json.load(open(qf))}
    C = {m: {r["qid"]: r["rank"] for r in map(json.loads, open(f"ctx_P1{m}_{tag}.jsonl")) if r["budget"] == B} for m in MODELS}
    jobs = []; cand = {}
    for qid, q in Q.items():
        R = rounds_of(q); g = [i for i, r in enumerate(R) if r[3]]
        if not g: continue
        s = set(g)
        for m in MODELS: s |= set(C[m][qid][:20])
        cand[qid] = sorted(s); jobs += [(f"{qid}|{i}", q, R[i][1], R[i][2]) for i in cand[qid]]
    D = run(jobs, a, f"out/ADJV_{a.what}.jsonl")
    miss = [j[0] for j in jobs if j[0] not in D]
    if miss: sys.exit(f"FAIL {len(miss)} candidate rounds without a verdict (see out/ADJV_{a.what}.jsonl.fail); rerun to resume")
    with open(f"out/ADJ_{a.what}.jsonl.tmp", "w") as f:
        for qid in cand:
            R = rounds_of(Q[qid]); V = {i: D[f"{qid}|{i}"] for i in cand[qid]}
            both = {i for i, v in V.items() if v["oss"] and v["gem"]}; none_ = {i for i, v in V.items() if not v["oss"] and not v["gem"]}
            gold = {i for i, r in enumerate(R) if r[3]}
            f.write(json.dumps({"qid": qid, "version": VERSION, "core": sorted(both & gold), "filler": sorted(none_ & gold), "split": sorted(gold - both - none_),
                                "alt": sorted(both - gold), "n_cand": len(V), "agree": sum(v["oss"] == v["gem"] for v in V.values())}) + "\n")
    os.replace(f"out/ADJ_{a.what}.jsonl.tmp", f"out/ADJ_{a.what}.jsonl"); print("ADJDONE", a.what)


if __name__ == "__main__": main()
