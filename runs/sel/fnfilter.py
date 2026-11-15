"""False-negative filter and negative candidates for D4 (D4_SPEC v2.4 §5, §6, §12).
  cands   : per training question, the candidate negatives of every source (8b, ret, stu, struct, rand), each a fixed list,
            never gold or support. Written once and frozen (the D4 groups are drawn only from these lists).
  controls: control pairs from the training worlds -- POS = one gold round of a question, HARD = a structural near-miss of
            the same question (sibling fact, same entity other attribute, same speaker other entity, date-window outside),
            FAR = a round of another world. Prints them for a manual label check before judging (--show).
  judge   : both judges (Qwen3.8-27B, gpt-oss-120b) on pairs (--controls / --sample N / all candidates). The judge must
            quote the round first; a YES whose quote is not in the round text counts as INVALID.
  verdict : per question, rounds both judges call YES -> support; split or invalid -> ignore (never a negative).
The text the judges see is exactly what the reranker sees: fmt.doc_str (previous user message context, 3000-char cut).
usage: python3 fnfilter.py cands --qfile Q.json --pooldir P --worlds ws4,ws6,ws6e --ids IDS [--student ctx_E0TR.jsonl] --out C.json
       python3 fnfilter.py controls --qfile Q.json --worlds ... --ids IDS --out CTL.json [--show]
       python3 fnfilter.py judge --pairs C.json|CTL.json --qfile Q.json --q27 PORTS --oss PORTS [--sample 500] --out V.jsonl
       python3 fnfilter.py verdict --pairs C.json --verdicts V.jsonl --out FN.json"""
import argparse, glob, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of
from fmt import doc_str, split_round, round_prev
VERSION = "fnfilter-v1.1-2026-11-03"; TOPK = 60; NRAND = 60
PROMPT = """You check whether one conversation round helps answer a question about past conversations.
Question (asked on {qdate}): {q}
Reference answer: {a}

The round, exactly as shown to the retrieval model:
{doc}

Does this round provide ANY part of the information needed to answer the question? For a question that asks for several items, providing one of them counts. Being on the same topic, greeting, acknowledging, or saying something similar but different does not count. Do not use general knowledge: judge only from the text of the round.
First write QUOTE: followed by the exact words from the round that provide the information (or QUOTE: NONE). Then on the last line write YES or NO."""


def lit(x):
    if isinstance(x, str):
        import ast
        try: return ast.literal_eval(x)
        except Exception: return x
    return x


def worlds(dirs):
    W = {}
    for d in dirs.split(","):
        for f in glob.glob(os.path.join(d, "w0*.json")):
            w = json.load(open(f)); W[w["world"]] = w
    return W


def qinfo(q, W, pool):
    """per round: fact id, world round index, fact, speaker; plus gold and support round sets"""
    wid = int(q["question_id"].split("_")[0][1:]); w = W[wid]; F = {f["id"]: f for f in w["facts"]}
    st = pool["stage"]; G = set(lit(q["gold_stages"])); S = set(lit(q.get("support_stages") or [])) - G
    fact = [F.get(x.split("#")[0]) if isinstance(x, str) else None for x in st]
    wk = [int(x.split("#")[1]) if isinstance(x, str) else None for x in st]
    spk = [w["rounds"][k].get("speaker") if k is not None and k < len(w["rounds"]) else None for k in wk]
    gold = {i for i, x in enumerate(st) if x in G}; sup = {i for i, x in enumerate(st) if x in S}
    return w, fact, spk, gold, sup, st


def structural(q, fact, spk, gold, sup, st, P):
    """near-miss rounds from the ledger (no model): sibling facts, same entity other attribute, same speaker other entity,
    date-window outside (dated questions only); all inside the candidate pool, never gold or support"""
    gf = [fact[i] for i in gold if fact[i]]; out = {}
    gent = {f["entity"] for f in gf}; gattr = {(f["entity"], f["attr"]) for f in gf}; gid = {f["id"] for f in gf}
    sibs = {f.get("sib_of") for f in gf if f.get("sib_of")} | {f["id"] for f in gf}; gsib = {g["sib_of"] for g in gf if g.get("sib_of")}
    gspk = {spk[i] for i in gold if spk[i]}
    gcat = {f["cat"] for f in gf}
    for i in P:
        f = fact[i]
        if i in gold or i in sup or f is None or f["id"] in gid: continue
        if (f.get("sib_of") and f["sib_of"] in sibs) or f["id"] in gsib: out[i] = "sibling"
        elif f["entity"] in gent and (f["entity"], f["attr"]) not in gattr: out[i] = "entity_other_attr"
        elif spk[i] and spk[i] in gspk and f["entity"] not in gent: out[i] = "speaker_other_entity"
        elif q.get("scope") == "date" and f["cat"] in gcat: out[i] = "date_outside"
    return out


def cands(a):
    Q = {q["question_id"]: q for q in json.load(open(a.qfile))}; ids = open(a.ids).read().split() if a.ids else list(Q)
    W = worlds(a.worlds); stu = {}
    if a.student:
        for r in map(json.loads, open(a.student)):
            if r["budget"] == max(int(x) for x in os.environ.get("K1_BUDGETS", "16000").split(",")) or r["qid"] not in stu: stu[r["qid"]] = r["rank"]
    out = {"version": VERSION, "qfile": a.qfile, "pooldir": a.pooldir, "q": {}}
    for qid in ids:
        q = Q[qid]; pool = json.load(open(os.path.join(a.pooldir, qid + ".json"))); P = pool["fused"][:600]
        w, fact, spk, gold, sup, st = qinfo(q, W, pool); bad = gold | sup; ng = [i for i in P if i not in bad]
        rr = {int(k): v for k, v in pool["rr8"].items()}
        e = {"8b": sorted([i for i in ng if i in rr], key=lambda i: -rr[i])[:TOPK], "ret": ng[:TOPK],
             "stu": [i for i in stu.get(qid, []) if i in set(ng)][:TOPK] if a.student else [],
             "struct": structural(q, fact, spk, gold, sup, st, P), "rand": random.Random(f"{VERSION}|{qid}").sample(ng, min(NRAND, len(ng)))}
        out["q"][qid] = e
    json.dump(out, open(a.out, "w")); n = sum(len(set(e["8b"]) | set(e["ret"]) | set(e["stu"]) | set(map(int, e["struct"])) | set(e["rand"])) for e in out["q"].values())
    print("questions", len(out["q"]), "unique (question, round) pairs", n)


def controls(a):
    Q = json.load(open(a.qfile)); ids = set(open(a.ids).read().split()) if a.ids else None; W = worlds(a.worlds)
    rng = random.Random(f"{VERSION}|controls"); Q = [q for q in Q if ids is None or q["question_id"] in ids]; rng.shuffle(Q)
    pairs = {"POS": [], "HARD": [], "FAR": []}
    for q in Q:
        pool = json.load(open(os.path.join(a.pooldir, q["question_id"] + ".json"))); P = pool["fused"][:600]
        w, fact, spk, gold, sup, st = qinfo(q, W, pool)
        if len(pairs["POS"]) < 40 and gold: pairs["POS"].append((q["question_id"], rng.choice(sorted(gold))))
        s = structural(q, fact, spk, gold, sup, st, P)
        if len(pairs["HARD"]) < 40 and s: i = rng.choice(sorted(s)); pairs["HARD"].append((q["question_id"], i, s[i]))
        if len(pairs["FAR"]) < 40:
            o = rng.choice([x for x in Q if x["question_id"].split("_")[0] != q["question_id"].split("_")[0]]); RO = rounds_of(o)
            pairs["FAR"].append((q["question_id"], o["question_id"], rng.randrange(len(RO))))
        if all(len(v) >= 40 for v in pairs.values()): break
    json.dump({"version": VERSION, "pairs": pairs}, open(a.out, "w")); print({k: len(v) for k, v in pairs.items()})
    if a.show:
        QQ = {q["question_id"]: q for q in Q}
        for c in ("POS", "HARD"):
            for p in pairs[c][:a.show]:
                q = QQ[p[0]]; R = rounds_of(q); print(f"\n[{c}] {p[0]} {p[2] if c == 'HARD' else ''}\nQ: {q['question'][:200]}\nA: {str(q['answer'])[:200]}\nROUND: {R[p[1]][2][:400]}")


def doc_of(q, i, R=None, prev=None):
    R = R or rounds_of(q); prev = prev or round_prev(q); return doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i])


def verdict_one(model, ports, q, doc):
    from a6judge import chat
    doc = doc.split("<|im_end|>")[0].rstrip()   # v1.1: drop the reranker chat-template suffix; the round text is unchanged
    t = chat(model, random.choice(ports), None, PROMPT.format(qdate=q["question_date"], q=q["question"][:3000], a=str(q["answer"])[:1500], doc=doc[:3600]), 1500)
    lines = [x.strip() for x in t.strip().splitlines() if x.strip()]; last = lines[-1].upper() if lines else ""
    v = "YES" if re.search(r"\bYES\b", last) else "NO" if re.search(r"\bNO\b", last) else "INVALID"
    m = re.search(r"QUOTE:\s*(.+)", t)
    if v == "YES":
        quote = (m.group(1) if m else "").strip().strip('"“”').strip()
        norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
        if not quote or quote.upper() == "NONE" or norm(quote)[:60] not in norm(doc): v = "INVALID"
    return v


def judge(a):
    Q = {q["question_id"]: q for q in json.load(open(a.qfile))}; D = json.load(open(a.pairs)); jobs = []
    if "pairs" in D:   # controls
        for c, L in D["pairs"].items():
            for p in L:
                q = Q[p[0]]
                if c == "FAR": o = Q[p[1]]; jobs.append((f"{c}|{p[0]}|{p[1]}|{p[2]}", q, doc_of(o, p[2])))
                else: jobs.append((f"{c}|{p[0]}|{p[1]}", q, doc_of(q, p[1])))
    else:
        for qid, e in D["q"].items():
            q = Q[qid]; R = rounds_of(q); prev = round_prev(q)
            for i in sorted(set(e["8b"]) | set(e["ret"]) | set(e["stu"]) | set(map(int, e["struct"])) | set(e["rand"])): jobs.append((f"{qid}|{i}", q, (R, prev, i)))
        if a.sample: jobs = random.Random(f"{VERSION}|sample").sample(jobs, min(a.sample, len(jobs)))
    done = set()
    if os.path.exists(a.out): done = {json.loads(l)["key"] for l in open(a.out) if json.loads(l).get("version") == VERSION}
    jobs = [j for j in jobs if j[0] not in done]; print("todo", len(jobs), flush=True)
    f = open(a.out, "a"); lk = threading.Lock(); fails = [0]
    def one(j):
        k, q, d = j
        if isinstance(d, tuple): R, prev, i = d; d = doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i])
        try: r = {"key": k, "version": VERSION, "q27": verdict_one("Qwen3.8-27B-FP8", a.q27.split(","), q, d), "oss": verdict_one("gpt-oss-120b", a.oss.split(","), q, d)}
        except Exception as e:
            with lk: fails[0] += 1
            r = {"key": k, "version": VERSION, "error": repr(e)[:300]}
        with lk: f.write(json.dumps(r) + "\n"); f.flush()
    import time; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, jobs))
    print(f"JUDGEDONE pairs {len(jobs)} failures {fails[0]} seconds {time.time() - t0:.0f} pairs/s {len(jobs) / max(1, time.time() - t0):.2f}", flush=True)
    if fails[0]: sys.exit(3)


def verdict(a):
    V = {}
    for l in open(a.verdicts):
        r = json.loads(l)
        if r.get("version") == VERSION and "error" not in r: V[r["key"]] = r
    D = json.load(open(a.pairs)); out = {}; miss = 0; c = {"support": 0, "ignore": 0, "neg": 0}
    for qid, e in D["q"].items():
        sup, ign = [], []
        for i in sorted(set(e["8b"]) | set(e["ret"]) | set(e["stu"]) | set(map(int, e["struct"])) | set(e["rand"])):
            r = V.get(f"{qid}|{i}")
            if r is None: miss += 1; continue
            if r["q27"] == "YES" and r["oss"] == "YES": sup.append(i); c["support"] += 1
            elif r["q27"] == "NO" and r["oss"] == "NO": c["neg"] += 1
            else: ign.append(i); c["ignore"] += 1
        out[qid] = {"support": sup, "ignore": ign}
    if miss: sys.exit(f"FAIL {miss} candidate pairs without verdict")
    json.dump({"version": VERSION, "q": out, "counts": c}, open(a.out, "w")); print(c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cands", "controls", "judge", "verdict"])
    for x in ("--qfile", "--pooldir", "--worlds", "--ids", "--student", "--out", "--pairs", "--verdicts", "--q27", "--oss"): ap.add_argument(x, default="")
    ap.add_argument("--sample", type=int, default=0); ap.add_argument("--show", type=int, default=0); ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args(); {"cands": cands, "controls": controls, "judge": judge, "verdict": verdict}[a.cmd](a)
