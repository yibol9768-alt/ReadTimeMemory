"""BEAM (Tavakoli et al., ICLR 2026, arXiv 2510.27246), 100K split: a harder benchmark than LoCoMo.
One user talks with an assistant over 3 sessions (~200 turns, ~500k characters, mostly long assistant answers);
20 probing questions per conversation, 2 for each of 10 abilities (abstention, contradiction resolution, event
ordering, information extraction, instruction following, knowledge update, multi-session reasoning, preference
following, summarization, temporal reasoning). Scoring as in the official code: an LLM judge scores every rubric
nugget 0 / 0.5 / 1 with the official prompt (unified_llm_judge_base_prompt, copied verbatim), averaged per question.
We report that judge score for event ordering too (the official code also computes Kendall tau there; skipped).

Conditions, all under the same 8000-character memory budget and the same reader prompt:
  rag          BM25 + dense over the raw turns (assistant turns cut to 1500 characters), RRF      (paper's RAG analogue)
  lychee       Lychee-lite records (mem/lychee.py, segmentation on the H20 + one LLM call per segment)
  lychee_roll  the same + roll-up records (mem/rollup.py, one extra LLM call per conversation)
  full         the whole conversation, uncut (no memory; ~100K tokens; upper reference, no budget)
  hindsight / lightmem   context retrieved by the external system for the question (runs/beam/ext/<cond>/ctx.jsonl,
               written by mem/ext_*.py in their own environments); shown as is, its size is reported
Shown source turns are cut to 800 characters, so one long assistant answer cannot eat the budget.

Usage: python3 -m mem.beam build|answer|judge|report [--convs 6] [--par 6] [--conds rag,lychee,lychee_roll]
"""
import argparse, ast, collections, json, os, random, re
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, lychee as Y, rollup as U, retrieve as R
from .lme_judge import get_anscheck_prompt as lme_prompt

GEN = "" if os.environ.get("V3_GEN", "1") == "1" else "_g" + os.environ["V3_GEN"]   # answers_g2.jsonl ...
SPLIT = os.environ.get("V3_BEAM_SPLIT", "100k")   # 100k | 1m (1M dev rows 0,1 = AMB units 1,2) | lme (LongMemEval-S)
TEST = os.environ.get("V3_TEST") == "1"   # held-out test sets (only after the method is frozen): 100K test convs = the
# complement of the dev pick, 1M test rows from 1M_test_rows.json, LME test ids from lme_split; outputs go to runs/*_test
DATA = os.path.join(os.path.dirname(__file__), "..", "data", *{"100k": ("beam", "100K_rows.json"),
                    "1m": ("beam", "1M_test_rows.json" if TEST else "1M_dev_rows.json"), "lme": ("longmemeval", "longmemeval_s.json")}[SPLIT])
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", {"100k": "beam", "1m": "beam1m", "lme": "lme"}[SPLIT] + ("_test" if TEST else ""),
                   os.environ.get("V3_OUT_ARM", llm.ARM))   # V3_OUT_ARM: judge another reader's answers with luna
LME_DEV = 60          # PROTOCOL 2: LongMemEval-S dev = 60 questions stratified by type (seed 0); test = 150 of the rest
ABILITIES = ["abstention", "contradiction_resolution", "event_ordering", "information_extraction",
             "instruction_following", "knowledge_update", "multi_session_reasoning", "preference_following",
             "summarization", "temporal_reasoning"]
CONDS = ["full", "rag", "lychee", "lychee_roll", "hindsight", "lightmem", "lightmem_k60", "lightmem_hybrid",
         "amb_hindsight", "amb_bm25", "lychee_b20", "lyx_b20", "lyx_b20_nonb", "rag_b20", "anc", "anc_noq", "ccr_A", "ccr_B", "ccr_C", "wide", "wide_mix", "asp_mix", "asp_full", "nadj", "lychee_cf", "cf_hindsight", "lyrag", "ctr", "ctr_cov", "ctr_rel", "rap", "ctr_tl", "note_hs", "note2_hs", "lychee_b60", "hs60", "lychee_b120", "hs8", "lychee_u27", "hs8n", "hs20", "note_ly", "note3_ly", "note3_only", "note3_8k", "note15_ly", "note3_hs", "note3_deep", "nd_p1234", "nd_p1345", "nd_p1245", "nd_p1235", "nd_p1"]   # amb_*: answers produced inside AMB (luna reader), re-judged here
EXT = ("hindsight", "lightmem")      # conditions whose context comes from an external system (prefix match)
RAW_CUT, SHOW_CUT = 1500, 800
B20 = int(os.environ.get("H33_B", 20000))                         # H33 pilot budget (characters)

ANS_SYS = ("You are the AI assistant in a long conversation with a user that spans several sessions. The earlier "
           "sessions are no longer available; only your memory of them is shown below. Reply to the user's new "
           "message as the assistant, using the memory for anything about the user, their projects, what was said "
           "and when, their preferences and standing instructions.")

M0_PROMPT = ("You are an AI assistant with access to stored memories from prior conversations with a user.\n"
             "Use these memories to answer the following question as accurately and completely as possible.\n\n"
             "IMPORTANT RULES:\n"
             "1. Scan ALL provided memories before answering \u2014 do not stop after the first relevant one.\n"
             "2. If multiple memories contain relevant information, combine and cross-reference them.\n"
             "3. The question may refer to something indirectly (e.g. 'the person I met at the festival') \u2014 resolve "
             "such references by searching the memories before concluding. Only if the specific information is genuinely "
             "absent after a careful search, say exactly: \"I don't have enough information to answer this question.\"\n"
             "4. Be specific and direct \u2014 include exact names, dates, numbers, and details from the memories.\n"
             "5. Do NOT invent or assume information that isn't in the memories.\n"
             "\nQUESTION: {query}\n\nRETRIEVED MEMORIES:\n{context}\n\nANSWER:")

# Official BEAM judge prompt (src/prompts.py, unified_llm_judge_base_prompt), verbatim.
JUDGE_PROMPT = open(os.path.join(os.path.dirname(__file__), "beam_judge_prompt.txt")).read()


def lme_split():
    """Stratified dev / test question ids of LongMemEval-S (type, with _abs as its own stratum), seed 0."""
    d = json.load(open(DATA))
    by = collections.defaultdict(list)
    for x in d:
        by[x["question_type"] + ("_abs" if x["question_id"].endswith("_abs") else "")].append(x["question_id"])
    rng, dev, test = random.Random(0), [], []
    for k in sorted(by):
        ids = sorted(by[k]); rng.shuffle(ids)
        nd = max(1, round(LME_DEV * len(ids) / len(d))); nt = max(1, round(150 * len(ids) / len(d)))
        dev += ids[:nd]; test += ids[nd:nd + nt]
    return d, dev, test


def load_lme():
    d, dev, test = lme_split()
    keep = {x["question_id"]: x for x in d}
    convs = []
    for qid in (test if TEST else dev):
        x = keep[qid]; ss, k = [], 0
        for n, (date, sess) in enumerate(zip(x["haystack_dates"], x["haystack_sessions"]), 1):
            turns = []
            for t in sess:
                k += 1
                turns.append({"id": f"T{k}", "speaker": "User" if t["role"] == "user" else "Assistant",
                              "text": t["content"].strip()})
            ss.append({"n": n, "date": date, "turns": turns})
        ab = x["question_type"] + ("_abs" if qid.endswith("_abs") else "")
        convs.append({"id": qid, "sessions": ss, "probing": {ab: [{"question": x["question"], "answer": x["answer"],
                      "rubric": [str(x["answer"])], "date": x["question_date"]}]}})
    return convs


def load():
    if SPLIT == "lme":
        return load_lme()
    rows = [x["row"] for x in json.load(open(DATA))["rows"]]
    convs = []
    for r in rows:
        ss = []
        for n, chunk in enumerate(r["chat"], 1):
            date = next((t["time_anchor"] for t in chunk if t.get("time_anchor")), "")
            turns = [{"id": f"T{t['id']}", "speaker": "User" if t["role"] == "user" else "Assistant",
                      "text": re.sub(r"\s*->->\s*\d+,\d+\s*$", "", t["content"]).strip()} for t in chunk]
            ss.append({"n": n, "date": date.replace("-", " "), "turns": turns})
        convs.append({"id": r["conversation_id"], "sessions": ss, "probing": ast.literal_eval(r["probing_questions"])})
    return convs


def sess(conv):
    return conv["sessions"]


def cut(ss, n):
    return [dict(s, turns=[dict(t, text=t["text"][:n] + (" [...]" if len(t["text"]) > n else "")) for t in s["turns"]])
            for s in ss]


def picked(convs):
    if SPLIT == "lme" or (TEST and SPLIT == "1m"):
        return list(range(len(convs)))
    order = list(range(len(convs)))
    random.Random(0).shuffle(order)
    if TEST:                                      # 100K: the conversations the dev pick (first ARGS.convs) left out
        return sorted(order[ARGS.convs:])
    return sorted(order[:ARGS.convs])


def questions(convs):
    out = []
    for ci in picked(convs):
        for ab in (sorted(convs[ci]["probing"]) if SPLIT == "lme" else ABILITIES):
            for j, q in enumerate(convs[ci]["probing"].get(ab, [])):
                out.append({"qid": f"b{ci}-{ab}-{j}", "conv": ci, "ability": ab, "question": q["question"],
                            "date": q.get("date", ""),
                            "rubric": q.get("rubric") or [], "gold": str(q.get("answer") or q.get("ideal_answer")
                                                                         or q.get("ideal_summary") or q.get("ideal_response") or "")})
    return out


def build(convs):
    cis = picked(convs)
    ldir = os.path.join(OUT, "lychee"); os.makedirs(ldir, exist_ok=True)
    seg = Y.segments(convs, cis, out=ldir, sess=sess)
    print("segments", {ci: len(v) for ci, v in seg.items()}, flush=True)
    llm.check_quota()

    def one(ci):
        Y.encode(ci, convs, seg[ci], out=ldir, sess=sess)
        U.roll(ci, convs, out=ldir, sess=sess, who_list=["*"])
    with ThreadPoolExecutor(min(ARGS.par, len(cis))) as ex:
        list(ex.map(one, cis))


def lyx(q, p, recs, ss, budget, prov_top, cut_n, nb):
    """H33 "records as an index into the raw dialogue". Same routes and order as lychee.context (Y.context);
    every record added while the list has at most prov_top lines (lychee.context: 6) is followed by its source turns (cut to cut_n characters)
    and, if nb, the turns right before and after each source. prov_top=6, cut_n=SHOW_CUT, nb=False is
    Y.context at another budget."""
    flat = [(t, s["date"]) for s in ss for t in s["turns"]]
    pos = {t["id"]: k for k, (t, _) in enumerate(flat)}
    def show(k):
        t, d = flat[k]
        return Y.show(dict(t, text=t["text"][:cut_n] + (" [...]" if len(t["text"]) > cut_n else "")), d)
    docs = [r["text"] for r in recs]
    bm = L.BM25([Y.doc(r) for r in recs])
    routes = []
    for x in p["queries"]:
        routes.append(R.dense_rank(x, docs))
        routes.append(bm.rank(x + " " + " ".join(p["entities"])))
    w = Y.span(p["time"])
    if w:
        hit = [i for i in bm.rank(q) if Y.span(recs[i]["time"]) and
               Y.span(recs[i]["time"])[0] <= w[1] and Y.span(recs[i]["time"])[1] >= w[0]]
        if hit:
            routes.append(hit)
    order, seen_words, items, used = R.rrf(routes), [], [], set()
    for i in order:
        ws = set(L.toks(recs[i]["text"]))
        if any(len(ws & o) / max(1, len(ws | o)) > 0.8 for o in seen_words):
            continue
        seen_words.append(ws)
        items.append(f"- [{recs[i]['time'] or 'no date'}] {recs[i].get('show', recs[i]['text'])}")
        if len(items) <= prov_top and not recs[i]["type"].startswith("rollup"):
            ks = [pos[sid] for sid in recs[i]["sources"] if sid in pos]
            if nb:
                ks = sorted({k + d for k in ks for d in (-1, 0, 1) if 0 <= k + d < len(flat)})
            for k in ks:
                if flat[k][0]["id"] not in used:
                    used.add(flat[k][0]["id"]); items.append("    " + show(k))
    raw_ids = [t["id"] for t, _ in flat]
    texts = [show(k) for k in range(len(flat))]
    raw = [texts[i] for i in L.BM25(texts).rank(q)[:8] if raw_ids[i] not in used]
    head = L.fill_to(items, budget - 1500)
    return head + "\n\nRaw dialogue matches:\n" + L.fill_to(raw, budget - len(head) - 25)


def order_records(q, p, recs):
    """Lychee's routes (dense + BM25 per planned query, time filter) fused by RRF, near-duplicates dropped."""
    docs = [r["text"] for r in recs]
    bm = L.BM25([Y.doc(r) for r in recs])
    routes = []
    for x in p["queries"]:
        routes.append(R.dense_rank(x, docs))
        routes.append(bm.rank(x + " " + " ".join(p["entities"])))
    w = Y.span(p["time"])
    if w:
        hit = [i for i in bm.rank(q) if Y.span(recs[i]["time"]) and
               Y.span(recs[i]["time"])[0] <= w[1] and Y.span(recs[i]["time"])[1] >= w[0]]
        if hit:
            routes.append(hit)
    out, seen = [], []
    for i in R.rrf(routes):
        ws = set(L.toks(recs[i]["text"]))
        if any(len(ws & o) / max(1, len(ws | o)) > 0.8 for o in seen):
            continue
        seen.append(ws); out.append(i)
    return out


WIDE_K, WIDE_DETAIL = 400, 3500   # H35: records considered for the topic timeline; detail characters in wide_mix
WIDE_AB = set(os.environ.get("H35_AB", "event_ordering,summarization").split(","))


def wide_context(q, p, recs, mix):
    """H35 "area view": the top WIDE_K records of lychee's order, merged by topic (first date, count), topics taken
    in rank order until the budget is full, shown in order of first date. mix: lychee-style detail records first."""
    order = order_records(q, p, recs)
    head = ""
    if mix:
        head = L.fill_to([f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}" for i in order], WIDE_DETAIL)
        head += "\n\nTopic timeline (first date, topic, number of records):\n"
    groups = {}
    for i in order[:WIDE_K]:
        t = recs[i]["topic"].strip() or "misc"
        g = groups.setdefault(t.lower(), {"topic": t, "first": None, "n": 0})
        d = (Y.span(recs[i]["time"]) or (None,))[0]
        if d and (g["first"] is None or d < g["first"]):
            g["first"] = d
        g["n"] += 1
    room, chosen = L.BUDGET - len(head), []
    for g in groups.values():                                            # rank order of first appearance
        line = f"- [{g['first'] or 'no date'}] {g['topic']} ({g['n']})"
        if len(line) + 1 > room:
            break
        room -= len(line) + 1; chosen.append((g["first"] or "9999", line))
    return head + "\n".join(l for _, l in sorted(chosen, key=lambda x: x[0]))


def aspect_full(q, p, recs, asp):
    """H37: wide_mix's detail records, then every aspect digest of the conversation in order of first date."""
    order = order_records(q, p, recs)
    head = L.fill_to([f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}" for i in order], WIDE_DETAIL)
    lines = sorted((a["first"] or "9999", f"- [{a['first'] or 'no date'}] {a['name']}: {' '.join(a['digest'].split())}")
                   for a in asp)
    return head + "\n\nAspect digests (first date, aspect: digest):\n" + "\n".join(l for _, l in lines)


def aspect_context(q, p, recs, asp):
    """H36: wide_mix's detail records, then compiled aspect digests (mem/aspect.py) instead of topic lines. Aspects
    are scored by the summed 1/(60+rank) of their records within the top WIDE_K, taken by score until the budget is
    full, shown in order of first date."""
    order = order_records(q, p, recs)
    head = L.fill_to([f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}" for i in order], WIDE_DETAIL)
    head += "\n\nAspect digests (first date, aspect: digest):\n"
    w = {i: 1 / (60 + n + 1) for n, i in enumerate(order[:WIDE_K])}
    sc = sorted(((sum(w.get(i, 0) for i in a["records"]), k) for k, a in enumerate(asp)), reverse=True)
    room, chosen = L.BUDGET - len(head), []
    for v, k in sc:
        if v <= 0:
            break
        a = asp[k]
        line = f"- [{a['first'] or 'no date'}] {a['name']}: {' '.join(a['digest'].split())}"
        if len(line) + 1 > room:
            continue
        room -= len(line) + 1; chosen.append((a["first"] or "9999", line))
    return head + "\n".join(l for _, l in sorted(chosen, key=lambda x: x[0]))


CF_AB = set(os.environ.get("H39_AB", "contradiction_resolution,knowledge_update").split(","))
CF_SYS = ("You check a memory before it is shown to someone who will answer a question. You get the question and "
          "the memory lines. Find pairs of lines that state things about the same matter that cannot both be true "
          "(e.g. 'has never done X' vs 'did X'). Do not count a later update of a plan, value or status as a "
          "conflict when the lines show it changed over time. Output one JSON object, nothing else: "
          "{\"conflicts\": [{\"a\": \"short quote of one line\", \"b\": \"short quote of the other\", "
          "\"about\": \"what they disagree on\"}]} with an empty list when there are none.")


def conflict_note(question, ctx):
    """H39: read-time conflict check over the context the reader will see (one luna call, effort low)."""
    v = llm.call(CF_SYS, f"Question: {question}\n\nMemory:\n{ctx}", effort="low", max_tokens=4000, tag="h39_cf")
    try:
        cs = json.loads(v[v.index("{"):v.rindex("}") + 1])["conflicts"]
    except Exception:
        cs = []
    cs = [c for c in cs if isinstance(c, dict) and c.get("a") and c.get("b")]
    if not cs:
        return ""
    return "\n\nMemory check - these memory lines conflict with each other:\n" + "\n".join(
        f"- on {c.get('about', '?')}: \"{c['a']}\" vs \"{c['b']}\"" for c in cs)


LYRAG_REC = 4500                  # H40: record characters; the rest of BUDGET goes to independently retrieved raw turns
LYRAG_AB = set(os.environ.get("H40_AB", "information_extraction").split(","))


def lyrag_context(q, p, recs, raws):
    """H40: lychee-ordered records, then raw turns in rag's own fused order (dense + BM25 per planned query)."""
    order = order_records(q, p, recs)
    head = L.fill_to([f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}" for i in order], LYRAG_REC)
    bm = L.BM25(raws)
    ro = R.rrf([r for x in p["queries"] for r in (bm.rank(x), R.dense_rank(x, raws))])
    return head + "\n\nRaw dialogue matches:\n" + L.fill_to([raws[i] for i in ro], L.BUDGET - len(head) - 25)


CTR_MAX = 30000                   # CTR: cap on characters of aspect digests (coarse layer)
SCOPE_SYS = ("You decide how much of a long conversation a question covers. You get a question the user asks about "
             "their past conversation and the list of aspects (threads) of that conversation, numbered, with first "
             "date and number of memory records. Pick the aspects whose whole history must be visible to answer "
             "well. A question about one specific fact, value, plan or preference needs none (an empty list) - the "
             "detailed memory will be retrieved separately. A question asking for an overview, a summary, how "
             "something developed, or the order of several things needs every aspect it spans. Output one JSON "
             "object, nothing else: {\"aspects\": [numbers]}")


CTR_T = 20000                     # CTR v2: one total budget (characters) for cover + refine


def ctr_scope(q, asp):
    listing = "\n".join(f"{k}. {a['name']} (first {a['first'] or '?'}, {len(a['records'])} records)"
                         for k, a in enumerate(asp))
    v = llm.call(SCOPE_SYS, f"Question: {q}\n\nAspects:\n{listing}", effort="low", max_tokens=4000, tag="ctr_scope")
    try:
        return sorted({k for k in json.loads(v[v.index("{"):v.rindex("}") + 1])["aspects"]
                       if isinstance(k, int) and 0 <= k < len(asp)})
    except Exception:
        return []


def ctr2_context(q, p, recs, asp, mode):
    """CTR v2 (refine2 round 1): same scope A(q) (same selector call as ctr), one total budget CTR_T.
    Cover: every selected aspect's digest, each cut to an equal share of at most CTR_T/2. Refine: the rest of the
    budget in records; mode 'rel' = lychee relevance order, 'cov' = round robin over [global relevance list] +
    [each selected aspect's records in relevance order]. Records are shown in date order in both modes."""
    pick = ctr_scope(q, asp)
    order = order_records(q, p, recs)
    cover = ""
    if pick:
        share = (CTR_T // 2) // len(pick)
        lines = sorted((asp[k]["first"] or "9999",
                        f"- [{asp[k]['first'] or 'no date'}] {asp[k]['name']}: {' '.join(asp[k]['digest'].split())[:share]}")
                       for k in pick)
        cover = "Aspect digests (first date, aspect: digest):\n" + "\n".join(l for _, l in lines) + "\n\n"
    room = CTR_T - len(cover)
    if mode == "cov" and pick:
        pos = {i: n for n, i in enumerate(order)}
        lists = [order] + [sorted(asp[k]["records"], key=lambda i: pos.get(i, 10 ** 6)) for k in pick]
        cand, used, ptr = [], set(), [0] * len(lists)
        while any(ptr[j] < len(lists[j]) for j in range(len(lists))) and len(cand) < 3000:
            for j, l in enumerate(lists):
                while ptr[j] < len(l) and l[ptr[j]] in used:
                    ptr[j] += 1
                if ptr[j] < len(l):
                    used.add(l[ptr[j]]); cand.append(l[ptr[j]]); ptr[j] += 1
    else:
        cand = order
    chosen, n = [], 0
    for i in cand:
        line = f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}"
        if n + len(line) + 1 > room:
            continue
        chosen.append(i); n += len(line) + 1
        if room - n < 80:
            break
    chosen.sort(key=lambda i: ((Y.span(recs[i]["time"]) or ("9999",))[0], i))
    return cover + "Detailed memory:\n" + "\n".join(f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}" for i in chosen)


RAP_SYS = ("You read a question about a long past conversation and a gist memory of that conversation: its "
           "aspects (threads), numbered, each with a short dated digest. Choose the aspects whose full detailed "
           "records you want to look up to answer well. Output one JSON object, nothing else: "
           "{\"aspects\": [numbers]}")


def rap_context(q, p, recs, asp, ss):
    """ReadAgent-P style baseline (refine2 round 1): one call over the gist (all aspect digests) picks aspects to
    expand; the context is the gist-free expansion: records of the picked aspects in date order, cut to the same
    number of characters as the ctr context for this question."""
    target = len(ctr_context(q, p, recs, asp, ss))
    gist = "\n".join(f"{k}. {a['name']} ({a['first'] or '?'}): {' '.join(a['digest'].split())}" for k, a in enumerate(asp))
    v = llm.call(RAP_SYS, f"Question: {q}\n\nGist memory:\n{gist}", effort="low", max_tokens=4000, tag="rap_pick")
    try:
        pick = [k for k in json.loads(v[v.index("{"):v.rindex("}") + 1])["aspects"] if isinstance(k, int) and 0 <= k < len(asp)]
    except Exception:
        pick = []
    pool = {i for k in pick for i in asp[k]["records"]}
    if not pool:
        return Y.context(q, p, recs, ss)
    chosen, n = [], 0                          # most relevant records of the picked aspects first, shown by date
    for i in [i for i in order_records(q, p, recs) if i in pool]:
        line = f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}"
        if n + len(line) + 1 > target:
            continue
        chosen.append(i); n += len(line) + 1
    chosen.sort(key=lambda i: ((Y.span(recs[i]["time"]) or ("9999",))[0], i))
    return "Looked-up memory records:\n" + "\n".join(f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}" for i in chosen)


def ctr_tl_context(q, p, recs, asp, ss):
    """Ablation: the ctr context without the 8K fine view (timelines of the selected aspects only)."""
    c = ctr_context(q, p, recs, asp, ss)
    return c.split("\n\nDetailed memory:\n")[0] if c.startswith("Aspect digests") else c


def ctr_context(q, p, recs, asp, ss):
    """CTR (docs/refine2): scope selection over the aspect list, then cover (all digests of the chosen aspects, by
    first date) and refine (lychee's 8K view, records of chosen aspects first)."""
    listing = "\n".join(f"{k}. {a['name']} (first {a['first'] or '?'}, {len(a['records'])} records)"
                         for k, a in enumerate(asp))
    v = llm.call(SCOPE_SYS, f"Question: {q}\n\nAspects:\n{listing}", effort="low", max_tokens=4000, tag="ctr_scope")
    try:
        pick = [k for k in json.loads(v[v.index("{"):v.rindex("}") + 1])["aspects"]
                if isinstance(k, int) and 0 <= k < len(asp)]
    except Exception:
        pick = []
    fine = Y.context(q, p, recs, ss)
    if not pick:
        return fine
    w = {i: 1 / (60 + n + 1) for n, i in enumerate(order_records(q, p, recs)[:WIDE_K])}
    pick = sorted(set(pick), key=lambda k: -sum(w.get(i, 0) for i in asp[k]["records"]))
    lines, room = [], CTR_MAX
    for k in pick:
        a = asp[k]
        line = f"- [{a['first'] or 'no date'}] {a['name']}: {' '.join(a['digest'].split())}"
        if len(line) + 1 > room:
            continue
        room -= len(line) + 1; lines.append((a["first"] or "9999", line))
    cover = "\n".join(l for _, l in sorted(lines))
    return f"Aspect digests (first date, aspect: digest):\n{cover}\n\nDetailed memory:\n{fine}"


NOTE_SYS = ("You prepare evidence notes for someone who will answer a user's question from a long memory of past "
            "conversations with that user. You get the question and the memory. Do not answer the question. Write "
            "notes of at most 6000 characters with these parts: (1) Evidence: every memory item relevant to the "
            "question, stated closely to the memory's own words with its date, in time order, grouped by thread or "
            "topic; keep names, numbers, versions and dates exactly. (2) Changes: values or plans that changed over "
            "time, oldest to newest. (3) Conflicts: statements that disagree with each other, both sides. (4) Not "
            "found: parts of the question the memory does not cover.")


NOTE3_SYS = NOTE_SYS + (" (5) Standing instructions: any instruction or preference the user gave earlier about how "
                        "answers should be written or what they should include (format, level of detail, items to "
                        "always mention), quoted exactly with its date, when it could apply to an answer to this "
                        "question.")


NOTE_PARTS = {"1": "Evidence: every memory item relevant to the question, stated closely to the memory's own words with "
                     "its date, in time order, grouped by thread or topic; keep names, numbers, versions and dates exactly.",
              "2": "Changes: values or plans that changed over time, oldest to newest.",
              "3": "Conflicts: statements that disagree with each other, both sides.",
              "4": "Not found: parts of the question the memory does not cover.",
              "5": NOTE3_SYS.split(" (5) ")[1]}


def note_sys(parts):
    """Leave-one-out ablation of the five note parts (nd_p<kept parts>); same wording as NOTE3_SYS, renumbered."""
    body = " ".join(f"({k}) {NOTE_PARTS[x]}" for k, x in enumerate(parts, 1))
    return NOTE_SYS.split("with these parts:")[0] + ("with these parts: " if len(parts) > 1 else "with this part: ") + body


def evidence_note(question, ctx, sys_=NOTE_SYS, tag="h41_note"):
    """H41: one call compiles a deep recall into question-specific evidence notes (docs/SYNTHESIS.md).
    H41c (NOTE3_SYS) adds the user's standing instructions, which v1 notes dropped."""
    return llm.call(sys_, f"Question: {question}\n\nMemory:\n{ctx}", max_tokens=12000, tag=tag).strip()


CCR_REC = 6500                     # characters for records; the rest of BUDGET is the raw-dialogue tail (as lychee)


def ccr_context(q, p, led, ss, mode):
    """CCR read side (docs/refine/FINAL_PROPOSAL.md) on the slot-keyed ledger led = {"records", "slots"}.
    A: seeds flat. B: seeds + every record of each admitted slot, flat (added records right after their seed).
    C: the same record ids as B, one block per admitted slot in time order, then unkeyed filler.
    A slot is admitted only if its whole chain fits (in both B and C renderings); otherwise it is skipped and its
    records are not used as filler. Same raw-dialogue tail for all modes."""
    recs, slots = led["records"], led["slots"]
    sdate = {t["id"]: s["date"] for s in ss for t in s["turns"]}
    tpos = {t["id"]: k for k, (s, t) in enumerate((s, t) for s in ss for t in s["turns"])}
    def said(i):
        src = [x for x in recs[i]["sources"] if x in tpos]
        return (min(tpos[x] for x in src), sdate[min(src, key=lambda x: tpos[x])]) if src else (10 ** 9, "")
    def line(i):
        d = said(i)[1]
        return f"- [said {d or 'no date'}" + (f" | about {recs[i]['time']}" if recs[i]["time"] else "") + f"] {recs[i]['text']}"
    members = collections.defaultdict(list)
    for i, r in enumerate(recs):
        for sid in r.get("slots", []):
            members[sid].append(i)
    order = order_records(q, p, recs)
    def render(sel, mode_):
        if mode_ in ("A", "B"):
            return "\n".join(line(i) for i in sel["flat"])
        out, shown = [], set()
        for sid in sel["slots"]:
            rows = [i for i in sorted(members[sid], key=lambda i: said(i)[0]) if i not in shown]
            shown |= set(rows)
            out.append(f"Topic: {slots[sid]}\n" + "\n".join(line(i) for i in rows))
        rest = [i for i in sel["flat"] if i not in shown]
        if rest:
            out.append("Other records:\n" + "\n".join(line(i) for i in rest))
        return "\n\n".join(out)
    sel = {"flat": [], "slots": []}
    used, skipped = set(), set()
    for i in order:
        if i in used:
            continue
        if mode == "A":
            group, new_slots = [i], []
        else:
            sids = [x for x in recs[i].get("slots", []) if x not in skipped]
            if recs[i].get("slots") and not sids:
                continue                                   # every slot of this record was skipped
            new_slots = [x for x in sids if x not in sel["slots"]]
            group = [i] + [j for x in new_slots for j in sorted(members[x], key=lambda j: said(j)[0])
                           if j != i and j not in used]
            group = list(dict.fromkeys(group))
        trial = {"flat": sel["flat"] + [j for j in group if j not in used], "slots": sel["slots"] + new_slots}
        modes = ("A",) if mode == "A" else ("B", "C")
        if all(len(render(trial, m)) <= CCR_REC for m in modes):
            sel = trial; used |= set(group)
        elif mode != "A":
            skipped |= set(new_slots)
    turns = [Y.show(t, s["date"]) for s in ss for t in s["turns"]]
    shown_turns = [t[:SHOW_CUT] for t in turns]
    raw = [shown_turns[k] for k in L.BM25(shown_turns).rank(q)[:8]]
    head = render(sel, "B" if mode == "B" else mode)
    return head + "\n\nRaw dialogue matches:\n" + L.fill_to(raw, L.BUDGET - len(head) - 25)


def raw_docs(ss):
    return [f"[{t['id']}] ({s['date']}) {t['speaker']}: {t['text'][:RAW_CUT]}" for s in ss for t in s["turns"]]


def answer(convs):
    qs = questions(convs)
    cis = sorted({q["conv"] for q in qs})
    ldir = os.path.join(OUT, "lychee")
    recs = {ci: json.load(open(os.path.join(ldir, f"mem_{ci}.json")))["records"] for ci in cis}
    rolls = {ci: json.load(open(os.path.join(ldir, f"roll_{ci}.json")))["rollups"] for ci in cis}
    pools = {"lychee": recs, "lychee_roll": {ci: recs[ci] + rolls[ci] for ci in cis}}
    if "nadj" in ARGS.conds:                          # H38 non-adjudicating ledger (mem/nadj.py)
        pools["nadj"] = {ci: json.load(open(os.path.join(OUT, "nadj", f"mem_{ci}.json")))["records"] for ci in cis}
    if {"ccr_A", "ccr_B", "ccr_C"} & set(ARGS.conds):     # CCR slot-keyed ledger (mem/ccr.py)
        cdir = os.path.join(OUT, "ccr")
        ccr = {ci: json.load(open(os.path.join(cdir, f"mem_{ci}.json"))) for ci in cis}
        for m in ("ccr_A", "ccr_B", "ccr_C"):
            pools[m] = {ci: led["records"] for ci, led in ccr.items()}
    if {"anc", "anc_noq"} & set(ARGS.conds):          # H34 anchored records (mem/anchor.py)
        adir = os.path.join(OUT, "anchor")
        anc = {ci: json.load(open(os.path.join(adir, f"mem_{ci}.json")))["records"] for ci in cis}
        pools["anc_noq"] = anc
        pools["anc"] = {ci: [dict(r, show=r["text"] + (f' (said: "{r["quote"]}")' if r.get("quote") else ""))
                             for r in rs] for ci, rs in anc.items()}
    shown = {ci: cut(convs[ci]["sessions"], SHOW_CUT) for ci in cis}
    raws = {ci: raw_docs(convs[ci]["sessions"]) for ci in cis}
    llm.check_quota()
    with ThreadPoolExecutor(ARGS.par) as ex:
        plans = list(ex.map(lambda q: Y.plan(q["question"]), qs))
    R.precompute_ranks([[r["text"] for r in m] for c in pools for m in pools[c].values()] + list(raws.values()),
                       [([r["text"] for r in pools[c][q["conv"]]], x) for c in pools for q, p in zip(qs, plans)
                        for x in p["queries"]] + [(raws[q["conv"]], x) for q, p in zip(qs, plans) for x in p["queries"]])
    ext = {}
    for c in [c.split("~")[0] for c in ARGS.conds] + (["hindsight"] if {"cf_hindsight", "note_hs", "note2_hs", "hs60", "hs8", "hs8n", "hs20", "note3_hs"} & {c.split("~")[0] for c in ARGS.conds} else []):
        f = os.path.join(OUT, "ext", c, "ctx.jsonl")
        if c.startswith(EXT) and os.path.exists(f):
            ext[c] = {r["qid"]: r["context"] for r in map(json.loads, open(f))}

    route = json.load(open(os.path.join(OUT, "route.json"))) if os.path.exists(os.path.join(OUT, "route.json")) else {}
    asp = ({ci: json.load(open(os.path.join(OUT, "aspect", f"asp_{ci}.json")))["aspects"] for ci in cis}
           if {"asp_mix", "asp_full", "ctr", "ctr_cov", "ctr_rel", "rap", "ctr_tl"} & {c.split("~")[0] for c in ARGS.conds} else {})
    path = os.path.join(OUT, "answers" + GEN + ".jsonl")
    done = {(r["qid"], r["cond"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(q, p, c) for q, p in zip(qs, plans) for c in ARGS.conds for b in [c.split("~")[0]]
            if (q["qid"], c) not in done
            and (not b.startswith(EXT) or q["qid"] in ext.get(b, {}))
            and (not b.startswith("wide") or q["ability"] in WIDE_AB)
            and (not (b.startswith("asp_") or b in ("lychee_b60", "hs60", "lychee_b120")) or route.get(q["qid"]) == "AREA"
                 or os.environ.get("V3_ALL_ROUTES"))      # budget-curve runs: deep conditions on every question
            and (b not in ("hs60", "hs8", "hs8n", "hs20") or q["qid"] in ext.get("hindsight", {}))
            and (b not in ("lychee_cf", "cf_hindsight") or q["ability"] in CF_AB)
            and (b not in ("cf_hindsight", "note_hs", "note2_hs") or q["qid"] in ext.get("hindsight", {}))
            and (b != "lyrag" or q["ability"] in LYRAG_AB)
            and (not os.environ.get("V3_ONLY_ROUTE") or route.get(q["qid"]) == os.environ["V3_ONLY_ROUTE"])]

    def context(q, p, c):
        ci = q["conv"]
        if c == "full":
            return "\n".join(f"[{t['id']}] ({s['date']}) {t['speaker']}: {t['text']}"
                             for s in convs[ci]["sessions"] for t in s["turns"])
        if c.startswith(EXT):
            return ext[c][q["qid"]]
        if c == "rag":
            docs = raws[ci]; bm = L.BM25(docs)
            order = R.rrf([r for x in p["queries"] for r in (bm.rank(x), R.dense_rank(x, docs))])
            return L.fill([docs[i] for i in order])
        if c == "rag_b20":
            docs = raws[ci]; bm = L.BM25(docs)
            order = R.rrf([r for x in p["queries"] for r in (bm.rank(x), R.dense_rank(x, docs))])
            return L.fill_to([docs[i] for i in order], B20)
        if c.startswith("nd_p"):                # note-part ablation of note3_deep: only the kept parts are asked for
            deep = lyx(q["question"], p, recs[ci], convs[ci]["sessions"], 60000, 6, SHOW_CUT, False)
            return ("Evidence notes compiled from memory:\n" + evidence_note(q["question"], deep, note_sys(c[4:]), "h41c_note")
                    + "\n\nDetailed memory:\n" + deep)
        if c == "note3_deep":                    # H41g: note3 notes followed by the full 60K record recall (not the 8K view)
            deep = lyx(q["question"], p, recs[ci], convs[ci]["sessions"], 60000, 6, SHOW_CUT, False)
            return ("Evidence notes compiled from memory:\n" + evidence_note(q["question"], deep, NOTE3_SYS, "h41c_note")
                    + "\n\nDetailed memory:\n" + deep)
        if c == "note15_ly":                     # H41e: note3 with a 15000-character cap (area questions need more material)
            deep = lyx(q["question"], p, recs[ci], convs[ci]["sessions"], 60000, 6, SHOW_CUT, False)
            fine = Y.context(q["question"], p, pools["lychee"][ci], shown[ci])
            note = evidence_note(q["question"], deep, NOTE3_SYS.replace("at most 6000 characters", "at most 15000 characters"),
                                 "h41e_note")
            return "Evidence notes compiled from memory:\n" + note + "\n\nDetailed memory:\n" + fine
        if c in ("note3_only", "note3_8k"):      # H41d ablations: reader sees notes only / notes compiled from the 8K view
            fine = Y.context(q["question"], p, pools["lychee"][ci], shown[ci])
            if c == "note3_only":
                deep = lyx(q["question"], p, recs[ci], convs[ci]["sessions"], 60000, 6, SHOW_CUT, False)
                return "Evidence notes compiled from memory:\n" + evidence_note(q["question"], deep, NOTE3_SYS, "h41c_note")
            return ("Evidence notes compiled from memory:\n" + evidence_note(q["question"], fine, NOTE3_SYS, "h41c_note")
                    + "\n\nDetailed memory:\n" + fine)
        if c in ("note_ly", "note3_ly"):         # evidence notes compiled from our 60K record recall + the 8K view
            deep = lyx(q["question"], p, recs[ci], convs[ci]["sessions"], 60000, 6, SHOW_CUT, False)
            fine = Y.context(q["question"], p, pools["lychee"][ci], shown[ci])
            note = (evidence_note(q["question"], deep) if c == "note_ly"
                    else evidence_note(q["question"], deep, NOTE3_SYS, "h41c_note"))
            return ("Evidence notes compiled from memory:\n" + note
                    + "\n\nDetailed memory:\n" + fine)
        if c in ("note_hs", "note2_hs"):
            h = ext["hindsight"][q["qid"]]
            note = "Evidence notes compiled from memory:\n" + evidence_note(q["question"], h)
            return note if c == "note_hs" else note + "\n\nFull retrieved memory:\n" + h
        if c in ("rap", "ctr_tl"):
            base = (rap_context if c == "rap" else ctr_tl_context)(q["question"], p, recs[ci], asp[ci], shown[ci])
            return base + conflict_note(q["question"], base)
        if c in ("ctr_cov", "ctr_rel"):
            base = ctr2_context(q["question"], p, recs[ci], asp[ci], c[4:])
            return base + conflict_note(q["question"], base)
        if c == "ctr":
            base = ctr_context(q["question"], p, recs[ci], asp[ci], shown[ci])
            return base + conflict_note(q["question"], base)
        if c == "lyrag":
            return lyrag_context(q["question"], p, recs[ci], raws[ci])
        if c == "cf_hindsight":                  # H39 plug-in on an external memory's context
            return ext["hindsight"][q["qid"]] + conflict_note(q["question"], ext["hindsight"][q["qid"]])
        if c == "lychee_cf":
            base = Y.context(q["question"], p, pools["lychee"][ci], shown[ci])
            return base + conflict_note(q["question"], base)
        if c == "asp_full":
            return aspect_full(q["question"], p, recs[ci], asp[ci])
        if c == "asp_mix":
            return aspect_context(q["question"], p, recs[ci], asp[ci])
        if c in ("wide", "wide_mix"):
            return wide_context(q["question"], p, recs[ci], c == "wide_mix")
        if c.startswith("ccr_"):
            return ccr_context(q["question"], p, ccr[ci], convs[ci]["sessions"], c[-1])
        if c in ("anc", "anc_noq"):
            return lyx(q["question"], p, pools[c][ci], convs[ci]["sessions"], L.BUDGET, 6, SHOW_CUT, False)
        if c == "note3_hs":                      # H41f: the same note step on Hindsight's context (notes from its full
            h = ext["hindsight"][q["qid"]]       # ~124K context + its first 8K cut at a memory boundary)
            cut = h.rfind("\n## Memory ", 0, 8000)
            fine = h[:cut] if len(h) > 8000 and cut > 0 else h[:8000]
            return ("Evidence notes compiled from memory:\n" + evidence_note(q["question"], h, NOTE3_SYS, "h41c_note")
                    + "\n\nDetailed memory:\n" + fine)
        if c in ("hs60", "hs8", "hs8n", "hs20"):  # Hindsight's own ranked memories, cut at a memory boundary
            lim = {"hs60": 60000, "hs20": 20000}.get(c, 8000)
            h = ext["hindsight"][q["qid"]]
            cut = h.rfind("\n## Memory ", 0, lim)
            base = h[:cut] if len(h) > lim and cut > 0 else h[:lim]
            return base + conflict_note(q["question"], base) if c == "hs8" else base
        if c == "lychee_u27":                    # uniform budget = ours60's mean reader budget on 1M (26.8K)
            return lyx(q["question"], p, recs[ci], convs[ci]["sessions"], 26800, 6, SHOW_CUT, False)
        if c in ("lychee_b60", "lychee_b120"):  # deep record recall for area questions (router AREA only)
            return lyx(q["question"], p, recs[ci], convs[ci]["sessions"], 60000 if c == "lychee_b60" else 124000, 6,
                       SHOW_CUT, False)
        if c == "lychee_b20":
            return lyx(q["question"], p, recs[ci], convs[ci]["sessions"], B20, 6, SHOW_CUT, False)
        if c == "lyx_b20":
            return lyx(q["question"], p, recs[ci], convs[ci]["sessions"], B20, 10 ** 6, RAW_CUT, True)
        if c == "lyx_b20_nonb":
            return lyx(q["question"], p, recs[ci], convs[ci]["sessions"], B20, 10 ** 6, RAW_CUT, False)
        return Y.context(q["question"], p, pools[c][ci], shown[ci])

    if ARGS.step == "contexts":          # write the exact contexts the reader saw (cached ranks: no H20, no LLM)
        for c in ARGS.conds:
            if c.startswith(EXT) or c == "full":
                continue
            d = os.path.join(OUT, "ext", c); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "ctx.jsonl"), "w") as f:
                for q, p in zip(qs, plans):
                    f.write(json.dumps({"qid": q["qid"], "context": context(q, p, c)}, ensure_ascii=False) + "\n")
            print("contexts", c, len(qs), flush=True)
        return

    def one(job):
        q, p, c = job
        ctx = context(q, p, c.split("~")[0])
        now = f"(Current date: {q['date']})\n" if q.get("date") else ""
        if c.endswith("~m0"):                 # mem0's BEAM answer prompt (AMB build_rag_prompt, verbatim)
            a = llm.call("", M0_PROMPT.format(query=now + q["question"], context=ctx), tag="beam_ans_" + c)
        else:
            a = llm.call(ANS_SYS, f"Memory:\n{ctx}\n\n{now}User: {q['question']}", tag="beam_ans_" + c)
        return {"qid": q["qid"], "cond": c, "ability": q["ability"], "q": q["question"], "gold": q["gold"],
                "rubric": q["rubric"], "pred": a.strip(), "ctx_chars": len(ctx)}
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


def judge(convs):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "answers" + GEN + ".jsonl"))]
    path = os.path.join(OUT, "judged" + GEN + ".jsonl")
    done = {(r["qid"], r["cond"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    rows = [r for r in rows if (r["qid"], r["cond"]) not in done]
    llm.check_quota()

    def item(r, it):
        if SPLIT == "lme":              # official LongMemEval check (evaluate_qa.get_anscheck_prompt), yes / no
            t = r["ability"].replace("_abs", "")
            v = llm.call("", lme_prompt(t, r["q"], it, r["pred"], r["ability"].endswith("_abs")), effort="low",
                         max_tokens=4000, tag="lme_judge")
            return 1.0 if "yes" in v.lower() else 0.0
        v = llm.call("", JUDGE_PROMPT.replace("<rubric_item>", it).replace("<llm_response>", r["pred"]),
                     effort="low", max_tokens=4000, tag="beam_judge")
        m = re.search(r'"score"\s*:\s*\[?\s*([01](?:\.\d+)?)', v)
        return float(m.group(1)) if m else 0.0

    def one(r):
        s = [item(r, it) for it in r["rubric"]] or [0.0]
        return dict(r, scores=s, score=sum(s) / len(s))
    with ThreadPoolExecutor(ARGS.par) as ex, open(path, "a") as f:
        for r in ex.map(one, rows):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()


def report(convs):
    keep = {q["qid"] for q in questions(convs)}
    rows = {(r["qid"], r["cond"]): r for r in map(json.loads, open(os.path.join(OUT, "judged" + GEN + ".jsonl")))
            if r["qid"] in keep}
    by = collections.defaultdict(list)
    for (qid, c), r in rows.items():
        by[(c, r["ability"])].append(r["score"]); by[(c, "all")].append(r["score"])
    abil = ABILITIES if SPLIT != "lme" else sorted({r["ability"] for r in rows.values()})
    short = {a: (a.split("_")[0][:9] if SPLIT != "lme" else a.replace("single-session-", "ss-")[:9]) for a in abil}
    print(f"{'condition':12s}{'all':>8s}" + "".join(f"{short[a]:>10s}" for a in abil) + "   n")
    for c in CONDS:
        if by[(c, "all")]:
            print(f"{c:12s}{100 * sum(by[(c, 'all')]) / len(by[(c, 'all')]):8.1f}" +
                  "".join(f"{100 * sum(by[(c, a)]) / len(by[(c, a)]):10.1f}" if by[(c, a)] else " " * 10
                          for a in abil) + f"   {len(by[(c, 'all')])}")
    if by[("lychee", "all")] and by[("lychee_roll", "all")]:
        pair = [(rows[(q, "lychee")]["score"], rows[(q, "lychee_roll")]["score"]) for q in keep
                if (q, "lychee") in rows and (q, "lychee_roll") in rows]
        print(f"paired lychee_roll - lychee: {100 * sum(b - a for a, b in pair) / len(pair):+.1f} over {len(pair)}; "
              f"better {sum(b > a for a, b in pair)}, worse {sum(a > b for a, b in pair)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "answer", "judge", "report", "contexts"])
    ap.add_argument("--convs", type=int, default=6)
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--conds", default=",".join(CONDS))
    ARGS = ap.parse_args()
    ARGS.conds = ARGS.conds.split(",")
    os.makedirs(OUT, exist_ok=True)
    {"build": build, "answer": answer, "judge": judge, "report": report, "contexts": answer}[ARGS.step](load())
