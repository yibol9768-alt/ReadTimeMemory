"""Independent judge for the step-3 answer table (audit 11-02, item 11): the reader (Qwen3.8-27B) no longer grades itself.
Judge = gpt-oss-120b (other model family) via /v1/chat/completions, temperature 0, each benchmark with its own scoring:
  lme   : LongMemEval official answer-check prompt (judge.ev.get_anscheck_prompt, as before) -> yes/no
  hard/other (BEAM): BEAM rubric prompt (s1/beam_judge_prompt.txt) per rubric item -> mean of item scores
  ep    : EpBench official judge prompt (ahstat/episodic-memory-benchmark, scoring_answers.py, "Prompt 3") -> lenient F1
  ev    : EverMemBench official llm_judge prompt (EverMind-AI/EverMemBench eval/config/prompts.yaml); letter compare for MCQ
  loc   : LoCoMo official string F1 (snap-research/locomo task_eval/evaluation.py), no LLM
Controls per stratum (first 20 frozen questions): gold answer as the response (positive) and "I don't know." (negative).
usage: python3 a6judge.py --port 18310 [--src out/A6.jsonl --out out/A6J.jsonl --workers 32]; python3 a6judge.py --controls --port ...;
       python3 a6judge.py --sum"""
import argparse, ast, collections, hashlib, json, os, random, re, statistics as st, string, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
S1 = os.path.expanduser("~/data/rtm/s1"); sys.path.insert(0, S1); sys.path.insert(0, ".")
JMODEL = "gpt-oss-120b"
JMAP = {}   # stratum -> (model, port): per-benchmark judge chosen by the control rule (11-02: BEAM -> gemma-4-31b-it)
def jm(s): return JMAP.get(s, (JMODEL, None))[0]
def route(s, port):
    m, p = JMAP.get(s, (JMODEL, None))   # p may be a comma list of ports of the same judge model (load balancing only)
    return m, (random.choice(p.split(",")) if p else port)
LT = {"single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning", "knowledge-update", "single-session-preference"}
QF = {"lme": "q_lmes150.json", "ep": "q_ep150.json", "ev": "q_ev150.json", "loc": "q_loc150.json", "hard": "beam_ex.json", "other": "beam_ex.json"}
BEAM_JP = open(os.path.join(S1, "beam_judge_prompt.txt")).read()

EP_PROMPT = """You are an expert judge evaluating the accuracy of an AI-generated answer against a known groundtruth. Questions can probe for different types or aspects, like what actions or events took place, what people were involved, what were the dates, or what were the locations or spaces.

Question type: {retrieval_type}
Groundtruth: {correct_answer_long}
AI-generated answer: {llm_answer}

Your task:

- Identify all unique items in the AI-generated answer that are relevant to the question type. Answer an empty list [] for this field in case of at least one negative information (e.g., when the answer begins by telling there is no information, or cannot answer)

- Determine a matching score between 0 and 1 for each ground truth item. Give 1 if the item has been found in the relevant items of the AI-generated answer, considering synonyms, paraphrases, or close meanings. Give 0.5 if the item could be considered related to any AI-generated item but without being explicitly stated as such. Give 0 if the item missed mentioning a specific AI-generated item.

- Provide a brief explanation of the evaluation

{adding_text}

Provide your evaluation in the following JSON format:

{{
"identified_items_in_AI_answer": ["AI_answer_item_1", "AI_answer_item_2", ...],
"matching_score": {d}
"explanation": "Brief explanation of your evaluation"
}}"""
EP_SYS = "You are an expert in memory tests."

EV_SYS = "You are an expert grader that determines if answers to questions match a gold standard answer."
EV_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:

(1) a question (about a multi-person group chat),

(2) a 'gold' (ground truth) answer,

(3) a generated answer

which you will score as CORRECT/WRONG.

The questions are about events, facts, or details mentioned in multi-person group chat conversations.

The gold answer is usually a concise answer that includes the key information.

For example:

Question: What project was announced on January 9th?

Gold answer: Carbon Emission Accounting Platform

The generated answer might be longer, but you should be generous with your grading -

as long as it contains the same key information as the gold answer, it should be CORRECT.

For time-related questions, the gold answer will be a specific date/time.

The generated answer might use different formats (e.g., "May 7th" vs "7 May" vs "2025-05-07"),

but as long as it refers to the same date/time, it should be CORRECT.

For the specific window of date, a +/- 1 day difference is acceptable due to timezone processing variations.

For multiple choice questions where the gold answer is a letter (A/B/C/D),

the generated answer should match exactly to be CORRECT.

Now grade this:

Question: {question}

Gold answer: {golden_answer}

Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning,

then finish with CORRECT or WRONG.

Do NOT include both CORRECT and WRONG in your response.

Return the label in JSON format with the key "label": {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


class JudgeFail(Exception):
    """the judge produced no usable verdict (empty, truncated, unparsable, out of range): never a zero score"""


VERSION = "a6judge-v3-2026-11-02"   # summary-only changes since v3 do not change verdicts; 11-02 15:10: EverMem multiple choice by letter (MCQ, own key)


def chat(model, port, system, user, n=4000):
    """content of the final message; empty or length-truncated outputs are retried with more tokens, then JudgeFail"""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    last = None
    for k in range(4):
        body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": n * (2 if k >= 2 else 1)}
        if "Qwen" in model: body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r: ch = json.loads(r.read())["choices"][0]
        except Exception as e:
            last = repr(e); continue
        c = ch["message"].get("content") or ""
        if c.strip() and ch.get("finish_reason") != "length": return c
        last = f"finish={ch.get('finish_reason')} empty={not c.strip()}"
    raise JudgeFail(f"no usable judge output: {last}")


# ---- LoCoMo official F1 (task_eval/evaluation.py) ----
try:
    from nltk.stem import PorterStemmer; _ps = PorterStemmer(); stem = _ps.stem
except Exception:
    stem = None
def _norm(s):   # official normalize_answer: drop commas, lower, remove punctuation, remove articles, fix whitespace
    s = str(s).replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())
def f1_score(pred, gt):
    p = [stem(w) if stem else w for w in _norm(pred).split()]; g = [stem(w) if stem else w for w in _norm(gt).split()]
    common = collections.Counter(p) & collections.Counter(g); ns = sum(common.values())
    if ns == 0: return 0.0
    pr, rc = ns / len(p), ns / len(g); return 2 * pr * rc / (pr + rc)
def f1_multi(pred, gt):
    ps = [x.strip() for x in pred.split(",")]; gs = [x.strip() for x in gt.split(",")]
    return st.mean(max(f1_score(p, g) for p in ps) for g in gs)
def loc_score(q, ans):
    t, gt = q["question_type"], str(q["answer"])
    if t == "multi-hop": return f1_multi(ans, gt)
    if t == "open-domain": gt = gt.split(";")[0].strip()
    return f1_score(ans, gt)


def lit(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return x
    return x


def score(port, s, q, ans):
    if s == "loc": return loc_score(q, ans), "f1"
    if s == "lme":
        import judge
        t = chat(*route(s, port), None, judge.ev.get_anscheck_prompt(q["question_type"] if q["question_type"] in LT else "multi-session", q["question"], q["answer"], ans, abstention="_abs" in q["question_id"]), 2000)
        if not re.search(r"\b(yes|no)\b", t.lower()): raise JudgeFail(f"LME verdict has neither yes nor no: {t[-200:]!r}")
        return float("yes" in t.lower()), t[-200:]   # LongMemEval official rule: 'yes' in response.lower()
    if s in ("hard", "other"):
        import qa
        rub = qa.lst(lit(q.get("rubric"))) if q.get("rubric") else []
        if not rub: raise JudgeFail("BEAM question without rubric")
        sc = []
        for it in rub:
            t = chat(*route(s, port), None, BEAM_JP.replace("<rubric_item>", it).replace("<llm_response>", ans), 2000)
            m = re.search(r'"score"\s*:\s*\[?\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*[\],}]', t)
            if not m: raise JudgeFail(f"BEAM rubric score not parsed: {t[-200:]!r}")
            v = float(m.group(1))
            if not (0.0 <= v <= 1.0): raise JudgeFail(f"BEAM rubric score out of range: {v}")
            sc.append(v)
        return st.mean(sc), f"{len(sc)} rubric items: {sc}"
    if s == "ep":
        items = lit(q.get("answer_items")) or [q["answer"]]
        d = [{it: "score_between_0_and_1"} for it in items]
        t = chat(*route(s, port), EP_SYS, EP_PROMPT.format(retrieval_type=q["question_type"].split("|")[0], correct_answer_long=items, llm_answer=ans, adding_text="", d=json.dumps(d)), 4000)
        try:
            j = t[t.index("{"): t.rindex("}") + 1]
            try: r = json.loads(j)
            except Exception: r = ast.literal_eval(j)
            ms = r["matching_score"]; vals = []
            if not isinstance(ms, list): raise ValueError(f"matching_score is {type(ms).__name__}, not a list")
            for x in ms:
                if isinstance(x, dict):
                    if len(x) != 1: raise ValueError(f"matching_score entry with {len(x)} keys")
                    x = next(iter(x.values()))
                # numeric strings are accepted: the official template asks for {"item": "score_between_0_and_1"}, i.e. a string value
                if isinstance(x, bool) or not isinstance(x, (int, float, str)): raise ValueError(f"bad score type {type(x).__name__}")
                vals.append(float(x))
            if len(vals) != len(items) or any(not (0 <= v <= 1) for v in vals): raise ValueError(f"{len(vals)} scores for {len(items)} items: {vals}")
            ident = r["identified_items_in_AI_answer"]
            if not isinstance(ident, list): raise ValueError("identified items not a list")
        except Exception as e:
            raise JudgeFail(f"EpBench judge output invalid: {e!r} | {t[-200:]!r}")
        ngt = len(items); npred = min(len(ident), ngt); ssum = sum(vals)
        if npred == 0: return 0.0, "no items in answer"   # official: P undefined -> F1 0
        p = ssum / npred; rc = ssum / ngt
        return (2 * p * rc / (p + rc) if p + rc > 0 else 0.0), f"f1_lenient P={p:.2f} R={rc:.2f}"
    if s == "ev":
        g = mc_gold(q)
        if g:   # 11-02: multiple choice (options inside the question text, gold "X. ..."): compare the chosen letter, no LLM
            p = mc_letter(q, ans)
            if p: return float(p == g), f"mcq letter {p} vs {g}"
        t = chat(*route(s, port), EV_SYS, EV_PROMPT.format(question=q["question"], golden_answer=q["answer"], generated_answer=ans), 2000)
        c = t.strip()
        c = re.sub(r"^```(json)?|```$", "", c).strip()
        try:
            j = json.loads(c[c.index("{"): c.rindex("}") + 1]); lab = j["label"]
            if isinstance(lab, dict): lab = lab["label"]
            lab = str(lab).strip().upper()
            if lab not in ("CORRECT", "WRONG"): raise ValueError(lab)
            return float(lab == "CORRECT"), "label"
        except Exception:
            u = c.upper(); hasc = re.search(r"(?<!IN)\bCORRECT\b", u) is not None; hasw = re.search(r"\bWRONG\b|\bINCORRECT\b", u) is not None
            if hasc == hasw: raise JudgeFail(f"EverMem label not found: {c[-200:]!r}")
            return float(hasc), "fallback (official keyword rule, word-bounded)"
    raise ValueError(s)


MCQ = "mcq-letter-v4"   # verdict rule for EverMem multiple-choice questions; part of their judgement key


def mc_opts(q):
    if "\nOptions:" not in q["question"]: return {}
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"(?m)^([A-D])\. (.+)$", q["question"].split("\nOptions:", 1)[1])}


def mc_gold(q):
    m = re.match(r"\s*([A-D])\.\s", str(q.get("answer") or "")); return m.group(1) if (m and mc_opts(q)) else None


def mc_letter(q, ans):
    """the option letter a reader answer commits to, or None (then the LLM judge decides, as before).
    Committing positions decide: a letter opening the answer, and every "Answer: X" / "final answer is X". Letters met
    while discussing the other options ("B is wrong", "unlike option C") do not count. Ambiguous -> None: the committing
    positions disagree, or a "correction"/"correct answer" names another letter. Upper-case letters only ("the answer is
    a woman" is not A). With no committing position: a single bold or "option X" letter, else a verbatim option quote."""
    t = re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()
    com = []
    m = re.match(r"^\W*([A-D])(?:[.):*]|\s*$|\s*\n)", t)
    if m: com.append(m.group(1))
    com += [x.group(1) for x in re.finditer(r"\b(?:[Ff]inal\s+)?[Aa]nswer\b\s*(?:is|:)?\W{0,6}([A-D])(?![A-Za-z])", t)]
    cor = [x.group(1) for x in re.finditer(r"\b(?:[Cc]orrection|[Cc]orrect\s+(?:answer|option|choice))\b\s*(?:is|:)?\W{0,6}(?:[Oo]ption\s*)?\W{0,3}([A-D])(?![A-Za-z])", t)]
    if com or cor:
        L = set(com) | set(cor); return L.pop() if len(L) == 1 else None
    neg = re.compile(r"\W{0,4}(?:is|are)?\s*(?:also\s+)?(?:not\b|incorrect|wrong|unsuitable|inappropriate|ruled out)", re.I)
    if re.search(r"(?i)cannot (?:be )?determine|can't determine|not enough information|unable to (?:answer|determine)|don't know|do not know", t): return None
    ws = [(x.group(1), x.end()) for pat in (r"\b[Oo]ption\s*\**\s*([A-D])\b", r"\*\*([A-D])[.)]?\*\*") for x in re.finditer(pat, t)]
    if any(neg.match(t, e) or re.match(r"\W{0,4}(?:and|or|,)\s", t[e:e + 8]) for _, e in ws): return None   # an option is being ruled out or listed: no clear commitment
    weak = {l for l, _ in ws}
    if weak: return weak.pop() if len(weak) == 1 else None
    hit = [k for k, v in mc_opts(q).items() if len(v) > 40 and v[:120] in t]   # answer quotes exactly one option text
    return hit[0] if len(hit) == 1 else None


def excluded_from_controls(s, q):
    """questions whose expected verdict for 'gold' / 'I don't know' is not clear-cut (audit 11-02)"""
    t = q.get("question_type", "")
    empty = not str(q.get("answer") or "").strip()   # e.g. BEAM preference/instruction following: the rubric, not the answer, is the reference
    return empty or "_abs" in q["question_id"] or t in ("abstention", "single-session-preference", "preference_following", "instruction_following") or (s == "loc" and t == "adversarial")


def ref_answer(s, q):
    a = q["answer"] if not isinstance(q["answer"], list) else "; ".join(map(str, q["answer"]))
    return str(a).split(";")[0].strip() if (s == "loc" and q.get("question_type") == "open-domain") else str(a)


def key(s, m, qid, ans, q=None):
    """judgement cache key; EverMem multiple-choice items also hash the letter rule, so they are re-scored by it"""
    v = VERSION + (MCQ if (s == "ev" and q is not None and mc_gold(q)) else "")
    return f"{s}|{m}|{qid}|{hashlib.md5((v + jm(s) + ans).encode()).hexdigest()}"


def main(a):
    Q = {}
    for s, f in QF.items():
        if f not in Q: Q[f] = {q["question_id"]: q for q in json.load(open(f))}
    if a.controls:
        ids = json.load(open("out/A6_ids.json")); items = []
        for s in QF:
            qs = [Q[QF[s]][i] for i in ids[s] if not excluded_from_controls(s, Q[QF[s]][i])][: a.nctl]
            for q in qs: items += [(s, "POS", q, ref_answer(s, q)), (s, "NEG", q, "I don't know.")]
        out = CTL
    else:
        last = {}
        for p in a.src.split(","):
            if not os.path.exists(p): raise SystemExit(f"FAIL source file missing: {p}")
            for r in map(json.loads, open(p)): last[(r["stratum"], r["m"], r["qid"])] = r   # a reader retry supersedes its error row
        bad = [r for r in last.values() if "error" in r]
        if bad: print(f"WARNING cells whose latest reader row is an error: {len(bad)} (not judged; --sum will FAIL)", flush=True)
        items = [(r["stratum"], r["m"], Q[QF[r["stratum"]]][r["qid"]], r["answer"]) for r in last.values() if "error" not in r]
        out = a.out
    done = set()
    if os.path.exists(out): done = {r["key"] for r in map(json.loads, open(out)) if r.get("status") == "ok"}
    items = [x for x in items if key(x[0], x[1], x[2]["question_id"], x[3], x[2]) not in done]; print("todo", len(items), flush=True)
    f = open(out, "a"); lk = threading.Lock(); P = a.port.split(","); n = [0]; nfail = [0]
    def one(x):
        s, m, q, ans = x
        with lk: n[0] += 1; port = P[n[0] % len(P)]
        base = dict(key=key(s, m, q["question_id"], ans, q), stratum=s, m=m, qid=q["question_id"], judge=jm(s), version=VERSION)
        try: sc, note = score(port, s, q, ans); row = dict(base, status="ok", score=sc, note=str(note)[-300:])
        except Exception as e:
            row = dict(base, status="fail", score=None, error=repr(e)[:600])
            with lk: nfail[0] += 1
        with lk: f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, items))
    print(f"JUDGEDONE failures {nfail[0]}", flush=True)
    if nfail[0]: sys.exit(3)


def expected(a):
    """current expected judging items: {(stratum, method, qid): key} for answers (from the source files) or controls"""
    Q = {}
    for s_, f in QF.items():
        if f not in Q: Q[f] = {q["question_id"]: q for q in json.load(open(f))}
    E = {}
    if a.controls:
        ids = json.load(open("out/A6_ids.json"))
        for s_ in QF:
            for q in [Q[QF[s_]][i] for i in ids[s_] if not excluded_from_controls(s_, Q[QF[s_]][i])][: a.nctl]:
                E[(s_, "POS", q["question_id"])] = key(s_, "POS", q["question_id"], ref_answer(s_, q), q)
                E[(s_, "NEG", q["question_id"])] = key(s_, "NEG", q["question_id"], "I don't know.", q)
    else:
        for p in a.src.split(","):
            if not os.path.exists(p): raise SystemExit(f"FAIL source file missing: {p}")
            for r in map(json.loads, open(p)):   # the latest source row per cell decides; a reader error makes the cell a failure
                E[(r["stratum"], r["m"], r["qid"])] = "READER_ERROR" if "error" in r else key(r["stratum"], r["m"], r["qid"], r["answer"], Q[QF[r["stratum"]]][r["qid"]])
    return E


def current(path, E):
    """per expected item: the LATEST judge row whose key matches the current answer and version"""
    L = {}
    if os.path.exists(path):
        for r in map(json.loads, open(path)):
            k = (r["stratum"], r["m"], r["qid"])
            if E.get(k) == r.get("key"): L[k] = r
    return L


def summ(a):
    ok = True
    # controls
    ac = argparse.Namespace(**{**vars(a), "controls": True}); EC = expected(ac); LC = current(CTL, EC)
    C = collections.defaultdict(list); F = collections.Counter(); M = collections.Counter()
    for k in EC:
        r = LC.get(k)
        if r is None: M[(k[0], k[1])] += 1
        elif r["status"] == "ok": C[(k[0], k[1])].append(r["score"])
        else: F[(k[0], k[1])] += 1
    print("controls: stratum  POS mean (n)  NEG mean (n)  failed  missing  verdict   [PASS = POS >= 0.95, NEG <= 0.05, none failed or missing]")
    for s_ in QF:
        p, ng = C[(s_, "POS")], C[(s_, "NEG")]; fl = F[(s_, "POS")] + F[(s_, "NEG")]; ms = M[(s_, "POS")] + M[(s_, "NEG")]
        v = bool(p) and bool(ng) and st.mean(p) >= 0.95 and st.mean(ng) <= 0.05 and fl == 0 and ms == 0; ok &= v
        print(f"  {s_:6s} {st.mean(p) if p else float('nan'):.2f} ({len(p)})   {st.mean(ng) if ng else float('nan'):.2f} ({len(ng)})   {fl}   {ms}   {'PASS' if v else 'FAIL'}")
    if a.controls_only:
        print("\nOVERALL (controls only)", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 3)
    # answer table: every frozen question x every method present in the source files must be judged ok
    E = expected(a); L = current(a.out, E); ids = json.load(open("out/A6_ids.json"))
    methods = a.methods.split(",") + (["full"] if "A6F" in a.src else []); T = collections.defaultdict(dict); fails = miss = rerr = 0
    for k in E:
        r = L.get(k)
        if E[k] == "READER_ERROR": rerr += 1
        elif r is None: miss += 1
        elif r["status"] == "ok": T[(k[0], k[1])][k[2]] = r["score"]
        else: fails += 1
    nofreeze = sum(len(set(ids[s_]) - set(T.get((s_, m), {}))) for s_ in QF for m in methods if m != "full" or s_ in ("lme", "loc"))
    print(f"\nanswer table (independent judges: {', '.join(sorted({s_ + '=' + jm(s_) for s_ in QF}))}); reader errors {rerr}; judge failures {fails}; not yet judged {miss}; frozen (stratum, method, question) cells without an ok score {nofreeze}")
    for s_ in QF:
        row = []
        for m in methods:
            d = T.get((s_, m), {}); gap = len(set(ids[s_]) - set(d))
            row.append((f"{m} {100*st.mean(d.values()):.1f}" + (f" (MISSING {gap})" if gap else "")) if d else f"{m} -")
        print(f"  {s_:6s} " + " | ".join(row))
    if "CX" in methods:
        print("\nCX minus X (paired, 95% CI, 8000 draws; BEAM and LoCoMo resampled by conversation; complete pairs only)")
        for m in methods:
            if m == "CX": continue
            out = []
            for s_ in QF:
                x, y = T.get((s_, "CX"), {}), T.get((s_, m), {}); K = sorted(set(x) & set(y))
                if not K: continue
                by = {}
                for q in K: by.setdefault(q.split("_")[0] if s_ in ("hard", "other", "loc") else q, []).append(x[q] - y[q])
                cl = list(by); rng = random.Random(0)
                bs = sorted(st.mean([v for c in rng.choices(cl, k=len(cl)) for v in by[c]]) for _ in range(8000))
                out.append(f"{s_} {100*st.mean(v for c in cl for v in by[c]):+.1f} [{100*bs[200]:+.1f},{100*bs[7799]:+.1f}] n={len(K)}")
            print(f"  CX-{m}: " + "; ".join(out))
    ok &= fails == 0 and miss == 0 and rerr == 0 and nofreeze == 0
    print("\nOVERALL", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", default="18310"); ap.add_argument("--src", default="out/A6.jsonl", help="comma list; every file must exist (add out/A6F.jsonl for full context)"); ap.add_argument("--methods", default="rag,A0,RR8,MR4,T1,CX", help="methods the table must contain")
    ap.add_argument("--out", default="out/A6J.jsonl"); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--nctl", type=int, default=20)
    ap.add_argument("--controls", action="store_true"); ap.add_argument("--sum", action="store_true"); ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--judge-model", default="gpt-oss-120b", help="served model name of the judge; outputs are kept per judge")
    ap.add_argument("--beam-judge", default="", help="NAME@PORT judge for the BEAM strata (hard, other)"); a = ap.parse_args()
    JMODEL = a.judge_model; tag = "" if JMODEL == "gpt-oss-120b" else "_" + re.sub(r"[^A-Za-z0-9]+", "", JMODEL)
    if a.beam_judge:   # NAME@PORT; BEAM strata judged by this model, chosen by the control rule
        bn, bp = a.beam_judge.split("@"); JMAP["hard"] = JMAP["other"] = (bn, bp); tag += "_beam" + re.sub(r"[^A-Za-z0-9]+", "", bn)
    CTL = f"out/A6J_controls{tag}.jsonl"
    if a.out == "out/A6J.jsonl": a.out = f"out/A6J{tag}.jsonl"
    summ(a) if a.sum else main(a)
