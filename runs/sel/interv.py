"""A2 cue interventions (11-01, after the Codex audit). Each edit takes a question and returns (edited copy, log) or raises Reject(reason).
Label-preserving edits only; state/scope flips are label mappings and live elsewhere.
usage: python3 interv.py NAME QFILE OUTFILE [--seed S]. OUTFILE keeps eligible questions only, OUTFILE.log.json keeps every mapping and every rejection.
Ids become <edit>s<seed>-<original>: code that keys a cache on the part before the first underscore (p0.py, l27.py) then gets a new key per edit and history.
Randomness is keyed on the history text, so questions sharing a history get the same edited history and one cache stays valid for all of them.
Candidate pools hold round indices, so rebuild pools on the edited text; never reuse an old pool. Score with cuesum.py, not ssum.py."""
import argparse, collections, copy, datetime, hashlib, json, random, re

class Reject(Exception): pass

MONTHS = "January February March April May June July August September October November December".split()
MN = "|".join(MONTHS)
MPAT = re.compile(r"\b(" + MN + r") (\d{1,2})(?:([-–])(\d{1,2}))?\b(?![\d/])(?:(,? )(\d{4})\b)?")
DPAT = re.compile(r"\b(" + MN + r")-(\d{2})-(\d{4})\b")
IPAT = re.compile(r"\b(\d{4})([-/])(\d{2})\2(\d{2})(?!\d)( \([A-Z][a-z]{2}\))?")
SYL = "ba ve zo ri ka mu lor tem dra vin ol sa keth ny".split()
STOP = set("The This That These Those What When Where Which Who Why How Can Could Would Should Will Did Does Do Is Are Was Were Have Has Had Please Thanks Thank Hey Hi Hello Yes No Okay Sure Great Also But And Or If So Then Here There Let Now Just Monday Tuesday Wednesday Thursday Friday Saturday Sunday User Assistant Group Chapter I We You My Our".split()) | set(MONTHS)

def pseudo(r):
    """capitalised non-word: carries no type, so it fits people, places and products alike"""
    return "".join(r.choice(SYL) for _ in range(3)).capitalize()

def turns(q):
    for s in q["haystack_sessions"]:
        for t in s: yield t

def gold_rounds(q): return [t["content"] for t in turns(q) if t.get("has_answer")]

def key_text(q):
    """everything the answer depends on: question, answer, rubric, options, gold rounds"""
    o = [q.get("question") or "", json.dumps(q.get("answer"), ensure_ascii=False), json.dumps(q.get("rubric"), ensure_ascii=False), json.dumps(q.get("options"), ensure_ascii=False), json.dumps(q.get("answer_items"), ensure_ascii=False)]
    return o + gold_rounds(q)

def ask_text(q): return " ".join(key_text(q)[:1] + [json.dumps(q.get("options"), ensure_ascii=False)])

def hkey(q):
    """same history text, same key"""
    h = hashlib.md5()
    for x in q["haystack_dates"]: h.update(str(x).encode())
    for t in turns(q): h.update(t["content"].encode())
    return h.hexdigest()

GLOBAL_LOW, GLOBAL_WORDS = set(), set()   # filled by run(): vocabulary of the whole file, so a common word is known even if one history never uses it in lower case
def bare(x): return re.sub(r"\S*[/@.]\w\S*", " ", x)   # URLs and addresses say nothing about whether a word is common

def set_vocab(Q):
    GLOBAL_LOW.clear(); GLOBAL_WORDS.clear(); seen = set()
    for q in Q:
        for x in key_text(q)[:5] + [t["content"] for t in turns(q)]:
            if x in seen: continue
            seen.add(x); GLOBAL_LOW.update(re.findall(r"\b[a-z]{3,}\b", bare(x))); GLOBAL_WORDS.update(re.findall(r"[a-z]+", x.lower()))

def edit_all(q, f, sf=None):
    """sf(i) gives the function for session i when it differs from f"""
    q = copy.deepcopy(q)
    for i, s in enumerate(q["haystack_sessions"]):
        g = sf(i) if sf else f
        for t in s: t["content"] = g(t["content"])
    def w(x):
        if isinstance(x, str): return f(x)
        if isinstance(x, list): return [w(v) for v in x]
        if isinstance(x, dict): return {k: w(v) for k, v in x.items()}
        return x
    for k in ("question", "answer", "rubric", "options", "answer_items", "cue", "evidence_refs"):
        if k in q: q[k] = w(q[k])
    return q

# ---- rename
NAME_Q = re.compile(r"\b(named?|called|spell\w*|initials?|letters?|nationality|languages?|countr(?:y|ies)|cit(?:y|ies)|continent|brand|compan(?:y|ies)|what kind of|what type of)\b", re.I)

def corpus_low(q):
    low = set()
    for x in [t["content"] for t in turns(q)] + key_text(q)[:5]: low |= set(re.findall(r"\b[a-z]{3,}\b", bare(x)))
    return low

def names_of(q, top=12):
    """capitalised mid-sentence tokens seen 3+ times, never in lower case anywhere (rounds, question, answer), not a numbered label such as Option 1"""
    c = collections.Counter(); low = corpus_low(q) | GLOBAL_LOW
    for t in turns(q):
        for m in re.finditer(r"(?<=[a-z,;:] )([A-Z][a-z]{2,})\b(?!\s+(?:\d|[A-Z]\b))", t["content"]):
            if m.group(1) not in STOP: c[m.group(1)] += 1
    return [w for w, n in sorted(c.items(), key=lambda x: (-x[1], x[0])) if n >= 3 and w.lower() not in low][:top]

def world_entities(q):
    m = re.match(r"w(\d+)_", q["question_id"])
    if not m: return []
    try: return sorted(set(json.load(open(f"ws4/w{int(m.group(1)):05d}.json"))["entities"]), key=len, reverse=True)
    except OSError: return []

def rename(q, seed=0):
    if NAME_Q.search(ask_text(q)): raise Reject("question depends on what a name means")
    r = random.Random(f"{seed}:{hkey(q)}"); W = world_entities(q); N = W or names_of(q)
    if not N: raise Reject("no names found")
    known = GLOBAL_WORDS or set(re.findall(r"[a-z]+", " ".join(key_text(q) + [t["content"] for t in turns(q)]).lower()))
    M = {}; used = set()
    for w in N:
        x = pseudo(r)
        while x in used or x.lower() in known: x = pseudo(r)
        used.add(x); M[w.lower() if W else w] = x
    # world entities are lower-case noun phrases, so match them in any case; real names match in their exact capitalised form
    pat = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, sorted(M, key=len, reverse=True))) + r")(?![A-Za-z0-9])", re.I if W else 0)
    return edit_all(q, lambda s: pat.sub(lambda m: M[m.group(1).lower() if W else m.group(1)], s)), {"map": M, "world_noun_phrases": bool(W)}

# ---- shift
SPAN_Q = re.compile(r"\b(how old|years? old|ages?|birthdays?|anniversar\w+|how (?:many|long)\b.{0,30}\b(?:months?|years?)|what (?:month|year|season|day of)|which (?:month|year|season))\b", re.I | re.S)
LEFT = re.compile(r"\b(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?\b|\b(Christmas|Easter|Thanksgiving|Halloween|New Year|Valentine|Labor Day|Memorial Day|Independence Day|Black Friday)\b|\b(spring|summer|autumn|fall|winter)\b|\b(?:19|20)\d{2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?(?: of)? (?:" + MN + r")\b|\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.? \d{1,2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
    r"|\b(?:January|February|March|April|June|July|August|September|October|November|December)\b|\bin May\b|\bMay \d{4}\b", re.I)

def strip_dates(s):
    return re.sub(r"\((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\)", " ", IPAT.sub(" ", DPAT.sub(" ", MPAT.sub(" ", s))))

def shift(q, seed=0):
    """move every date by the same number of days. Rejects a question when the answer could rest on a time expression this cannot move"""
    if SPAN_Q.search(ask_text(q)): raise Reject("age, span, weekday or calendar question")
    for x in key_text(q):
        m = LEFT.search(strip_dates(x))
        if m: raise Reject("key text has a time expression that cannot be moved: " + m.group(0).lower())
    r = random.Random(f"{seed}:{hkey(q)}"); d = r.choice([-1, 1]) * r.randint(40, 300); step = datetime.timedelta(days=d); bad = []
    year = lambda x, o: int(m.group()) if (m := re.search(r"\b(?:19|20)\d{2}\b", str(x))) else o
    qy = year(q.get("question_date"), year(q["haystack_dates"][-1] if q["haystack_dates"] else "", 2000))
    def mv(s, y0=qy):
        """a date without a year takes the year of its session, so March 1 and 2023-03-01 in one session land on the same day"""
        def f(m):
            yr = int(m.group(6)) if m.group(6) else y0
            try:
                x = datetime.date(yr, MONTHS.index(m.group(1)) + 1, int(m.group(2))) + step
                y = datetime.date(yr, MONTHS.index(m.group(1)) + 1, int(m.group(4))) + step if m.group(4) else None
            except ValueError: bad.append(m.group(0)); return m.group(0)
            o = f"{MONTHS[x.month - 1]} {x.day:02d}" if len(m.group(2)) == 2 else f"{MONTHS[x.month - 1]} {x.day}"
            if y:
                if m.group(6) and y.year != x.year: o += f"{m.group(5)}{x.year}"
                o += f"{m.group(3)}{y.day}" if (y.month, y.year) == (x.month, x.year) else f"{m.group(3)}{MONTHS[y.month - 1]} {y.day}"
            return o + (f"{m.group(5)}{(y or x).year}" if m.group(6) else "")
        return MPAT.sub(f, s)
    def dash(s):
        def f(m):
            try: x = datetime.date(int(m.group(3)), MONTHS.index(m.group(1)) + 1, int(m.group(2))) + step
            except ValueError: bad.append(m.group(0)); return m.group(0)
            return f"{MONTHS[x.month - 1]}-{x.day:02d}-{x.year}"
        return DPAT.sub(f, s)
    def iso(s):
        def f(m):
            try: x = datetime.date(int(m.group(1)), int(m.group(3)), int(m.group(4))) + step
            except ValueError: bad.append(m.group(0)); return m.group(0)
            o = x.isoformat().replace("-", m.group(2))
            return o + f" ({x.strftime('%a')})" if m.group(5) else o
        return IPAT.sub(f, s)
    g = lambda s, y=qy: dash(iso(mv(s, y)))
    Y = [year(x, qy) for x in q["haystack_dates"]]; n0 = len(bad)
    for x in key_text(q):
        n = len(bad); o = {g(x, y) for y in set(Y) | {qy}}
        if len(bad) > n: raise Reject("key text has an invalid date: " + bad[-1])
        if len(o) > 1: raise Reject("key text has a date whose shift depends on the year")   # a year-free date crossing a leap day lands on different days
    n0 = len(bad)
    q = edit_all(q, g, lambda i: (lambda s: g(s, Y[i] if i < len(Y) else qy)))
    q["haystack_dates"] = [g(str(x)) for x in q["haystack_dates"]]; q["question_date"] = g(str(q.get("question_date", "")))
    return q, {"days": d, "bad_outside_key_text": bad[n0:]}

# ---- shuffle
ORDER_T = re.compile(r"tempor|time|chrono|order|knowledge|update|contradict", re.I)
ORDER_Q = re.compile(r"\b(first|last|latest|earliest|before|after|previous|prior|earlier|later|recent|most recently|changed?|updated?|currently|now|still|sequence|order)\b", re.I)
REFER = re.compile(r"\b(as i (?:said|mentioned)|like i (?:said|mentioned)|as mentioned|mentioned (?:earlier|before)|last time|earlier|previously|the same one|that one|again)\b", re.I)

def shuffle_sessions(q, seed=0):
    """present sessions in random order with their dates attached; gold marks and stable message ids travel with the rounds"""
    if ORDER_T.search(str(q.get("question_type"))): raise Reject("question type uses order")
    if ORDER_Q.search(ask_text(q)): raise Reject("question wording uses order")
    if any(REFER.search(x) for x in gold_rounds(q)): raise Reject("gold round refers back to an earlier session")
    r = random.Random(f"{seed}:{hkey(q)}"); q = copy.deepcopy(q); p = list(range(len(q["haystack_sessions"]))); r.shuffle(p)
    q["haystack_sessions"] = [q["haystack_sessions"][i] for i in p]; q["haystack_dates"] = [q["haystack_dates"][i] for i in p]
    if "haystack_session_ids" in q: q["haystack_session_ids"] = [q["haystack_session_ids"][i] for i in p]
    return q, {"perm": p}

# ---- untemplate
SENT = r"(^|[.!?]\s+)"   # a bare line break is not a sentence start: it may sit inside a title or a list
OPEN = re.compile(r"^(?:hey|hi|hello)[,!.]\s+", re.I)
TPL = [OPEN, re.compile(SENT + r"by the way,\s+", re.I), re.compile(SENT + r"btw,\s+", re.I), re.compile(SENT + r"just so you know,\s+", re.I)]
WORD_Q = re.compile(r"\b(exact(?:ly)?|verbatim|word(?:s|ing)?|phrase|quote|said|say|tone|mood|feel\w*|emotion\w*|attitude|style|greet\w*)\b", re.I)

def untemplate(q, seed=0, scope="all"):
    """remove confirmed filler phrases from user rounds, capitalise only at the cut. scope gold / other edits only gold or only non-gold rounds:
    a shortcut predicts recall falls when the phrase leaves gold rounds and rises when it leaves the others"""
    if WORD_Q.search(ask_text(q)) or re.search(r"style|tone", str(q.get("question_type")), re.I): raise Reject("question is about wording or tone")
    if any(re.search(r"by the way|btw|just so you know", x, re.I) for x in key_text(q)[1:5]): raise Reject("answer contains the template phrase")
    n = [0]
    def f(s):
        s, k = OPEN.subn("", s); n[0] += k; lead = k > 0
        for p in TPL[1:]:
            s, k = p.subn(lambda m: m.group(1) + "\0", s); n[0] += k
        return re.sub("\0+(.?)", lambda m: m.group(1).upper(), ("\0" if lead else "") + s, flags=re.S)
    q = copy.deepcopy(q)
    for t in turns(q):
        if t["role"] == "user" and (scope == "all" or (scope == "gold") == bool(t.get("has_answer"))): t["content"] = f(t["content"])
    if not n[0]: raise Reject("no template found")
    return q, {"cuts": n[0]}

# ---- marker swap and marker insertion (A4): matched controls for the "By the way," shortcut
BTW = re.compile(SENT + r"by the way,(\s+)", re.I); ALT = "On that note,"; MARK = "By the way,"

def markswap(q, seed=0, scope="gold"):
    """replace "By the way," by a marker of the same role, position and near length, in gold or in other user rounds only"""
    if WORD_Q.search(ask_text(q)): raise Reject("question is about wording or tone")
    if any(re.search(r"by the way", x, re.I) for x in key_text(q)[1:5]): raise Reject("answer contains the template phrase")
    q = copy.deepcopy(q); n = 0
    for t in turns(q):
        if t["role"] == "user" and (scope == "gold") == bool(t.get("has_answer")):
            t["content"], k = BTW.subn(lambda m: m.group(1) + ALT + m.group(2), t["content"]); n += k
    if not n: raise Reject("no marker in the edited rounds")
    return q, {"swaps": n}

def bm25_top_other(q, k):
    import math
    U = [(i, t) for i, t in enumerate(turns(q)) if t["role"] == "user"]
    tok = lambda x: re.findall(r"[a-z0-9]+", x.lower()); D = [tok(t["content"]) for _, t in U]; Qt = tok(q["question"])
    N = len(D); avg = sum(map(len, D)) / max(N, 1); df = collections.Counter(w for d in D for w in set(d))
    def sc(d):
        c = collections.Counter(d); return sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * c[w] * 2.2 / (c[w] + 1.2 * (.25 + .75 * len(d) / avg)) for w in Qt if w in c)
    cand = sorted((-sc(d), j) for j, d in enumerate(D) if not U[j][1].get("has_answer") and not BTW.search(U[j][1]["content"]))
    return {U[j][0] for _, j in cand[:k]}

def markins(q, seed=0, marker=MARK):
    """prefix a marker to the k non-gold user rounds that look most like the question (BM25), k = gold user rounds; run with MARK and with ALT"""
    if WORD_Q.search(ask_text(q)): raise Reject("question is about wording or tone")
    k = sum(1 for t in turns(q) if t["role"] == "user" and t.get("has_answer"))
    if not k: raise Reject("no gold user round")
    pick = bm25_top_other(q, k)
    if len(pick) < k: raise Reject("not enough non-gold rounds to insert into")
    q = copy.deepcopy(q)
    for i, t in enumerate(turns(q)):
        if i in pick: t["content"] = marker + " " + t["content"]
    return q, {"inserted": sorted(pick)}

# ---- unspeaker
PREFIX = re.compile(r"^\[Group \d+\] ([^:]{1,40}): ")
WHO_Q = re.compile(r"\b(who|whose|whom|which (?:person|member|colleague|group|team|employee)|group|team|speaker|according to|said|told|asked|boss|manager)\b|\bby [A-Z][a-z]+", re.I)

def unspeaker(q, seed=0):
    """drop group chat speaker prefixes; rejected when the answer may depend on the person or group"""
    names = {PREFIX.match(t["content"]).group(1) for t in turns(q) if PREFIX.match(t["content"])}
    if not names: raise Reject("no speaker prefix")
    a = ask_text(q)
    if WHO_Q.search(a) or re.search(r"attrib|^P_|^MA_", str(q.get("question_type"))): raise Reject("question depends on the person or group")
    parts = {w for n in names for w in n.split()}
    if any(re.search(r"\b" + re.escape(w) + r"\b", a) for w in parts): raise Reject("question names a speaker")
    q = copy.deepcopy(q)
    for t in turns(q): t["content"] = PREFIX.sub("", t["content"])
    return q, {"speakers": len(names)}

# ---- unlist
HINTS = [(re.compile(r"\s*List \d+ items?\.?\s*$"), ""), (re.compile(r"\ball \d+ (items|decisions|things|points|details|facts)\b"), r"all the \1"), (re.compile(r"\blist \d+ (items)\b", re.I), r"list the \1"),
    (re.compile(r"\bthe \d+ (?=(?:[\w-]+ ){0,4}(?:things|items|decisions|problems|points)\b)"), "the "), (re.compile(r"\bWhat \d+ (items|things)\b"), r"What \1")]

RANK = re.compile(r"\b(most|top|first|last|latest|earliest|recent|best|worst|biggest|largest|smallest|main|key|major|next|only)\b", re.I)

def unlist(q, seed=0):
    """remove only a redundant answer-count hint; a count that limits the scope stays"""
    if RANK.search(q["question"]): raise Reject("the count limits the scope")
    q = copy.deepcopy(q); s = q["question"]
    for p, b in HINTS: s = p.sub(b, s).strip()
    if s == q["question"]: raise Reject("no count hint")
    if len(s.split()) < 4: raise Reject("nothing left of the question")
    q["question"] = s
    return q, {}

FN = dict(rename=rename, shift=shift, shuffle=shuffle_sessions, untemplate=untemplate, unspeaker=unspeaker, unlist=unlist,
          untplgold=lambda q, seed=0: untemplate(q, seed, "gold"), untplother=lambda q, seed=0: untemplate(q, seed, "other"),
          swapgold=lambda q, seed=0: markswap(q, seed, "gold"), swapother=lambda q, seed=0: markswap(q, seed, "other"),
          insbtw=lambda q, seed=0: markins(q, seed, MARK), insalt=lambda q, seed=0: markins(q, seed, ALT))

def run(name, Q, seed=0):
    if seed < 0: raise ValueError("seed must not be negative: cuesum.py strips the id prefix")
    out, log = [], {"edit": name, "seed": seed, "kept": {}, "rejected": {}}; set_vocab(Q)
    for q in Q:
        try: e, info = FN[name](q, seed)
        except Reject as x: log["rejected"][q["question_id"]] = str(x); continue
        e["question_id"] = f"{name}s{seed}-{q['question_id']}"; e["orig_question_id"] = q["question_id"]; out.append(e); log["kept"][q["question_id"]] = info
    return out, log

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("qfile"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    out, log = run(a.name, json.load(open(a.qfile)), a.seed)
    json.dump(out, open(a.out, "w")); json.dump(log, open(a.out + ".log.json", "w"))
    print(a.name, "kept", len(out), "rejected", len(log["rejected"]), collections.Counter(v.split(":")[0] for v in log["rejected"].values()).most_common())
