"""World generator (10-31, PLAN_WORLD). Every round is written by one pipeline and carries one ledger fact; questions are drawn last,
so nothing about a round tells whether it will become evidence.
  plan   : project, entities, hidden categories, ledger of facts (program picks structure, model fills content, ledger written once)
  render : user turn, assistant turn, follow-ups (same prompts for all rounds)
  verify : ledger back-check and category check by two model families; only judge, rewrite at most twice, then the fact can never be gold
  build  : sessions, dates, questions (program picks evidence, model writes wording, answers come from the ledger)
  gates  : blind, sufficiency, group deletion
usage: python3 world.py plan|render|verify|build|gates --out DIR --start S --n N --rounds 240 --q27 M@P,.. --gem M@P --oss M@P [--par 32]"""
import argparse, collections, concurrent.futures as cf, datetime, json, os, random, re, sys
sys.path.insert(0, os.path.expanduser("~/data/rtm/env"))
from gen import Pool, jparse, personas, NAMES, ntok

KINDS = [("decision", .35), ("problem", .2), ("result", .2), ("number", .1), ("instruction", .08), ("preference", .07)]
ASK = ["asks how to do the next practical step", "asks for a review of what they did", "asks why something behaves the way it does", "asks for options to compare",
       "asks for help debugging or fixing something", "asks for a quick explanation of a concept they ran into", "asks what they might be missing"]
PLAN0 = """A person works for months on this project: {project}.
Return JSON with:
"entities": 14 concrete named things in this project (components, tools, people, documents, places), each 1-4 words;
"categories": 8 broad concerns that cut across the project (for example cost control, user trust, schedule risk), each 2-4 words, clearly different from each other.
Return only JSON."""
FILL = """A person works on this project: {project}.
Write one ledger fact for each line below. A fact is a specific thing that happened or was decided, with concrete details (names, numbers, versions, places). Never write calendar dates, months or years.
It must clearly matter for the given concern, but must not use the concern's words, and must not explain why it matters.
For kind=instruction write a standing rule beginning "When I ask about". For kind=preference write a lasting personal preference.
For op=update give a new value that differs from the old value of the same attribute.
Lines:
{lines}
Return a JSON list, one object per line, with "id", "attr" (short attribute name, 1-4 words), "value" (short), "statement" (one sentence, first person)."""
USER = """You are {name}. {persona}
You are working on: {project}. You are chatting with an AI assistant; here is your recent context:
{recent}
Write your next message. In it you {ask} about {entity}. While doing that, mention this as something that is simply true, in passing: {statement}
Rules: {length}. Casual, like a real chat message, {style}. Work it into your message in your own words, as part of what you are asking, instead of pasting the sentence. Keep every concrete detail of it. Do not explain why it matters. Do not use these words: {ban}.
Return only the message."""
UINST = """You are {name}, working on: {project}. Tell your AI assistant this standing rule or preference in a casual chat message, in your own words, keeping every detail: {statement}
Rules: {length}. Do not greet, do not introduce yourself, do not name the project. Do not use these words: {ban}. Return only the message."""
ASST = """You are a helpful AI assistant in a long-running chat with a user about their project. Reply to the user's latest message. Be practical and specific. Keep it to {paras} short paragraphs, no headings.
{prev}User: {msg}"""
FOLLOW = """You are chatting with an AI assistant. Its last reply was:
{reply}
Write a very short casual follow-up message ({how}). One short sentence. Return only the message."""
FHOW = ["ask which option is better", "ask for a small clarification", "ask for an example", "say you tried it and ask what next", "push back a little"]
STYLES = ["lowercase and a bit sloppy", "polite and precise", "terse", "chatty", "a little stressed"]
ENTAIL = """Message: {m}
Does this message state or clearly imply the following? {s}
Answer YES or NO only."""
CAT = """A person's message about their project:
{m}
Which of these concerns does what the person reports bear on directly? Answer with the numbers separated by commas, or NONE.
{cats}"""
QWORD = """Rewrite this as one natural question a user would ask an assistant that remembers their past chats. Keep its exact meaning, including any dates and any requested number of items. Do not add details.
{q}
Return only the question."""
QDIRECT = """A user once said this to an assistant: {s}
Write one question that this statement answers, using different words where you can, and that names enough context to be answered only by this statement. The answer is: {v}.
Return only the question."""
QINST = """A user once gave this standing rule or preference: {s}
Write a natural new request from the same user to which this rule or preference would apply. Do not mention the rule or preference.
Return only the request."""
READ = """Here are some of a user's past chats with an assistant, then a question. Answer from the chats.
{ctx}
Question (asked on {qdate}): {q}
Answer briefly but completely."""
CLOSED = """A user asks an assistant the question below, but you have no access to their past chats. Answer with your best guess, briefly and concretely.
Question: {q}"""
JUDGE = """Response:
{r}

Fact: {x}

Does the response state this fact or its main point, in any wording? Answer YES or NO."""
ORDER = """Response:
{r}

Correct order of events:
{x}

Does the response put the events it mentions in this same relative order? Answer YES or NO."""
QTEAM = """{y} said this in the team group chat: {s}
{x} is another member of the same team. Write one natural question or request from {x} that starts with "I ({x})" and that this statement answers. Name enough context (the thing or task involved) that only this statement answers it, but do not name {y}. The answer is: {v}.
Return only the question."""
JUDGEI = """Request: {q}
Response: {r}
Rule the user set earlier: {x}
Does the response follow this rule, or at least take it into account? Answer YES or NO only."""


LEGACY = {"p_group": 0, "p_long": 0, "p_dated": .5, "p_greet": None, "p_sib": 0}   # D2 (11-01): world-level knobs; LEGACY reproduces the 10-31 generator exactly
GROUP_KINDS = ("decision", "problem", "result", "number")
GUSER = """You are {name}, on a team working on: {project}. You are posting in the team's group chat; recent messages:
{recent}
Write your next message to the team. In it you mention this, in passing, as something that is simply true: {statement}
Rules: {length}. Casual, like a real work chat. Keep every concrete detail. Do not explain why it matters. Do not greet. Do not use these words: {ban}.
Return only the message."""
LONG = "eight to twelve sentences; besides the point above, talk about your day and the work in general, but state no other decision, number, result, rule or preference; keep the whole message under 1400 characters and do not put the point above in the last sentence"
def knobs(w): return {**LEGACY, **w.get("knobs", {})}
def is_legacy(w): return knobs(w) == LEGACY


def wpath(a, i): return os.path.join(a.out, f"w{i:05d}.json")
def save(a, w, i): json.dump(w, open(wpath(a, i) + ".tmp", "w")); os.replace(wpath(a, i) + ".tmp", wpath(a, i))
def fam(a, i, shift=0): return [Pool(a.q27), Pool(a.gem), Pool(a.oss)][(i + shift) % 3]
def norm(t): return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()
def ban_words(cats): return sorted({x for c in cats for x in norm(c).split() if len(x) > 3})
def leak(t, ban):
    w = set(norm(t).split()); return [b for b in ban if any(x.startswith(b[:5]) for x in w)]


# D4 (11-02): held-out worlds must not share a project or any person name with existing worlds. 30 names that are not in
# gen.NAMES (checked at load); used only with --exclude-from, which leaves every existing world bit-for-bit reproducible.
NEW_NAMES = ["Bea", "Caspar", "Dalia", "Emeka", "Freya", "Gustavo", "Hiroshi", "Ilse", "Jamal", "Katya", "Lorenzo", "Maeve", "Nadia",
             "Oskar", "Pilar", "Quentin", "Rosa", "Soren", "Tamsin", "Ugo", "Vera", "Wanjiru", "Xiomara", "Yaro", "Zainab", "Anders",
             "Bilal", "Carmen", "Dmitri", "Esme"]


def excluded(a):
    """projects and names already used by any world in the --exclude-from directories"""
    import glob as _g
    P, N = set(), set(NAMES)
    for d in a.exclude_from.split(","):
        own = os.path.abspath(d) == os.path.abspath(a.out)   # the held-out dir itself: avoid its projects, but its names are NEW_NAMES by design
        for f in _g.glob(os.path.join(d, "w*.json")):
            w = json.load(open(f)); P.add(w["project"])
            if not own: N.add(w["name"]); N |= set(w.get("team", []))
    assert not set(NEW_NAMES) & N, f"NEW_NAMES overlap with earlier worlds: {set(NEW_NAMES) & N}"
    return P, N


def plan(i, a):
    rng = random.Random(1000 + i); gp = Pool(a.q27); projects = json.load(open(a.projects))
    if a.exclude_from:
        used, _ = excluded(a); projects = [p for p in projects if p not in used]
    names = NEW_NAMES if a.exclude_from else NAMES
    project = rng.choice(projects)
    for _ in range(4):
        x = jparse(gp.chat(PLAN0.format(project=project), max_tokens=900, temperature=0.8)[0])
        if isinstance(x, dict) and len(x.get("entities", [])) >= 10 and len(x.get("categories", [])) >= 6: break
    else: return None
    ents, cats = [str(e) for e in x["entities"]], [str(c) for c in x["categories"]][:8]; facts, slots = [], {}
    N = a.rounds; K = {**LEGACY, **a.knobs}; krng = random.Random(f"knob{i}"); team = []
    if K["p_group"]:
        team = krng.sample([n for n in names], 5)
    for b in range(0, N, 20):
        be, bc = rng.sample(ents, min(5, len(ents))), rng.sample(cats, min(3, len(cats))); lines, batch = [], []
        for k in range(b, min(b + 20, N)):
            kind = rng.choices([x_[0] for x_ in KINDS], [x_[1] for x_ in KINDS])[0]; ent = rng.choice(be); cat = rng.choice(bc)
            f = {"id": f"f{k}", "pos": k, "entity": ent, "cat": cat, "kind": kind, "op": "add", "prev": None}
            old = [fid for fid in slots.get(ent, []) if fid["kind"] in ("decision", "number")]
            if kind in ("decision", "number") and old and rng.random() < .25:
                p = rng.choice(old); f.update(op="update", prev=p["id"], cat=p["cat"]); line = f"{f['id']} | entity={ent} | concern={f['cat']} | kind={kind} | op=update, attribute {p['attr']}, old value {p['value']}"
            else:
                line = f"{f['id']} | entity={ent} | concern={cat} | kind={kind} | op=add"
                sibs = [x for x in facts if not x.get("dead") and x["attr"] and x["kind"] == kind and x["kind"] in GROUP_KINDS]
                if K["p_sib"] and sibs and krng.random() < K["p_sib"]:
                    p = krng.choice(sibs); f.update(entity=p["entity"], cat=p["cat"], sib_of=p["id"])
                    line = f"{f['id']} | entity={p['entity']} | concern={p['cat']} | kind={kind} | op=add, about a different attribute than {p['attr']}, but easy to confuse with this earlier fact: {p['statement']}"
            if K["p_group"] and kind in GROUP_KINDS and krng.random() < K["p_group"]: f["speaker"] = krng.choice(team)
            if K["p_long"] and krng.random() < K["p_long"]: f["long"] = True
            lines.append(line); batch.append(f)
        for _ in range(4):
            y = jparse(gp.chat(FILL.format(project=project, lines="\n".join(lines)), max_tokens=2600, temperature=0.8)[0])
            got = {str(o.get("id")): o for o in y if isinstance(o, dict)} if isinstance(y, list) else {}
            if sum(f["id"] in got for f in batch) >= .9 * len(batch): break
        byid = {f["id"]: f for f in facts}
        for f in batch:
            o = got.get(f["id"])
            if not o or not o.get("statement") or not o.get("value"): f.update(attr="", value="", statement="", dead=True); facts.append(f); continue
            f.update(attr=str(o.get("attr", "")), value=str(o["value"]), statement=str(o["statement"]))
            if f["op"] == "update":
                if norm(f["value"]) == norm(byid[f["prev"]]["value"]): f["dead"] = True
                else: f["attr"] = byid[f["prev"]]["attr"]
            if f["kind"] == "instruction" and "when i ask about" not in f["statement"].lower(): f["dead"] = True
            facts.append(f)
            if f.get("sib_of") and norm(f["attr"]) == norm(byid.get(f["sib_of"], {}).get("attr", "")): f["dead"] = True   # a sibling must differ in attribute, else it would join the update chain
            if not f.get("dead"): slots.setdefault(f["entity"], []).append(f)
    name = rng.choice([n for n in names if n not in team]) if a.exclude_from else rng.choice(NAMES); persona = str(rng.choice(personas(os.path.join(a.bg, "persona", "persona.jsonl"), 500, rng)))[:300]
    w = {"world": i, "project": project, "name": name, "persona": persona, "entities": ents, "cats": cats, "style": rng.choice(STYLES), "facts": facts, "tilt": ["early", "uniform", "late"][i % 3]}
    if K != LEGACY: w["knobs"] = K; w["team"] = team
    return w


GREET = re.compile(r"^\s*(hey|hi|hello|hiya)( there)?[,!.\s]+", re.I)
def round_user(w, f, rng, recent, pool):
    t = _round_user(w, f, rng, recent, pool)
    if not is_legacy(w) and knobs(w)["p_greet"] is not None and random.Random(f"greet{w['world']}:{f['id']}").random() >= knobs(w)["p_greet"]:
        t = GREET.sub("", t); t = t[:1].upper() + t[1:]   # models ignore "do not greet"; strip deterministically
    return t


def _round_user(w, f, rng, recent, pool):
    ban = ", ".join(ban_words([f["cat"]])) or "none"; length = rng.choice(["one sentence", "two sentences", "two sentences", "three short sentences"])
    if not is_legacy(w):
        if f.get("long"): length = LONG
        if f.get("speaker"): return pool.chat(GUSER.format(name=f["speaker"], project=w["project"], recent=recent or "(start of chat)", statement=f["statement"], length=length, ban=ban), max_tokens=700 if f.get("long") else 220, temperature=0.9)[0]
        g = knobs(w)["p_greet"]
        if g is not None and random.Random(f"greet{w['world']}:{f['id']}").random() >= g: length += ". Do not open with a greeting"
    if f["kind"] in ("instruction", "preference"): p = UINST.format(name=w["name"], project=w["project"], statement=f["statement"], length=length, ban=ban)
    else: p = USER.format(name=w["name"], persona=w["persona"], project=w["project"], recent=recent or "(start of chat)", ask=rng.choice(ASK), entity=f["entity"], statement=f["statement"], length=length, style=w["style"], ban=ban)
    return pool.chat(p, max_tokens=700 if f.get("long") and not is_legacy(w) else 220, temperature=0.9)[0]


def render(w, i, a):
    up, ap_ = fam(a, i), fam(a, i, 1); rng = random.Random(2000 + i); live = [f for f in w["facts"] if not f.get("dead")]
    seq = live; out = []; seen = []
    for f in seq:
        seen = [x for x in seen if (x["entity"], x["attr"]) != (f["entity"], f["attr"])]   # a superseded value is never restated
        out.append((f, "ev"))
        if seen and rng.random() < .10: out.append((rng.choice(seen[-40:]), "repeat"))
        seen.append(f)
    def job(x):
        k, (f, kind) = x; r = random.Random(3000 * i + k); recent = "\n".join(s["statement"][:120] for s, _ in out[max(0, k - 3):k])
        u = round_user(w, f, r, recent, up)
        if f.get("speaker") and not is_legacy(w): return [{"fid": f["id"], "kind": kind, "user": u, "assistant": None, "ok": kind == "ev", "speaker": f["speaker"]}]
        ar = ap_.chat(ASST.format(paras=r.choice([2, 3, 3, 4]) if not (f.get("long") and not is_legacy(w)) else r.choice([5, 6, 7]), prev="", msg=u), max_tokens=520 if not f.get("long") else 900, temperature=0.7)[0]
        rs = [{"fid": f["id"], "kind": kind, "user": u, "assistant": ar, "ok": kind == "ev"}]
        if r.random() < a.fu:
            fu = up.chat(FOLLOW.format(reply=ar[:900], how=r.choice(FHOW)), max_tokens=60, temperature=0.9)[0]
            rs.append({"fid": f["id"], "kind": "fu", "user": fu, "assistant": ap_.chat(ASST.format(paras=r.choice([1, 2, 2]), prev=f"User: {u[:500]}\nAssistant: {ar[:500]}\n", msg=fu), max_tokens=400, temperature=0.7)[0], "ok": False})
        return rs
    with cf.ThreadPoolExecutor(a.par) as ex: w["rounds"] = [r for rs in ex.map(job, enumerate(out)) for r in rs]
    return w


def verify(w, i, a):
    up = fam(a, i); F = {f["id"]: f for f in w["facts"]}; vp = [Pool(a.q27), Pool(a.gem), Pool(a.oss)]; ctxt = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(w["cats"]))
    def job(x):
        k, r = x
        if r["kind"] != "ev": return r
        f = F[r["fid"]]; pools = [vp[(i + 1) % 3], vp[(i + 2) % 3]]; rr = random.Random(5000 * i + k)
        for t in range(3):
            ent = all("YES" in p.chat(ENTAIL.format(m=r["user"], s=f["statement"]), max_tokens=8, temperature=0)[0].upper() for p in pools)
            bad = leak(r["user"], ban_words([f["cat"]]))
            if ent and not bad: break
            if t < 2:
                r["user"] = round_user(w, f, rr, "", up); r["rewrites"] = t + 1
                if r.get("speaker"): continue
                r["assistant"] = fam(a, i, 1).chat(ASST.format(paras=rr.choice([2, 3, 3, 4]), prev="", msg=r["user"]), max_tokens=520, temperature=0.7)[0]
        r["ok"] = bool(ent and not bad)
        votes = []
        for p in pools:
            c = p.chat(CAT.format(m=r["user"], cats=ctxt), max_tokens=24, temperature=0)[0]
            votes.append({w["cats"][int(x) - 1] for x in re.findall(r"\d+", c) if 0 < int(x) <= len(w["cats"])})
        r["cats"] = sorted(votes[0] & votes[1]); r["cats_any"] = sorted(votes[0] | votes[1])
        return r
    with cf.ThreadPoolExecutor(a.par) as ex: w["rounds"] = list(ex.map(job, enumerate(w["rounds"])))
    w["verified"] = True; return w


def w_long(w, r): return bool({f["id"]: f for f in w["facts"]}.get(r["fid"], {}).get("long"))


def timeline(w, rng):
    start = datetime.date(2024, 1, 8) + datetime.timedelta(days=rng.randint(0, 300)); span = rng.randint(120, 240); R = w["rounds"]; cuts = [0]
    while cuts[-1] < len(R): cuts.append(cuts[-1] + rng.randint(6, 30))
    if not is_legacy(w):   # one carrier per session: cut wherever the speaker kind changes
        car = lambda k: bool(R[k].get("speaker")); extra = [k for k in range(1, len(R)) if car(k) != car(k - 1)]
        cuts = sorted(set([c for c in cuts if c < len(R)] + extra)) + [len(R)]
        cuts = [c for n_, c in enumerate(cuts) if n_ == 0 or c > cuts[n_ - 1]]
    if len(cuts) - 1 > span - 1: span = len(cuts) + 1   # 11-02 fix: more sessions than days left the last rounds without a date (KeyError in questions); only reached when it would have failed, so existing worlds are unchanged
    days = sorted(rng.sample(range(span), min(len(cuts) - 1, span - 1))); hs, dates, tid = [], [], 0; day = {}
    for s, d in enumerate(days):
        ss = []
        for k in range(cuts[s], min(cuts[s + 1], len(R))):
            r = R[k]; day[k] = d
            if r.get("speaker") and not is_legacy(w):
                g = 1 + w["team"].index(r["speaker"]) % 2
                ss.append({"role": "user", "content": f"[Group {g}] {r['speaker']}: {r['user']}", "id": tid, "stage": f"{r['fid']}#{k}", "kind": r["kind"], "carrier": "group"}); tid += 1; continue
            c = {} if is_legacy(w) else {"carrier": "long" if w_long(w, r) else "dialog"}
            ss.append({"role": "user", "content": r["user"], "id": tid, "stage": f"{r['fid']}#{k}", "kind": r["kind"], **c}); ss.append({"role": "assistant", "content": r["assistant"], "id": tid + 1, "stage": f"{r['fid']}#{k}", "kind": r["kind"], **c}); tid += 2
        hs.append(ss); dates.append((start + datetime.timedelta(days=d)).strftime("%B-%d-%Y"))
    return hs, dates, (start + datetime.timedelta(days=span + 5)).strftime("%B-%d-%Y"), day, start


def questions(w, a, rng):
    gp = Pool(a.q27); R = w["rounds"]; hs, dates, qdate, day, start = timeline(w, rng); F = {f["id"]: f for f in w["facts"]}
    first = {}
    for k, r in enumerate(R):
        if r["kind"] == "ev" and r["ok"]: first[r["fid"]] = k
    mon = lambda d: (start + datetime.timedelta(days=d)).strftime("%B %d")
    def word(q):
        for _ in range(3):
            t = gp.chat(QWORD.format(q=q), max_tokens=120, temperature=0.8)[0]
            if t and all(x in t for x in re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d+)\b", q)): return t
        return q
    toks = [set(norm(r["user"] + " " + (r["assistant"] or "")).split()) for r in R]
    team = bool(knobs(w)["p_group"]) and not is_legacy(w)
    if not is_legacy(w): toks = [t | (set(norm(R[j - 1]["user"]).split()) if j else set()) for j, t in enumerate(toks)]   # fmt.doc_str shows the previous user message too
    I = (lambda s: s) if not team else (lambda s: s.replace("the things I brought up", "the things we discussed").replace("problems I ran into", "problems that came up").replace("decisions I made", "decisions that were made").replace("results I got", "results we got").replace("What did I bring up about", "What came up about"))
    say = lambda k: (f"{R[k]['speaker']}: " if R[k].get("speaker") else "")
    def carriers(ks):
        """rounds that also carry a gold value (cross-fact echoes in user or assistant text): never negatives"""
        out = set()
        for k in ks:
            v = set(norm(F[R[k]["fid"]]["value"]).split())
            e = set(norm(F[R[k]["fid"]]["entity"]).split())   # value alone is not enough: "5%" matched every round containing a 5 (11-01)
            if len(v) >= 2 or any(ch.isdigit() for ch in "".join(v)) or not is_legacy(w): out |= {j for j, t in enumerate(toks) if j not in ks and v <= t and e <= t}
        return out
    def mk(qt, q, ks, ans, tilt=None, sup=(), rub=None):
        sup = sorted(set(sup) | carriers(set(ks)))
        return {"qtype": qt, "question": q, "keys": [f"{R[k]['fid']}#{k}" for k in ks], "answer": ans, "rubric": [f"LLM response should contain: {say(k)}{F[R[k]['fid']]['statement']}" for k in ks] if qt in ("aggregation", "ordering") else [f"LLM response should contain: {rub or ans}"], "support": [f"{R[k]['fid']}#{k}" for k in sup], "tilt": tilt or "none"}
    slot = lambda f: (f["entity"], norm(f["attr"]))
    last = {}
    for f in w["facts"]:
        if not f.get("dead"): last[slot(f)] = f["id"]
    stale = {f["id"] for f in w["facts"] if not f.get("dead") and last[slot(f)] != f["id"]}
    jobs = []; ncat = {c: sorted(k for fid, k in first.items() if c in R[k].get("cats", [])) for c in w["cats"]}; L = max(len(R), 1)
    for c, ks in ncat.items():
        for K in (3, 5, 8, 12):
            if len(ks) < K: continue
            sts = list(range(len(ks) - K + 1)); tilt = rng.choice(["early", "uniform", "late"]); centre = {"early": 0.2, "uniform": 0.5, "late": 0.8}[tilt]
            s = min(sts, key=lambda j: abs(ks[j + K // 2] / L - centre) + rng.random() * .05); g = ks[s:s + K]
            d0, d1 = day[g[0]], day[g[-1]]; g = [k for k in ks if d0 <= day[k] <= d1]
            amb = [k for k, r in enumerate(R) if d0 <= day[k] <= d1 and k not in g and r["kind"] == "ev" and (c in r.get("cats_any", []) or F[r["fid"]]["cat"] == c)]
            when = f"on {mon(d0)}" if d0 == d1 else f"between {mon(d0)} and {mon(d1)}"
            if rng.random() < .5: jobs.append(("aggregation", I(f"What were all the things I brought up {when} that had to do with {c.lower()}? List {len(g)} items."), g, amb, tilt))
            else: jobs.append(("ordering", f"In chronological order, what happened with {c.lower()} {when}? List {len(g)} items.", g, amb, tilt))
    # half of the multi-evidence questions carry no date window: scope is a kind of event or an entity inside the hidden category
    NOUN = {"problem": "problems I ran into", "decision": "decisions I made", "result": "results I got"}
    dated = jobs; jobs = []
    undated = []
    for c, ks in ncat.items():
        for kind, noun in NOUN.items():
            g = [k for k in ks if F[R[k]["fid"]]["kind"] == kind]
            if 3 <= len(g) <= 14:
                amb = [k for k, r in enumerate(R) if k not in g and r["kind"] == "ev" and F[r["fid"]]["kind"] == kind and (c in r.get("cats_any", []) or F[r["fid"]]["cat"] == c)]
                undated.append(("aggregation", I(f"What were all the {noun} that had to do with {c.lower()}? List {len(g)} items."), g, amb) if rng.random() < .5 else ("ordering", I(f"In chronological order, what were the {noun} that had to do with {c.lower()}? List {len(g)} items."), g, amb))
        for ent in w["entities"]:
            g = [k for k in ks if F[R[k]["fid"]]["entity"] == ent]
            if 3 <= len(g) <= 14:
                amb = [k for k, r in enumerate(R) if k not in g and r["kind"] == "ev" and F[r["fid"]]["entity"] == ent and (c in r.get("cats_any", []) or F[r["fid"]]["cat"] == c)]
                undated.append(("aggregation", I(f"What did I bring up about {ent} that had to do with {c.lower()}? List {len(g)} items."), g, amb))
    rng.shuffle(dated); rng.shuffle(undated); n = min(len(dated), len(undated), 24); nd = nu = n
    if not is_legacy(w):
        T = min(len(dated) + len(undated), 48); nd = min(len(dated), round(knobs(w)["p_dated"] * T)); nu = min(len(undated), T - nd)
    jobs = [(*j, "date") for j in dated[:nd]] + [(*j, "none", "none") for j in undated[:nu]]
    qs = []
    def fin(j):
        if len(j) == 6: qt, q, g, amb, tilt, scope = j
        else: qt, q, g, amb, scope, tilt = j
        o = mk(qt, word(q), g, "; ".join(F[R[k]["fid"]]["statement"] for k in g), tilt=tilt, sup=amb); o["scope"] = scope; return o
    with cf.ThreadPoolExecutor(8) as ex: qs += list(ex.map(fin, jobs))
    # update chains: the latest live value of a slot; skipped when that latest fact failed verification
    for key, fid in list(last.items()):
        f = F[fid]
        if f["op"] != "update" or fid not in first: continue
        olds = [first[x["id"]] for x in w["facts"] if x["id"] in first and slot(x) == key and x["id"] != fid]
        qs.append(mk("knowledge_update", word(f"What is the current {f['attr']} of {f['entity']}?"), [first[fid]], f["value"], sup=olds))
    def single(fid):
        k = first[fid]; f = F[fid]
        if f["kind"] in ("instruction", "preference"): return mk("instruction", gp.chat(QINST.format(s=f["statement"]), max_tokens=120, temperature=0.8)[0], [k], f["statement"])
        if R[k].get("speaker") and not is_legacy(w):
            return mk("direct", gp.chat(QDIRECT.replace("A user once said this to an assistant", f"{R[k]['speaker']} once said this in the team group chat").replace("Write one question", f"Write one question a teammate would ask about what {R[k]['speaker']} said").format(s=f["statement"], v=f["value"]), max_tokens=120, temperature=0.8)[0], [k], f["value"], rub=f"{f['attr']}: {f['value']}")
        return mk("direct", gp.chat(QDIRECT.format(s=f["statement"], v=f["value"]), max_tokens=120, temperature=0.8)[0], [k], f["value"], rub=f"{f['attr']}: {f['value']}")
    has_value = lambda fid: set(norm(F[fid]["value"]).split()) <= set(norm(R[first[fid]]["user"]).split())
    # instruction/preference facts are only ever stated by the asker; group facts never carry them (plan restricts kinds)
    ins = [fid for fid in first if F[fid]["kind"] in ("instruction", "preference") and fid not in stale]
    plainf = [fid for fid in first if F[fid]["kind"] not in ("instruction", "preference") and fid not in stale and has_value(fid)]
    pool_ = rng.sample(ins, min(10, len(ins))) + rng.sample(plainf, min(20, len(plainf)))
    with cf.ThreadPoolExecutor(8) as ex: qs += [q for q in ex.map(single, pool_) if q["question"]]
    if team and getattr(a, "team_q", 0): qs += team_questions(w, a, rng, R, F, first, stale, has_value, mk, word)
    return qs, hs, dates, qdate


def team_questions(w, a, rng, R, F, first, stale, has_value, mk, word):
    """11-02 (stage-1 failure study, change 1): group-chat questions asked by a team member who names themself ("I (X) ...")
    while the evidence was said by OTHER members. X also posts in the same chat, so X's own messages are natural distractors
    (mkdata --spk-neg). Up to a.team_q single-evidence and a.team_q // 2 multi-evidence questions per group world.
    Never asked by the world's main user (who is not in the group chat). Every question carries "asker": X."""
    gp = Pool(a.q27); spoke = collections.Counter(r.get("speaker") for r in R if r.get("speaker"))
    team = [x for x in w["team"] if spoke[x]]
    gfacts = [fid for fid in first if R[first[fid]].get("speaker") and fid not in stale and has_value(fid)]
    out = []
    def one(fid):
        k = first[fid]; f = F[fid]; y = R[k]["speaker"]; xs = [x for x in team if x != y]
        if not xs: return None
        x = random.Random(f"teamq{w['world']}:{fid}").choice(xs)
        for _ in range(3):
            t = gp.chat(QTEAM.format(y=y, x=x, s=f["statement"], v=f["value"]), max_tokens=140, temperature=0.8)[0].strip()
            if f"I ({x})" in t and y.lower() not in t.lower(): o = mk("direct", t, [k], f["value"], rub=f"{f['attr']}: {f['value']}"); o["asker"] = x; o["scope"] = "team"; return o
        return None
    pick = rng.sample(gfacts, min(a.team_q, len(gfacts)))
    with cf.ThreadPoolExecutor(8) as ex: out += [o for o in ex.map(one, pick) if o]
    # multi-evidence: everything the OTHER members reported about one entity
    jobs = []
    for ent in w["entities"]:
        g = sorted(first[fid] for fid in gfacts if F[fid]["entity"] == ent)
        spk = {R[k]["speaker"] for k in g}; xs = [x for x in team if x not in spk]
        if 3 <= len(g) <= 8 and xs:
            x = random.Random(f"teamagg{w['world']}:{ent}").choice(xs)
            amb = [k for k, r in enumerate(R) if k not in g and r["kind"] == "ev" and F[r["fid"]]["entity"] == ent]
            jobs.append((x, ent, g, amb))
    rng.shuffle(jobs)
    def agg(j):
        x, ent, g, amb = j; raw = f"I ({x}) am taking over the work on {ent}. What did the others on the team report about it in the group chat? List {len(g)} items."
        t = word(raw); t = t if f"I ({x})" in t else raw
        o = mk("aggregation", t, g, "; ".join(F[R[k]["fid"]]["statement"] for k in g), sup=amb); o["asker"] = x; o["scope"] = "team"; return o
    with cf.ThreadPoolExecutor(8) as ex: out += list(ex.map(agg, jobs[: max(1, a.team_q // 2)]))
    return out


def build(w, i, a):
    rng = random.Random(4000 + i); qs, hs, dates, qdate = questions(w, a, rng); out = []; tok = sum(ntok(t["content"]) for s in hs for t in s)
    F_ = {f["id"]: f for f in w["facts"]}; sibbed = {f["sib_of"] for f in w["facts"] if f.get("sib_of") and not f.get("dead")}
    for n, q in enumerate(qs):
        keys = set(q["keys"]); H = [[{**t, "has_answer": t["stage"] in keys and t["role"] == "user"} for t in s] for s in hs]
        out.append({"question_id": f"w{i}_{q['qtype']}_{q.get('scope', 'single')}_{n}", "question_type": q["qtype"], "split": "train", "question": q["question"], "answer": q["answer"], "rubric": q["rubric"], "gold_stages": q["keys"], "support_stages": q["support"], "tilt": q["tilt"], "scope": q.get("scope", "single"), **({"asker": q["asker"]} if q.get("asker") else {}), "question_date": qdate, "haystack_sessions": H, "haystack_dates": dates, "hist_tok": tok,
                    **({} if is_legacy(w) else {"carriers": sorted({t.get("carrier") for s in H for t in s if t["stage"] in keys and t["role"] == "user"}), "has_sib": any(F_[x.split("#")[0]].get("sib_of") or x.split("#")[0] in sibbed for x in q["keys"])})})
    json.dump(out, open(os.path.join(a.out, f"q{i:05d}.json"), "w")); return len(out), tok


def gates(i, a):
    Q = json.load(open(os.path.join(a.out, f"q{i:05d}.json"))); rp = Pool(a.q27); sibs = {}
    try:
        for f in json.load(open(wpath(a, i)))["facts"]:
            if f.get("sib_of") and not f.get("dead"): sibs.setdefault(f["sib_of"], set()).add(f["id"]); sibs.setdefault(f["id"], set()).add(f["sib_of"])
    except (OSError, KeyError): pass
    def rounds(q):
        out = []
        for s, d in zip(q["haystack_sessions"], q["haystack_dates"]):
            j = 0
            while j < len(s):
                n = 2 if j + 1 < len(s) and s[j]["role"] == "user" and s[j + 1]["role"] == "assistant" else 1; out.append((d, s[j:j + n])); j += n
        return out
    def fmt(rs): return "\n".join(f"({d}) User: {t[0]['content']}" + (f"\nAssistant: {t[1]['content'][:600]}" if len(t) > 1 else "") for d, t in rs)
    def score(q, r):
        J = JUDGEI if q["question_type"] == "instruction" else JUDGE
        hit = [("YES" in rp.chat(J.format(q=q["question"], r=r[:3000], x=x.replace("LLM response should contain: ", "")), max_tokens=8, temperature=0)[0].upper()) for x in q["rubric"]]
        return sum(hit) / len(hit)
    def one(q):
        rng = random.Random(q["question_id"]); R = rounds(q); keys = set(q["gold_stages"]); gold = [x for x in R if x[1][0]["stage"] in keys]; fact = lambda x: x[1][0]["stage"].split("#")[0]; fids = {fact(x) for x in gold}; sup = set(q.get("support_stages", [])); near = {y for f_ in fids for y in sibs.get(f_, ())}
        other = [x for x in R if fact(x) not in fids and x[1][0]["stage"] not in sup and fact(x) not in near]   # filler never carries the answer, nor a sibling under test
        filler = rng.sample(other, min(5, len(other))); ask = lambda c: rp.chat(READ.format(ctx=fmt(c), qdate=q["question_date"], q=q["question"]), max_tokens=1200, temperature=0)[0]
        o = {"qid": q["question_id"], "type": q["question_type"], "k": len(gold), "tilt": q["tilt"]}
        o["blind"] = score(q, rp.chat(CLOSED.format(q=q["question"]), max_tokens=300, temperature=0)[0]); ans = ask(gold + filler); o["suff"] = score(q, ans)
        if q["question_type"] == "ordering":
            o["order"] = "YES" in rp.chat(ORDER.format(r=ans[:3000], x="\n".join(f"{n + 1}. {x.replace('LLM response should contain: ', '')}" for n, x in enumerate(q["rubric"]))), max_tokens=8, temperature=0)[0].upper()
        sib = [x for x in R if x[1][0]["stage"].split("#")[0] in sibs.get(next(iter(fids)), ())] if len(fids) == 1 else []
        if sib:   # sibling control: a near fact alone must not answer the question
            o["sib_only"] = score(q, ask(sib[:1] + filler)); o["gold_only"] = o["suff"]
        if len(gold) > 1:
            drop = rng.sample(range(len(gold)), max(1, len(gold) // 3)); o["del"] = score(q, ask([x for n, x in enumerate(gold) if n not in drop] + filler)); o["ndrop"] = len(drop)
        return o
    with cf.ThreadPoolExecutor(a.par // 2) as ex: rows = list(ex.map(one, Q))
    json.dump(rows, open(os.path.join(a.out, f"g{i:05d}.json"), "w")); return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("--out", required=True); ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=240); ap.add_argument("--q27", required=True); ap.add_argument("--gem", required=True); ap.add_argument("--oss", required=True); ap.add_argument("--par", type=int, default=32)
    ap.add_argument("--knobs", default="{}", help='D2 knobs as JSON, e.g. {"p_group": .4}; only read by plan, then stored in the world');
    ap.add_argument("--exclude-from", dest="exclude_from", default="", help="plan only (D4, 11-02): comma list of world dirs; new worlds avoid their projects and use NEW_NAMES (never gen.NAMES); empty = old behaviour")
    ap.add_argument("--team-q", dest="team_q", type=int, default=0, help="build only (11-02): per group world, this many 'I (member) ...' single-evidence questions answered by other members, plus half as many multi-evidence ones; 0 = old behaviour")
    ap.add_argument("--fu", type=float, default=.15); ap.add_argument("--projects", default=os.path.expanduser("~/data/rtm/env/projects.json")); ap.add_argument("--bg", default=os.path.expanduser("~/data/rtm/env/bg"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); a.knobs = json.loads(a.knobs)
    for i in range(a.start, a.start + a.n):
        if a.step == "plan":
            w = plan(i, a); assert w, f"plan {i}"; save(a, w, i); print("plan", i, "live facts", sum(not f.get("dead") for f in w["facts"]), "of", len(w["facts"]), flush=True); continue
        if a.step == "gates":
            r = gates(i, a); import statistics as st
            for t in sorted({x["type"] for x in r}):
                v = [x for x in r if x["type"] == t]; d = [x["del"] for x in v if "del" in x]; so = [x["sib_only"] for x in v if "sib_only" in x]
                if so: print("gates", i, t, "sibling control n", len(so), "sib_only", round(st.mean(so), 2), "gold_only", round(st.mean(x["gold_only"] for x in v if "sib_only" in x), 2), flush=True)
                print("gates", i, t, "n", len(v), "k", round(st.mean(x["k"] for x in v), 1), "blind", round(st.mean(x["blind"] for x in v), 2), "suff", round(st.mean(x["suff"] for x in v), 2), "del", round(st.mean(d), 2) if d else None, "del<suff", round(st.mean(x["del"] < x["suff"] for x in v if "del" in x), 2) if d else None, flush=True)
            continue
        w = json.load(open(wpath(a, i)))
        if a.step == "render": save(a, render(w, i, a), i); print("render", i, len(w["rounds"]), collections.Counter(r["kind"] for r in w["rounds"]), flush=True)
        elif a.step == "verify":
            w = verify(w, i, a); save(a, w, i); ev = [r for r in w["rounds"] if r["kind"] == "ev"]; print("verify", i, "ok", sum(r["ok"] for r in ev), "with cats", sum(bool(r.get("cats")) for r in ev), "per cat", dict(collections.Counter(c for r in ev for c in r.get("cats", []))), "of", len(ev), "rewrites", sum(bool(r.get("rewrites")) for r in ev), flush=True)
        elif a.step == "build": print("build", i, build(w, i, a), flush=True)


if __name__ == "__main__": main()
