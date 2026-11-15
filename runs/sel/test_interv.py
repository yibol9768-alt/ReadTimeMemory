"""Constructed cases for interv.py, including every case from both audits. usage: python3 test_interv.py"""
import interv as V
def Q(q="What did I decide?", u=(("user", "x", True),), dates=("2024-01-10",), qd="2024-02-01", t="direct", i="t1", ans="a"):
    return dict(question_id=i, question=q, answer=ans, haystack_sessions=[[dict(role=r, content=c, has_answer=g) for r, c, g in u]], haystack_dates=list(dates), question_date=qd, question_type=t)
def R(f, q):
    try: e = f(q)[0]; return e["question"], [t["content"] for s in e["haystack_sessions"] for t in s], e
    except V.Reject as x: return "REJECT", str(x), None
fails = []
def ck(name, ok):
    print(("ok   " if ok else "FAIL ") + name); ok or fails.append(name)
def shifted(q):
    r = R(V.shift, q); return r[1]
x = shifted(Q(u=(("user", "We met on January 1 2024 there.", True),))); ck("year without comma moves", "2024 there" not in x[0] or "January 1 " not in x[0])
for seed in range(8):
    e, d = V.shift(Q(u=(("user", "On March 1 and 2023-03-01 we met.", True),), dates=("2023-03-01",), qd="2023-04-01", i=f"s{seed}"), seed)
    t = e["haystack_sessions"][0][0]["content"]; import datetime, re
    z = datetime.date(2023, 3, 1) + datetime.timedelta(days=d["days"]); ck(f"yearless and iso agree seed {seed}", z.isoformat() in t and f"{V.MONTHS[z.month-1]} {z.day} " in t)
e, d = V.shift(Q(u=(("user", "Trip May 4-5, 2023 ok.", True),), dates=("2023-05-04",), qd="2023-06-01", i="range"), 0)
s = e["haystack_sessions"][0][0]["content"]; a = datetime.date(2023, 5, 4) + datetime.timedelta(days=d["days"]); b = datetime.date(2023, 5, 5) + datetime.timedelta(days=d["days"]); print("   ", s); ck("range keeps both ends", str(b.year) in s and str(a.day) in s)
for seed in range(40000):
    e, m = V.shift(Q(u=(("user", "Trip May 4-5, 2023 ok.", True),), dates=("2023-05-04",), qd="2023-06-01", i="range"), seed)
    if m["days"] == -124: break
s = e["haystack_sessions"][0][0]["content"]; print("   ", s); ck("range across a new year", s == "Trip December 31, 2022-January 1, 2023 ok.")
e, _ = V.shift(Q(u=(("user", "x", True),), dates=("2023-03-01T10:00:00",), qd="2023-04-01T10:00:00")); ck("T timestamp moves", e["haystack_dates"][0] != "2023-03-01T10:00:00" and e["haystack_dates"][0].endswith("T10:00:00"))
hit = 0
for seed in range(400):
    try: V.shift(Q("When is it?", u=(("user", "It is on March 1 ok.", True),), dates=("2023-03-01",), qd="2024-03-01", ans="March 1"), seed)
    except V.Reject as x: hit += "depends on the year" in str(x)
    else:
        e, m = V.shift(Q("When is it?", u=(("user", "It is on March 1 ok.", True),), dates=("2023-03-01",), qd="2024-03-01", ans="March 1"), seed)
        if e["answer"] not in e["haystack_sessions"][0][0]["content"]: ck("answer and evidence agree", False); break
else: ck("answer and evidence agree", hit > 0)
ck("fall rejected", R(V.shift, Q(u=(("user", "See you this fall.", True),)))[0] == "REJECT")
ck("(May) rejected", R(V.shift, Q(u=(("user", "Back in (May) ok, in May.", True),)))[0] == "REJECT")
ck("bad iso rejected", R(V.shift, Q(u=(("user", "Code 2024-13-45 ok.", True),)))[0] == "REJECT")
ck("weekday rejected", R(V.shift, Q(u=(("user", "Every Monday.", True),)))[0] == "REJECT")
ck("month only rejected", R(V.shift, Q(u=(("user", "Back in March.", True),)))[0] == "REJECT")
ck("age question rejected", R(V.shift, Q("How old was I?"))[0] == "REJECT")
h = tuple(("user", f"I met with Cook and Marta at noon {i}. then Cook left, and Marta too.", i == 0) for i in range(4))
r = R(V.rename, Q("What should I cook?", u=h)); ck("verb cook keeps its name out", "Cook" in r[1][0] and "cook" in r[0])
r = R(V.rename, Q("Where did I meet?", u=h)); ck("name cook renamed", "Cook" not in r[1][0])
ck("country question rejected", R(V.rename, Q("Which country?", u=h))[0] == "REJECT")
d = tuple(("user", f"The ticket is marked as Done now, and Marta said so {i}.", True) for i in range(4))
qs = [Q("Is the task finished?", u=d), Q("What did Marta do with it?", u=(("user", "it is done", True),), i="t2")]
o, l = V.run("rename", qs); ck("status word Done survives with file vocabulary", "Done" in o[0]["haystack_sessions"][0][0]["content"] and "Marta" not in o[0]["haystack_sessions"][0][0]["content"])
V.GLOBAL_LOW.clear(); V.GLOBAL_WORDS.clear()
for seed in range(40):
    h2 = tuple(("user", f"I met with Alice today {i}, and Alice left.", i == 0) for i in range(3))
    e, m = V.rename(Q("Did Alice or Dralorri come?", u=h2, i=f"c{seed}"), seed)
    if e["question"].count(m["map"]["Alice"]) > 1 or m["map"]["Alice"] == "Dralorri": ck("pseudonym avoids question words", False); break
else: ck("pseudonym avoids question words", True)
r = R(V.untemplate, Q(u=(("user", "My newsletter title is\nBy the way, life goes on.", True), ("user", "Hi, ok. By the way, I moved.", False)))); ck("line break is not a sentence start", "By the way, life" in r[1][0]); ck("sentence start cut", r[1][1] == "Ok. I moved.")
r = R(V.untemplate, Q(u=(("user", "Left by the wayside, sadly.", True), ("user", "Hi, By the way, I moved.", False)))); ck("wayside intact", r[1][0] == "Left by the wayside, sadly."); ck("two layers cut", r[1][1] == "I moved.")
r = R(V.untemplate, Q(u=(("user", "It rained.  By the way, I moved.", True),))); ck("double space cut", r[1][0] == "It rained.  I moved.")
u2 = (("user", "Ok. By the way, I moved.", True), ("user", "Hi, ok. By the way, I sold it.", False))
ck("gold scope", R(V.FN["untplgold"], Q(u=u2))[1] == ["Ok. I moved.", "Hi, ok. By the way, I sold it."]); ck("other scope", R(V.FN["untplother"], Q(u=u2))[1] == ["Ok. By the way, I moved.", "Ok. I sold it."])
u3 = (("user", "Ok. By the way, I moved to Denver.", True), ("user", "Hi. By the way, I sold the car.", False), ("user", "Where did I move? Denver is nice.", False), ("user", "Pasta recipe.", False))
r = R(V.FN["swapgold"], Q("Where did I move?", u=u3)); ck("swap gold", r[1][:2] == ["Ok. On that note, I moved to Denver.", "Hi. By the way, I sold the car."])
r = R(V.FN["swapother"], Q("Where did I move?", u=u3)); ck("swap other", r[1][:2] == ["Ok. By the way, I moved to Denver.", "Hi. On that note, I sold the car."])
r = R(V.FN["insbtw"], Q("Where did I move?", u=u3)); ck("insert into the most similar non-gold round", r[1][2] == "By the way, Where did I move? Denver is nice." and r[1][3] == "Pasta recipe.")
r = R(V.FN["insalt"], Q("Where did I move?", u=u3)); ck("insert control marker", r[1][2].startswith("On that note, "))
ck("phrase in answer rejected", R(V.untemplate, Q(u=(("user", "Hi, ok.", True),), ans="By the way, life goes on"))[0] == "REJECT")
ck("all 3 days stays", R(V.unlist, Q("Summarise all 3 days of the trip."))[0] == "REJECT")
ck("3 most recent stays", R(V.unlist, Q("What were the 3 most recent decisions?"))[0] == "REJECT")
ck("lone list rejected", R(V.unlist, Q("List 3 items."))[0] == "REJECT")
ck("top 5 stays", R(V.unlist, Q("Show the top 5 products by sales."))[0] == "REJECT")
ck("the 13 things cut", R(V.unlist, Q("What were the 13 things I brought up?"))[0] == "What were the things I brought up?")
ck("trailing list cut", R(V.unlist, Q("What did I raise about the budget? List 3 items."))[0] == "What did I raise about the budget?")
g = (("user", "[Group 1] Wei Zhang: Budget approved.", True),)
ck("who rejected", R(V.unspeaker, Q("Who approved it?", u=g))[0] == "REJECT")
ck("speaker name rejected", R(V.unspeaker, Q("What did Zhang decide?", u=g))[0] == "REJECT")
ck("by name rejected", R(V.unspeaker, Q("What was defined by Jinli for it?", u=g))[0] == "REJECT")
ck("plain unspeaker", R(V.unspeaker, Q("What was approved?", u=g))[1] == ["Budget approved."])
ck("order rejected", R(V.shuffle_sessions, Q("What did I do first?"))[0] == "REJECT")
same = (("user", "On May 4, 2023 I met Marta at noon, and Marta left, and Marta came.", True),)
a, _ = V.run("shift", [Q("What did I buy?", u=same, i="h1_a", dates=("2023-05-04",)), Q("What did I sell?", u=same, i="h1_b", dates=("2023-05-04",))])
ck("same history same edit", a[0]["haystack_sessions"] == a[1]["haystack_sessions"]); ck("id prefix gives a new history key", a[0]["question_id"].split("_")[0] == "shifts0-h1" and a[0]["orig_question_id"] == "h1_a")
r = R(V.FN["insbtw"], Q("Where?", u=(("user", "a", True), ("user", "b", True), ("user", "c", False)))); ck("too few rounds to insert rejected", r[0] == "REJECT")
print("FAILED:", fails) if fails else print("all passed")
raise SystemExit(1 if fails else 0)
