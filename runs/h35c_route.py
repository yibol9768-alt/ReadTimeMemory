"""H35c: point / area router on the question text (luna, low). Writes runs/<split>/pro6/route.json {qid: AREA|POINT}."""
import argparse, collections, json, os
from concurrent.futures import ThreadPoolExecutor
from mem import llm, beam as B
SYS = ("Classify a question a user asks about their own long past conversation with an assistant. AREA: answering "
       "needs an overview across the whole conversation of one broad theme - several separate items, how something "
       "developed in order, or a summary of a whole process. POINT: answering needs specific facts (a value, a date, "
       "a decision, whether something was said, a preference to apply), even if they must be combined. "
       "Reply with one word: AREA or POINT.")
ap = argparse.ArgumentParser(); ap.add_argument("--convs", type=int); B.ARGS = ap.parse_args()
qs = B.questions(B.load())
lab = list(ThreadPoolExecutor(8).map(lambda q: "AREA" if "AREA" in llm.call(SYS, q["question"], effort="low",
                                     max_tokens=1000, tag="h35c_route").upper() else "POINT", qs))
json.dump({q["qid"]: l for q, l in zip(qs, lab)}, open(os.path.join(B.OUT, "route.json"), "w"), indent=1)
c = collections.Counter((q["ability"] in ("event_ordering", "summarization"), l) for q, l in zip(qs, lab))
print("(area ability, route):", dict(c))
print("AREA on:", collections.Counter(q["ability"] for q, l in zip(qs, lab) if l == "AREA"))
