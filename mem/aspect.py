"""H36 aspect compilation (METHOD_TREE L1b; PROGRESS 2026-10-16 10:55). Built only from the lychee ledger, never
edits it, can be rebuilt at any time.
  1. group: one call per conversation (topics in chunks) merges the ledger's topic strings into aspects - one side of
     something the user works on over time (e.g. "API rate limiting").
  2. digest: one call per aspect with >= 2 records: its records in time order -> a dated digest (first mention, key
     steps in order, last state), <= DIGEST characters.
Output: runs/<split>/<arm>/aspect/asp_<ci>.json {"aspects": [{"name", "records": [idx], "first", "digest"}]}
Usage: V3_BEAM_SPLIT=1m python3 -m mem.aspect build --convs 5 [--par 8]
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
from . import llm, lychee as Y, beam as B

DIGEST, CHUNK = 400, 450
GROUP_SYS = ("You organize the topic labels of a long conversation memory. Merge the numbered topic labels into "
             "aspects: an aspect is one side of something the user worked on or talked about over time (e.g. 'API "
             "rate limiting', 'React frontend state management', 'wedding venue choice'), broad enough that labels "
             "about the same thing at different times fall into it, narrow enough that it is one thread, not a whole "
             "project. Every label goes into exactly one aspect. Output one JSON object, nothing else: "
             "{\"aspects\": [{\"name\": \"short name\", \"labels\": [label numbers]}, ...]}")
DIG_SYS = ("You compile one aspect of a long conversation memory into a short dated digest for a reader who will "
           "answer questions about the conversation. You get the aspect name and its memory records in time order. "
           f"Write at most {DIGEST} characters, plain text, no list markers: when it first came up, the key steps or "
           "changes in time order with their dates, and the last state. Keep names, numbers and decisions; keep "
           "both sides if records disagree. Use only what the records say.")


def group(recs):
    topics = list(dict.fromkeys(r["topic"].strip() or "misc" for r in recs))
    t2a, names = {}, []
    for c0 in range(0, len(topics), CHUNK):
        chunk = topics[c0:c0 + CHUNK]
        prior = "\n".join(f"- {n}" for n in names[-200:])
        body = ((f"Aspects already defined (reuse a name exactly when a label belongs to it):\n{prior}\n\n" if prior
                 else "") + "Topic labels:\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(chunk)))
        asp = []
        for effort in ("low", "medium"):            # retry with another cache key when the reply is unusable
            out = llm.call(GROUP_SYS, body, effort=effort, max_tokens=30000, tag="asp_group")
            try:
                asp = json.loads(out[out.index("{"):out.rindex("}") + 1])["aspects"]
            except Exception:
                asp = []
            got = {k for a in asp for k in (a.get("labels") or []) if isinstance(k, int)}
            if len(got) >= 0.8 * len(chunk):
                break
            print("group chunk", c0, "covered", len(got), "of", len(chunk), "with effort", effort, flush=True)
        for a in asp:
            n = str(a.get("name", "")).strip()
            if not n:
                continue
            if n not in names:
                names.append(n)
            for k in a.get("labels") or []:
                if isinstance(k, int) and 0 <= k < len(chunk):
                    t2a.setdefault(chunk[k], n)
        for t in chunk:                                     # labels the model left out keep their own aspect
            if t not in t2a:
                t2a[t] = t; names.append(t)
    by = {}
    for i, r in enumerate(recs):
        by.setdefault(t2a[r["topic"].strip() or "misc"], []).append(i)
    return by


def first_date(recs, idx):
    ds = [Y.span(recs[i]["time"])[0] for i in idx if Y.span(recs[i]["time"])]
    return min(ds) if ds else ""


def build_one(ci, out):
    path = os.path.join(out, f"asp_{ci}.json")
    if os.path.exists(path):
        return
    recs = json.load(open(os.path.join(B.OUT, "lychee", f"mem_{ci}.json")))["records"]
    by = group(recs)

    def dig(item):
        name, idx = item
        idx = sorted(idx, key=lambda i: (Y.span(recs[i]["time"]) or ("9999",))[0])
        if len(idx) == 1:
            return {"name": name, "records": idx, "first": first_date(recs, idx), "digest": recs[idx[0]]["text"]}
        body = f"Aspect: {name}\n\nRecords:\n" + "\n".join(f"- [{recs[i]['time'] or 'no date'}] {recs[i]['text']}"
                                                          for i in idx)
        d = llm.call(DIG_SYS, body[:120000], effort="low", max_tokens=4000, tag="asp_digest").strip()
        return {"name": name, "records": idx, "first": first_date(recs, idx), "digest": d[:DIGEST + 200]}

    with ThreadPoolExecutor(ARGS.par) as ex:
        asp = list(ex.map(dig, by.items()))
    json.dump({"aspects": asp}, open(path, "w"), ensure_ascii=False, indent=1)
    print("aspect", ci, "records", len(recs), "aspects", len(asp), "multi", sum(len(a["records"]) > 1 for a in asp),
          "largest", max(len(a["records"]) for a in asp), flush=True)


def build(_):
    convs = B.load()
    out = os.path.join(B.OUT, "aspect"); os.makedirs(out, exist_ok=True)
    llm.check_quota()
    for ci in B.picked(convs):
        build_one(ci, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build"])
    ap.add_argument("--convs", type=int, default=5)
    ap.add_argument("--par", type=int, default=8)
    ARGS = ap.parse_args()
    B.ARGS = ARGS
    {"build": build}[ARGS.step](None)
