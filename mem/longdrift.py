"""Drift at long horizons: BEAM 500K / 1M conversations (one user with an assistant, ~800-1700 turns).

Each BEAM conversation comes in 10 dated chunks; every chunk is cut into sessions of SESSION turns, so a 1M
conversation gives ~85 sessions (LoCoMo has 19-32). Assistant turns are cut to 1500 characters when shown.
Same two memories as mem/drift.py, same prompts and budget:
  roll     rewritten from itself after every session
  rebuild  rebuilt from the append-only ledger of per-session records at each checkpoint
Measurement uses our ledger probes instead of BEAM's questions (only a few BEAM questions per conversation have
evidence early enough): FACTS records from the first third of the sessions get one question each, asked of both
memories at checkpoints 1/3, 2/3 and the end (answers graded against the record).

Usage: python3 -m mem.longdrift fetch|build|report [--split 1M] [--convs 0,1,2,3,4] [--budget 8000] [--par 2]
"""
import argparse, collections, json, os, random, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, drift as D, rehearse as H

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "beam")
SESSION, CUT, FACTS = 20, 1500, 60


def fetch(_):
    for ci in ARGS.convs:
        path = os.path.join(DATA, f"{ARGS.split}_{ci}.json")
        if not os.path.exists(path):
            url = (f"https://datasets-server.huggingface.co/rows?dataset=Mohammadta/BEAM&config=default"
                   f"&split={ARGS.split}&offset={ci}&length=1")
            row = json.loads(urllib.request.urlopen(url, timeout=300).read())["rows"][0]["row"]
            json.dump({"conversation_id": row["conversation_id"], "chat": row["chat"]}, open(path, "w"))
            print("fetched", ARGS.split, ci, flush=True)


def sessions(ci):
    chat = json.load(open(os.path.join(DATA, f"{ARGS.split}_{ci}.json")))["chat"]
    out = []
    for chunk in chat:
        date = next((t["time_anchor"] for t in chunk if t.get("time_anchor")), "").replace("-", " ")
        turns = [{"id": f"T{t['id']}", "speaker": "User" if t["role"] == "user" else "Assistant",
                  "text": (lambda x: x[:CUT] + (" [...]" if len(x) > CUT else ""))(
                      re.sub(r"\s*->->\s*\d+,\d+\s*$", "", t["content"]).strip())} for t in chunk]
        for i in range(0, len(turns), SESSION):
            out.append({"n": len(out) + 1, "date": date, "turns": turns[i:i + SESSION]})
    return out


def build_conv(ci):
    path = os.path.join(OUT, f"mem_{ci}.json")
    if os.path.exists(path):
        return
    ss = sessions(ci)
    cps = D.checkpoints(len(ss))

    def records(s):
        body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(L.show_turn(t, s["date"]) for t in s["turns"])
        return L.parse_records(llm.call(L.REC_SYS, body, tag="ld_rec"), {t["id"] for t in s["turns"]})

    def roll():
        mem, out = "", {}
        for i, s in enumerate(ss, 1):
            mem = L.within_budget(f"Your current memory of the earlier sessions:\n\n{mem or '(empty)'}\n\n"
                                  f"The next session:\n\n{L.show_session(s)}\n\n"
                                  f"Write the updated memory of everything so far (at most {L.BUDGET} characters).",
                                  "ld_roll")
            if i in cps:
                out[str(i)] = mem
        return out
    with ThreadPoolExecutor(2) as ex:
        roll_f = ex.submit(roll)
        recs = list(ex.map(records, ss))
        rebuild = {}
        for k in cps:
            ledger = [r["text"] for rs in recs[:k] for r in rs]
            rebuild[str(k)] = L.within_budget(
                "Memory records extracted so far from the conversation (in time order; each is self-contained):\n\n"
                + "\n".join("- " + t for t in ledger)
                + f"\n\nWrite the memory of everything so far (at most {L.BUDGET} characters).", "ld_rebuild")
        m = {"checkpoints": cps, "n_sessions": len(ss), "roll": roll_f.result(), "rebuild": rebuild,
             "records": recs}
    # probes: FACTS records from the first third, one question each, asked of both memories at every checkpoint
    early = [r for rs in recs[:cps[0]] for r in rs]
    facts = random.Random(ci).sample(early, min(FACTS, len(early)))
    qs = H.probes_for(facts)
    probe = {}
    for arm in ("roll", "rebuild"):
        for k in cps:
            bad = set()
            for b in range(0, len(facts), 20):
                bad |= {b + j for j in H.ask(m[arm][str(k)], facts[b:b + 20], qs[b:b + 20])}
            probe[f"{arm}@{k}"] = [int(j not in bad) for j in range(len(facts))]
    m["probe"] = {"facts": [r["text"] for r in facts], "questions": qs, "kept": probe}
    json.dump(m, open(path, "w"), ensure_ascii=False, indent=1)
    print("built", ci, "sessions", len(ss), "checkpoints", cps,
          {k: f"{100 * sum(v) / len(v):.0f}%" for k, v in probe.items()}, flush=True)


def build(_):
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(build_conv, ARGS.convs))


def report(_):
    by, pair = collections.defaultdict(list), [0, 0]
    for ci in ARGS.convs:
        p = os.path.join(OUT, f"mem_{ci}.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        for pos, k in enumerate(m["checkpoints"]):
            r, b = m["probe"]["kept"][f"roll@{k}"], m["probe"]["kept"][f"rebuild@{k}"]
            by[("roll", pos)] += r; by[("rebuild", pos)] += b
            if pos == 2:
                pair[0] += sum(y > x for x, y in zip(r, b)); pair[1] += sum(x > y for x, y in zip(r, b))
    print(f"early-fact retention, BEAM {ARGS.split}, budget {L.BUDGET}, checkpoints 1/3, 2/3, end")
    for arm in ("roll", "rebuild"):
        print(f"  {arm:8s}" + "".join(f"{100 * sum(by[(arm, p)]) / len(by[(arm, p)]):8.1f} n={len(by[(arm, p)]):<4d}"
                                      for p in range(3) if by[(arm, p)]))
    print(f"  end: rebuild-only {pair[0]}, roll-only {pair[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "build", "report"])
    ap.add_argument("--split", default="1M")
    ap.add_argument("--convs", default="0,1,2,3,4")
    ap.add_argument("--budget", type=int, default=L.BUDGET)
    ap.add_argument("--par", type=int, default=2)
    ARGS = ap.parse_args()
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    if ARGS.budget != L.BUDGET:
        L.SUM_SYS = L.SUM_SYS.replace(str(L.BUDGET), str(ARGS.budget))
        L.BUDGET = ARGS.budget
    OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "longdrift",
                       f"{llm.ARM}_{ARGS.split}" + ("" if L.BUDGET == 8000 else f"_b{L.BUDGET}"))
    os.makedirs(OUT, exist_ok=True)
    {"fetch": fetch, "build": build, "report": report}[ARGS.step](None)
