"""Test-selected rebuild (pilot): at a rebuild, write several candidate memories from the same ledger, quiz each on
SEL_FACTS ledger facts, keep the best. Tests choose between whole memories instead of patching one (patching is
zero-sum under a fixed budget, PROGRESS 10-15 10:26), so the only way to win is higher information density.

Candidates (same budget, same ledger = all records at the end of the conversation):
  base      drift.py's rebuild prompt (cache hit)
  base2     the same prompt, second sample (tells apart format gains from sampling noise)
  dense     telegraphic: one fact per line, no prose
  person    grouped by person, then by topic, "key: value"
  timeline  dated entries in time order
Evaluation: the EVAL_FACTS ledger facts of rehearse lprobe/tavg (seed eval-{ci}); selection facts are a disjoint
sample (seed sel-{ci}). Reported: every candidate on eval, the candidate picked on selection facts, and the oracle.

Usage: python3 -m mem.selrebuild build|report|merge|controls|chunk [--convs 0,2,4,5,7,8] [--par 3]
"""
import argparse, json, os, random
from concurrent.futures import ThreadPoolExecutor
from . import llm, locomo_exp as L, rehearse as H

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "select", llm.ARM)
SEL_FACTS = 40
STYLE = {
    "base": "",
    "base2": "\n\n(Second draft: write it afresh.)",
    "dense": "\n\nFormat: telegraphic notes, one fact per line, no prose, no filler words; keep names, dates, numbers "
             "and specifics, drop narration. Pack in as many distinct facts as the limit allows.",
    "person": "\n\nFormat: one section per person; inside, group by topic, lines of the form 'topic: detail (date)'. "
              "Pack in as many distinct facts as the limit allows.",
    "timeline": "\n\nFormat: a timeline, one line per dated event or fact ('YYYY-MM-DD: ...'), in time order, then a "
                "short 'standing facts' list for undated facts. Pack in as many distinct facts as the limit allows.",
}


def score(mem, facts, qs):
    bad = set()
    for b in range(0, len(facts), 20):
        bad |= {b + j for j in H.ask(mem, facts[b:b + 20], qs[b:b + 20])}
    return [int(j not in bad) for j in range(len(facts))]


def build_conv(ci, data):
    path = os.path.join(OUT, f"sel_{ci}.json")
    if os.path.exists(path):
        return
    ss = L.sessions(data[ci])

    def records(s):
        body = f"Session {s['n']}, date {s['date']}:\n" + "\n".join(L.show_turn(t, s["date"]) for t in s["turns"])
        return L.parse_records(llm.call(L.REC_SYS, body, tag="drift_rec"), {t["id"] for t in s["turns"]})
    recs = [records(s) for s in ss]
    pairs = [(si, r) for si, rs in enumerate(recs, 1) for r in rs]
    ev = random.Random(f"eval-{ci}").sample(pairs, min(H.EVAL_FACTS, len(pairs)))
    ev_ids = {id(r) for _, r in ev}
    sel = random.Random(f"sel-{ci}").sample([p for p in pairs if id(p[1]) not in ev_ids], SEL_FACTS)
    ev, sel = [r for _, r in ev], [r for _, r in sel]
    ev_q, sel_q = H.probes_for(ev), H.probes_for(sel)
    prompt = ("Memory records extracted so far from the conversation (in time order; each is self-contained):\n\n"
              + "\n".join("- " + r["text"] for rs in recs for r in rs)
              + f"\n\nWrite the memory of everything so far (at most {L.BUDGET} characters).")

    def one(name):
        mem = L.within_budget(prompt + STYLE[name], "drift_rebuild" if name == "base" else "sel_" + name)
        return name, {"mem": mem, "chars": len(mem), "sel": score(mem, sel, sel_q), "eval": score(mem, ev, ev_q)}
    with ThreadPoolExecutor(len(STYLE)) as ex:
        cands = dict(ex.map(one, STYLE))
    json.dump({"cands": cands}, open(path, "w"), ensure_ascii=False, indent=1)
    print("select", ci, {n: (sum(c["sel"]), sum(c["eval"]), c["chars"]) for n, c in cands.items()}, flush=True)


MERGE = ("Below are {n} independent drafts of the memory of the same conversation, each written from the same "
         "records. They overlap a lot, but each keeps some facts the others dropped. Write one memory that keeps "
         "every distinct fact found in any draft (names, dates, numbers, specifics), merging duplicates and "
         "dropping filler, in at most {b} characters.")


def merge(data):
    """H29: merge the candidates into one memory (union of facts, same budget) instead of picking one."""
    def one(ci):
        path = os.path.join(OUT, f"sel_{ci}.json")
        d = json.load(open(path))
        if "merge" in d:
            return
        ss = L.sessions(data[ci])
        recs = [L.parse_records(llm.call(L.REC_SYS, f"Session {s['n']}, date {s['date']}:\n" + "\n".join(
            L.show_turn(t, s["date"]) for t in s["turns"]), tag="drift_rec"), {t["id"] for t in s["turns"]}) for s in ss]
        pairs = [(si, r) for si, rs in enumerate(recs, 1) for r in rs]
        ev = [r for _, r in random.Random(f"eval-{ci}").sample(pairs, min(H.EVAL_FACTS, len(pairs)))]
        ev_q = H.probes_for(ev)
        out = {}
        for name, keys in {"merge3": ["base", "dense", "timeline"], "merge5": list(STYLE)}.items():
            drafts = "\n\n".join(f"=== Draft {i + 1} ===\n{d['cands'][k]['mem']}" for i, k in enumerate(keys))
            mem = L.within_budget(drafts + "\n\n" + MERGE.format(n=len(keys), b=L.BUDGET), "sel_" + name)
            out[name] = {"mem": mem, "chars": len(mem), "eval": score(mem, ev, ev_q)}
        d["merge"] = out
        json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
        print("merge", ci, {k: (sum(v["eval"]), v["chars"]) for k, v in out.items()},
              "base", sum(d["cands"]["base"]["eval"]), flush=True)
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(one, ARGS.convs))
    tot = {}
    for ci in ARGS.convs:
        d = json.load(open(os.path.join(OUT, f"sel_{ci}.json")))
        for k in ["base", "base2"]:
            tot.setdefault(k, []).append(d["cands"][k]["eval"])
        for k, v in d["merge"].items():
            tot.setdefault(k, []).append(v["eval"])
    for k, rows in tot.items():
        xs = [x for r in rows for x in r]
        b = [x for r in tot["base"] for x in r]
        print(f"  {k:7s} {100 * sum(xs) / len(xs):5.1f}  vs base {sum(x > y for x, y in zip(xs, b))}:{sum(y > x for x, y in zip(xs, b))}")


REFINE = ("Below is a draft of the memory of a conversation. Rewrite it into a better memory: keep every distinct fact "
          "(names, dates, numbers, specifics), merge duplicates, drop filler, and use the space well, in at most {b} "
          "characters.")
REFINE_LEDGER = ("Below are the memory records of a conversation and a draft memory written from them. Rewrite the draft "
                 "into a better memory of the records: keep every distinct fact the draft has, add important facts it "
                 "missed, merge duplicates, drop filler, in at most {b} characters.")


def controls(data):
    """H29 controls: is the merge gain from a second pass, from format diversity, or from sampling noise?
      refine1        the base draft alone, rewritten once (second pass, no extra drafts)
      refine_ledger  the base draft rewritten with the whole ledger in view
      same5          merge of 5 drafts of the SAME prompt (base, base2 and 3 more samples)
      merge5_s2      merge5 generated a second time (generation noise of the merge itself)"""
    def one(ci):
        path = os.path.join(OUT, f"sel_{ci}.json")
        d = json.load(open(path))
        if "controls" in d:
            return
        ss = L.sessions(data[ci])
        recs = [L.parse_records(llm.call(L.REC_SYS, f"Session {s['n']}, date {s['date']}:\n" + "\n".join(
            L.show_turn(t, s["date"]) for t in s["turns"]), tag="drift_rec"), {t["id"] for t in s["turns"]}) for s in ss]
        pairs = [(si, r) for si, rs in enumerate(recs, 1) for r in rs]
        ev = [r for _, r in random.Random(f"eval-{ci}").sample(pairs, min(H.EVAL_FACTS, len(pairs)))]
        ev_q = H.probes_for(ev)
        ledger = "\n".join("- " + r["text"] for rs in recs for r in rs)
        prompt = ("Memory records extracted so far from the conversation (in time order; each is self-contained):\n\n"
                  + ledger + f"\n\nWrite the memory of everything so far (at most {L.BUDGET} characters).")
        base = d["cands"]["base"]["mem"]
        extra = [L.within_budget(prompt + f"\n\n(Draft {k}: write it afresh.)", "sel_same") for k in (3, 4, 5)]
        same = [base, d["cands"]["base2"]["mem"]] + extra
        mems = {
            "refine1": L.within_budget(f"=== Draft ===\n{base}\n\n" + REFINE.format(b=L.BUDGET), "sel_refine1"),
            "refine_ledger": L.within_budget(f"Records:\n{ledger}\n\n=== Draft ===\n{base}\n\n"
                                             + REFINE_LEDGER.format(b=L.BUDGET), "sel_refine_ledger"),
            "same5": L.within_budget("\n\n".join(f"=== Draft {i + 1} ===\n{m}" for i, m in enumerate(same))
                                     + "\n\n" + MERGE.format(n=5, b=L.BUDGET), "sel_same5"),
            "merge5_s2": L.within_budget("\n\n".join(f"=== Draft {i + 1} ===\n{d['cands'][k]['mem']}"
                                                       for i, k in enumerate(STYLE))
                                         + "\n\n" + MERGE.format(n=5, b=L.BUDGET) + "\n\n(Second merge.)",
                                         "sel_merge5_s2"),
        }
        d["controls"] = {k: {"mem": m, "chars": len(m), "eval": score(m, ev, ev_q)} for k, m in mems.items()}
        json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
        print("controls", ci, {k: (sum(v["eval"]), v["chars"]) for k, v in d["controls"].items()},
              "base", sum(d["cands"]["base"]["eval"]), "merge5", sum(d["merge"]["merge5"]["eval"]), flush=True)
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(one, ARGS.convs))
    tot = {}
    for ci in ARGS.convs:
        d = json.load(open(os.path.join(OUT, f"sel_{ci}.json")))
        rows = {"base": d["cands"]["base"], "base2": d["cands"]["base2"], **d["merge"], **d.get("controls", {})}
        for k, v in rows.items():
            tot.setdefault(k, []).extend(v["eval"])
    for k, xs in tot.items():
        b = tot["base"]
        print(f"  {k:13s} {100 * sum(xs) / len(xs):5.1f}  vs base {sum(x > y for x, y in zip(xs, b))}:{sum(y > x for x, y in zip(xs, b))}")


def within(prompt, budget, tag):
    """within_budget with an arbitrary limit (the system prompt names the limit too)."""
    sys_ = L.SUM_SYS.replace(str(L.BUDGET), str(budget))
    text = llm.call(sys_, prompt, tag=tag)
    for i in range(3):
        if len(text) <= budget:
            break
        text = llm.call(sys_, prompt + f"\n\nYour previous attempt was {len(text)} characters, over the limit. "
                        f"Write it again in at most {int(budget * (0.85 - 0.1 * i))} characters.", tag=tag + "_retry")
    return text[:budget]


def chunked(recs, k, budget, tag):
    """H30: split the ledger into k contiguous groups of sessions with about equal numbers of records, compress
    each group on its own into a share of the budget proportional to its records, and concatenate."""
    n = sum(map(len, recs))
    groups, cur, cnt = [], [], 0
    for si, rs in enumerate(recs, 1):
        cur.append((si, rs)); cnt += len(rs)
        if cnt >= n * (len(groups) + 1) / k and len(groups) < k - 1:
            groups.append(cur); cur = []
    groups.append(cur)
    parts = []
    for g in groups:
        if not g:
            continue
        m = sum(len(rs) for _, rs in g)
        head = f"[Sessions {g[0][0]}-{g[-1][0]}]\n"
        b = int(budget * m / n) - len(head) - 2
        prompt = (f"Memory records extracted from sessions {g[0][0]}-{g[-1][0]} of the conversation (in time order; "
                  "each is self-contained):\n\n" + "\n".join("- " + r["text"] for _, rs in g for r in rs)
                  + f"\n\nWrite the memory of these sessions (at most {b} characters).")
        parts.append(head + within(prompt, b, tag))
    return "\n\n".join(parts)[:budget]


def chunk(data):
    """H30 static test at the final checkpoint: chunked compression with k = 2, 4, 8 (cost about one rebuild)."""
    def one(ci):
        path = os.path.join(OUT, f"sel_{ci}.json")
        d = json.load(open(path))
        if "chunk" in d:
            return
        ss = L.sessions(data[ci])
        recs = [L.parse_records(llm.call(L.REC_SYS, f"Session {s['n']}, date {s['date']}:\n" + "\n".join(
            L.show_turn(t, s["date"]) for t in s["turns"]), tag="drift_rec"), {t["id"] for t in s["turns"]}) for s in ss]
        pairs = [(si, r) for si, rs in enumerate(recs, 1) for r in rs]
        ev = [r for _, r in random.Random(f"eval-{ci}").sample(pairs, min(H.EVAL_FACTS, len(pairs)))]
        ev_q = H.probes_for(ev)
        out = {}
        for k in (2, 4, 8):
            mem = chunked(recs, k, L.BUDGET, f"sel_chunk{k}")
            out[f"chunk{k}"] = {"mem": mem, "chars": len(mem), "eval": score(mem, ev, ev_q)}
        d["chunk"] = out
        json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
        print("chunk", ci, {k: (sum(v["eval"]), v["chars"]) for k, v in out.items()},
              "base", sum(d["cands"]["base"]["eval"]), "merge5", sum(d["merge"]["merge5"]["eval"]), flush=True)
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(one, ARGS.convs))
    tot = {}
    for ci in ARGS.convs:
        d = json.load(open(os.path.join(OUT, f"sel_{ci}.json")))
        rows = {"base": d["cands"]["base"], "base2": d["cands"]["base2"], "merge5": d["merge"]["merge5"],
                **d.get("controls", {}), **d["chunk"]}
        for k, v in rows.items():
            tot.setdefault(k, []).extend(v["eval"])
    for k, xs in tot.items():
        b = tot["base"]
        print(f"  {k:13s} {100 * sum(xs) / len(xs):5.1f}  vs base {sum(x > y for x, y in zip(xs, b))}:{sum(y > x for x, y in zip(xs, b))}")


def build(data):
    llm.check_quota()
    with ThreadPoolExecutor(min(ARGS.par, len(ARGS.convs))) as ex:
        list(ex.map(lambda ci: build_conv(ci, data), ARGS.convs))


def report(data):
    tot, picked, oracle, pair, n = {}, [], [], [0, 0], 0
    for ci in ARGS.convs:
        p = os.path.join(OUT, f"sel_{ci}.json")
        if not os.path.exists(p):
            continue
        c = json.load(open(p))["cands"]
        for name, v in c.items():
            tot.setdefault(name, []).extend(v["eval"])
        best = max(c, key=lambda x: (sum(c[x]["sel"]), x == "base"))
        picked += c[best]["eval"]; oracle += max(c.values(), key=lambda v: sum(v["eval"]))["eval"]
        pair[0] += sum(x > y for x, y in zip(c[best]["eval"], c["base"]["eval"]))
        pair[1] += sum(y > x for x, y in zip(c[best]["eval"], c["base"]["eval"]))
        print(f"  conv {ci}: picked {best}", {k: f"{sum(v['sel'])}/{sum(v['eval'])}/{v['chars']}" for k, v in c.items()})
    print("eval retention (final checkpoint, ledger facts)")
    for name, xs in sorted(tot.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {name:9s} {100 * sum(xs) / len(xs):5.1f}  n={len(xs)}")
    print(f"  selected  {100 * sum(picked) / len(picked):5.1f}  vs base {pair[0]}:{pair[1]}")
    print(f"  oracle    {100 * sum(oracle) / len(oracle):5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "report", "merge", "controls", "chunk"])
    ap.add_argument("--convs", default="0,2,4,5,7,8")
    ap.add_argument("--par", type=int, default=3)
    ARGS = ap.parse_args()
    ARGS.convs = [int(x) for x in ARGS.convs.split(",")]
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(L.DATA))
    {"build": build, "report": report, "merge": merge, "controls": controls, "chunk": chunk}[ARGS.step](data)
