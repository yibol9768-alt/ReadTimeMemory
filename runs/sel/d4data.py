"""Opt-in D4 sampling and shared four-arm eligibility. No model calls.

Frozen-table runs require cands AND shared verdicts even for C/I: these inputs
determine common drops. Use --shared-fn-support for unfiltered C/I. Only the candidate
file's frozen lists may supply negatives. Hard-skip is computed on the original
pool ranking, before filtering. P0P1 does not use C's hard-skip.
"""
import collections
import json
import hashlib
import math
from pathlib import Path
import random

SOURCES = ("struct", "8b", "ret", "stu", "rand")
QUOTAS = (5, 3, 2, 2, 3)
ARMS = ("C", "I", "P0", "P0P1")
C_RECIPE = ["--no-set", "--ignore-kinds", "repeat,fu", "--hard-skip", "10",
            "--boost", "3", "--steps", "8"]


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_map(path, wrapped=True):
    if not path:
        return {}
    data = read_json(path)
    return data.get("q", data) if wrapped else data


def blocked_rounds(verdicts, qid):
    row = verdicts[qid]
    return set(map(int, row.get("support", []))) | set(map(int, row.get("ignore", [])))


def owned_sources(entry, allowed):
    """Dedup ownership before quotas: an unused struct round cannot become rand."""
    seen = set()
    out = {}
    for src in SOURCES:
        out[src] = []
        for value in entry.get(src, []):  # struct is round -> structural kind
            i = int(value)
            if i in allowed and i not in seen:
                out[src].append(i)
                seen.add(i)
    return out


def sample_mix(entry, allowed, rng):
    owned = owned_sources(entry, allowed)
    negs, labels, carry = [], [], 0
    for src, quota in zip(SOURCES, QUOTAS):
        n = quota + carry
        pick = rng.sample(owned[src], min(n, len(owned[src])))
        negs.extend(pick)
        labels.extend([src] * len(pick))
        carry = n - len(pick)
    # If the final source runs out, exhaust remaining allowed sources in the same
    # order. §12.2 requires trying all allowed candidates before a shared drop.
    for src in SOURCES:
        rest = [i for i in owned[src] if i not in negs]
        pick = rng.sample(rest, min(15 - len(negs), len(rest)))
        negs.extend(pick)
        labels.extend([src] * len(pick))
    return negs, labels


def classic_sources(entry, hard, nongold, exclude=()):
    allowed = set(nongold) - set(exclude)
    if entry is None:
        return [i for i in hard if i in allowed], [i for i in nongold if i in allowed]
    hardset = set(map(int, entry["8b"]))
    return ([i for i in hard if i in allowed and i in hardset],
            list(dict.fromkeys(int(i) for i in entry["rand"] if int(i) in allowed)))


def sample_classic(entry, hard, nongold, exclude, rng, k=15):
    hard, rand = classic_sources(entry, hard, nongold, exclude)
    negs, labels = [], []
    for src, values, count in (("8b", hard, k - 4), ("rand", rand, k), ("8b", hard, k)):
        values = [i for i in values if i not in negs]
        pick = rng.sample(values, min(max(0, count), k - len(negs), len(values)))
        negs.extend(pick)
        labels.extend([src] * len(pick))
    return negs, labels


def stratified_take(rows, n, rng, key):
    """Proportional allocation, largest remainders; sample without replacement."""
    if n >= len(rows):
        result = list(rows)
        rng.shuffle(result)
        return result
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    exact = {k: n * len(v) / len(rows) for k, v in buckets.items()}
    sizes = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(buckets, key=lambda k: (-(exact[k] - sizes[k]), k))[:n - sum(sizes.values())]:
        sizes[k] += 1
    result = []
    for k in sorted(buckets):
        result.extend(rng.sample(buckets[k], sizes[k]))
    rng.shuffle(result)
    return result


def prepare(q, pool, ignore_kinds, hard_skip, p=600):
    from mkdata import kinds_of
    st = list(pool["stage"])
    ks = kinds_of(q)
    ignored = set(ignore_kinds.split(","))
    st = ["__ignore__" if ks[i] in ignored else x for i, x in enumerate(st)]
    gold = set(q["gold_stages"])
    support = set(q.get("support_stages", [])) - gold
    candidates = pool["fused"][:p]
    rr = {int(k): v for k, v in pool["rr8"].items()}
    by_stage = {}
    for i in candidates:
        if st[i] in gold:
            by_stage.setdefault(st[i], []).append(i)
    present = [g for g in q["gold_stages"] if g in by_stage]
    ng = [i for i in candidates if st[i] not in gold | support | {"__ignore__"}]
    ranked = sorted([i for i in ng if i in rr], key=lambda i: -rr[i])
    skipped = set(ranked[:hard_skip])
    return dict(st=st, present=present, gold=gold, rr=rr, nongold=ng,
                hard=ranked[hard_skip:hard_skip + 60],
                classic_ng=[i for i in ng if i not in skipped])


def frozen_main(a):
    import fmt
    from k1 import rounds_of
    table = read_json(a.group_table)
    shared_fn = a.shared_fn_support or a.fn_support
    if not a.arm or not a.cands or not shared_fn:
        raise ValueError("--group-table requires --arm, --cands and shared verdicts (--shared-fn-support for C/I, --fn-support for P0/P0P1)")
    if a.arm in ("C", "I") and a.fn_support:
        raise ValueError("C/I must be unfiltered: use --shared-fn-support for common eligibility only")
    if a.arm in ("P0", "P0P1") and not a.fn_support:
        raise ValueError("P0/P0P1 require --fn-support")
    if table.get("status", "ready") != "ready":
        raise ValueError("freeze is not ready; inspect its problems before constructing arms")
    if a.teacher_only or a.gold_rounds or a.fact_stage or a.split_not or a.ids:
        raise ValueError("frozen tables fix the question list and positive policy; do not combine with selection overrides")
    if a.K != 15:
        raise ValueError("D4 requires --K 15")
    if a.neg_mix != (a.arm == "P0P1"):
        raise ValueError("--neg-mix must be enabled exactly for --arm P0P1")
    expected_inst = "v1" if a.arm == "C" else "v2"
    if a.inst and a.inst != expected_inst:
        raise ValueError(f"arm {a.arm} requires --inst {expected_inst}")
    fmt.INST_VERSION = expected_inst[1:]
    fmt.INST = fmt.INST_V1 if a.arm == "C" else fmt.INST_V2
    if a.qtext and a.arm in ("C", "I"):
        raise ValueError("C/I require original question text")
    recipe = table.get("recipe", {})
    if recipe.get("policy", "rr8") != "rr8" or not recipe.get("no_set", True):
        raise ValueError("only the frozen pointwise C recipe is supported")
    primary, spare = table["groups"], table.get("spare", [])
    target = table.get("target_groups", len(primary))
    if len(primary) < target:
        raise ValueError("freeze lacks the requested C groups; regenerate or explicitly reduce the frozen target")
    if len({(r["qid"], r["positive"], r["neg_seed"]) for r in primary + spare}) != len(primary + spare):
        raise ValueError("duplicate frozen group identity")
    needed = {r["qid"] for r in primary + spare}
    qs, pools = {}, {}
    sources = table.get("sources") or [{"qfile": a.qfile, "pooldir": a.pooldir}]
    base = Path(a.group_table).resolve().parent
    for src in sources:
        qpath, ppath = Path(src["qfile"]), Path(src["pooldir"])
        if not qpath.is_absolute(): qpath = base / qpath
        if not ppath.is_absolute(): ppath = base / ppath
        for q in read_json(qpath):
            qid = q["question_id"]
            if qid in needed:
                if qid in qs: raise ValueError(f"duplicate question: {qid}")
                qs[qid], pools[qid] = q, ppath / (qid + ".json")
    if needed - qs.keys():
        raise ValueError(f"missing frozen questions: {sorted(needed - qs.keys())[:5]}")
    cands, verdicts = load_map(a.cands), load_map(shared_fn)
    if a.fn_support and a.shared_fn_support and load_map(a.fn_support) != verdicts:
        raise ValueError("own and shared verdict maps must agree")
    qtext = load_map(a.qtext, wrapped=False)
    if any(not isinstance(v, str) or not v.strip() for v in qtext.values()):
        raise ValueError("qtext values must be nonempty strings")
    states, counts = {}, {}
    for qid in sorted(needed):
        if qid not in cands or qid not in verdicts:
            raise ValueError(f"missing shared candidate/verdict input for {qid}")
        state = prepare(qs[qid], read_json(pools[qid]), recipe.get("ignore_kinds", "repeat,fu"),
                        recipe.get("hard_skip", 10), recipe.get("P", 600))
        blocked = blocked_rounds(verdicts, qid)
        entry = cands[qid]
        classic = classic_sources(entry, state["hard"], state["classic_ng"])
        filtered = classic_sources(entry, state["hard"], [i for i in state["classic_ng"] if i not in blocked])
        mix = owned_sources(entry, set(state["nongold"]) - blocked)
        counts[qid] = dict(C=len(set(classic[0] + classic[1])), I=len(set(classic[0] + classic[1])),
                           P0=len(set(filtered[0] + filtered[1])), P0P1=sum(map(len, mix.values())))
        states[qid] = state
    dropped, chosen, replacements = [], [], []
    for is_spare, rows in ((False, primary), (True, spare)):
        for row in rows:
            if len(chosen) == target: break
            qid, pos = row["qid"], row["positive"]
            state = states[qid]
            if not isinstance(pos, int) or pos < 0 or pos >= len(state["st"]) or state["st"][pos] not in state["gold"]:
                raise ValueError(f"invalid frozen positive: {row}")
            bad = [arm for arm in ARMS if counts[qid][arm] < 15]
            if bad:
                dropped.append(dict(row, spare=is_spare, failing_arms=bad, available=counts[qid]))
            else:
                chosen.append(row)
                if is_spare:
                    replacements.append(dict(row, replaces=dropped[len(replacements)]))
    source_counts, source_lengths, structural_counts, carrier_counts = (collections.Counter() for _ in range(4))
    with open(a.out, "w") as f:
        last_qid, R, prev = None, None, None
        for row in chosen:
            qid, pos = row["qid"], row["positive"]
            q, state = qs[qid], states[qid]
            if qid != last_qid:
                R, prev = rounds_of(q), fmt.round_prev(q)
                last_qid = qid
            rng = random.Random(row["neg_seed"])
            blocked = blocked_rounds(verdicts, qid) if a.arm in ("P0", "P0P1") else set()
            if a.neg_mix:
                negs, labels = sample_mix(cands[qid], set(state["nongold"]) - blocked, rng)
            else:
                negs, labels = sample_classic(cands[qid], state["hard"],
                                             [i for i in state["classic_ng"] if i not in blocked], (), rng)
            if len(negs) != 15 or len(set(negs)) != 15:
                raise AssertionError(f"eligibility/sampling mismatch: {qid}")
            docs = [fmt.doc_str(R[i][1], *fmt.split_round(R[i][2]), prev=prev[i]) for i in [pos] + negs]
            for i, label, doc in zip(negs, labels, docs[1:]):
                source_counts[label] += 1
                source_lengths[label] += len(doc)
                if label == "struct": structural_counts[cands[qid]["struct"][str(i)]] += 1
                # Round carriers are aligned with k1's user + optional assistant pairing.
                session = q["haystack_sessions"][R[i][0]]
                carrier = "group" if "[Group " in R[i][2] else next((t.get("carrier", "dialog") for t in session), "dialog")
                carrier_counts[f"{label}:{carrier}"] += 1
            result = dict(qid=qid, type=q["question_type"],
                          query=fmt.query_str(qtext.get(qid, q["question"]), q["question_date"], []),
                          docs=docs, label=0, nsel=0, ncov=0, ngold=len(state["present"]),
                          kinds=["pos"] + ["hard" if label == "8b" else "rand" for label in labels],
                          teacher=[state["rr"].get(i) for i in [pos] + negs],
                          positive=pos, neg_seed=row["neg_seed"])
            if a.neg_mix: result["neg_src"] = labels
            f.write(json.dumps(result) + "\n")
    g = len(chosen)
    def fingerprint(path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""): digest.update(chunk)
        return digest.hexdigest()
    report = dict(arm=a.arm, instruction=expected_inst,
                  input_sha256=dict(group_table=fingerprint(a.group_table), candidates=fingerprint(a.cands),
                                    shared_verdicts=fingerprint(shared_fn)),
                  quota_rule="carry shortfalls forward; if the last source exhausts, refill earlier leftovers in source order",
                  candidate_rule="frozen cands only; C/P0 hard-skip on original pool order before filtering",
                  target=target, groups=g, dropped=dropped, replacements=replacements,
                  selected=chosen, spare_examined=len(replacements) + sum(r["spare"] for r in dropped),
                  spare_exhausted=g < target, steps=g // 16, I_steps=g // 32,
                  checkpoint_interval=max(10, 10 * int(100 * g / 9600 / 10)),
                  source_counts=source_counts,
                  source_mean_chars={k: source_lengths[k] / v for k, v in source_counts.items()},
                  structural_counts=structural_counts, source_carriers=carrier_counts,
                  difficulty="not measured; requires separate untrained-model inference (no GPU in data builder)")
    write_json(a.drop_report or a.out + ".drops.json", report)
    print(f"groups {g} shared drops {len(dropped)} replacements {len(replacements)} shortfall {target - g}")
