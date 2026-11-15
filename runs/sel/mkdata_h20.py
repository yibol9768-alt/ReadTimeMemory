"""Step-level training groups for the set-conditioned selector, from env questions + pools.

State S (the rounds already selected) is drawn to look like test-time states (review fix: s0 trained on mostly-gold
states and collapsed at test time):
  policy states : S = the first k rounds of a ranking policy over the pool (k ~ U[0, 30]); the policy is the 8B
                  teacher order (--policy rr8) or a trained model's order read from ctx_<TAG>.jsonl (--policy ctx:TAG,
                  i.e. DAgger on our own selector's choices)
  mixed states  : a random subset of covered gold stages + distractors (old scheme), with prob --p-mixed
Given S, covered stages C = stages of gold rounds in S.
  positives  = pool rounds of gold stages not in C (one sampled per group);
  negatives  = redundant rounds (stage in C, not in S), teacher-hard non-gold rounds, random pool rounds.
  Support-only rounds (e.g. older values for knowledge-update) and other positives are never negatives.
Every document and selected item carries its previous-user-message context (fmt.round_prev). No question type.

usage: python3 mkdata.py QFILE POOLDIR OUT.jsonl [--steps 8 --K 15 --P 600 --policy rr8|ctx:TAG --p-mixed 0.3 --ids FILE]"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1 import rounds_of
from fmt import query_str, doc_str, split_round, round_prev


def kinds_of(q):
    """synthetic round kind per round, aligned with rounds_of (user + following assistant = one round)"""
    out = []
    for sess in q["haystack_sessions"]:
        i = 0
        while i < len(sess):
            k = sess[i].get("kind")
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant": i += 1
            i += 1; out.append(k)
    return out


def main(argv=None, group_sink=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("qfile"); ap.add_argument("pooldir"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=8); ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--P", type=int, default=600); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ids", default="")
    ap.add_argument("--policy", default="rr8"); ap.add_argument("--p-mixed", type=float, default=0.3)
    ap.add_argument("--kmax", type=int, default=45)
    # 10-31 transfer tests: --no-set = pointwise (empty <Selected> always); --gold-rounds = real data without stage
    # labels (each gold round is its own stage); --hard-skip N = do not use the N top-scored non-gold rounds as hard
    # negatives (the 8B top is where unlabelled on-topic siblings concentrate, i.e. false negatives)
    ap.add_argument("--no-set", action="store_true"); ap.add_argument("--gold-rounds", action="store_true")
    ap.add_argument("--hard-skip", type=int, default=0); ap.add_argument("--split-not", default="")
    ap.add_argument("--boost", type=int, default=1, help="step multiplier for event ordering and summarization")
    # 10-31 experiment X: rounds whose synthetic kind is listed are neither positives nor negatives ("ignore")
    ap.add_argument("--ignore-kinds", default="")
    ap.add_argument("--fact-stage", action="store_true", help="world data: rounds of one ledger fact share a stage, so a restatement of a selected fact is a redundant negative")
    ap.add_argument("--p-empty", type=float, default=0.0, help="share of groups with an empty selected set")
    ap.add_argument("--teacher", default="rr8")
    ap.add_argument("--teacher-config", default="")
    ap.add_argument("--teacher-only", action="store_true", help="SPEC_STAGE23 v3.2 stage 2: unlabelled teacher groups; never reads gold, stage or has_answer")
    ap.add_argument("--n-groups", type=int, default=0, help="--teacher-only: total groups to write (questions cycled in a seeded order)")
    ap.add_argument("--cands", default="", help="fnfilter cands JSON")
    ap.add_argument("--fn-support", default="", help="fnfilter verdict JSON; support and ignore never negatives")
    ap.add_argument("--neg-mix", action="store_true", help="D4: 15 negatives, struct/8b/ret/stu/rand quotas 5/3/2/2/3")
    ap.add_argument("--group-table", default="", help="D4 freeze JSON (groups + spare)")
    ap.add_argument("--arm", choices=["C", "I", "P0", "P0P1"], default="", help="required with --group-table")
    ap.add_argument("--shared-fn-support", default="", help="frozen C/I: verdicts used ONLY for common eligibility")
    ap.add_argument("--drop-report", default="", help="default OUT.drops.json")
    ap.add_argument("--qtext", default="", help="qid -> rewritten question text JSON")
    ap.add_argument("--inst", choices=["v1", "v2"], default="", help="overrides FMT_INST (default unchanged)")
    a = ap.parse_args(argv)
    if a.inst:
        import fmt
        fmt.INST_VERSION = a.inst[1:]
        fmt.INST = fmt.INST_V2 if a.inst == "v2" else fmt.INST_V1
    if a.group_table:
        from d4data import frozen_main
        return frozen_main(a)
    if a.arm or a.shared_fn_support: ap.error("--arm/--shared-fn-support require --group-table")
    if a.neg_mix and (not a.cands or a.K != 15): ap.error("--neg-mix requires --cands and --K 15")
    if a.teacher_only and (a.cands or a.fn_support or a.neg_mix or a.qtext):
        ap.error("D4 labels/candidates/qtext cannot be combined with --teacher-only")
    from d4data import load_map, blocked_rounds, sample_mix, sample_classic
    candidates = load_map(a.cands); verdicts = load_map(a.fn_support)
    qtext = load_map(a.qtext, wrapped=False)
    rng = random.Random(a.seed)
    qs = json.load(open(a.qfile))
    if a.ids: keep = set(open(a.ids).read().split()); qs = [q for q in qs if q["question_id"] in keep]
    pol = {}
    if a.policy.startswith("ctx:"):
        for l in open(f"ctx_{a.policy[4:]}.jsonl"):
            r = json.loads(l)
            if r["budget"] == 16000: pol[r["qid"]] = r.get("sel", [])
    if a.teacher_only: return teacher_only(a, qs, rng)
    n_out = n_skip = 0
    with open(a.out, "w") as f:
        for q in qs:
            qid = q["question_id"]
            blocked = blocked_rounds(verdicts, qid) if a.fn_support else set()
            entry = candidates[qid] if a.cands else None
            if a.fn_support and qid not in verdicts: raise ValueError(f"missing fn verdict: {qid}")
            pp = os.path.join(a.pooldir, q["question_id"] + ".json")
            if not os.path.exists(pp): n_skip += 1; continue
            if a.split_not and q.get("split") == a.split_not: n_skip += 1; continue
            pool = json.load(open(pp)); R = rounds_of(q); st = list(pool["stage"]); prev = round_prev(q)
            if a.ignore_kinds:
                IK = set(a.ignore_kinds.split(",")); ks = kinds_of(q)
                st = [("__ignore__" if (i < len(ks) and ks[i] in IK) else x) for i, x in enumerate(st)]
            if a.gold_rounds:
                st = [f"g{i}" if x else None for i, x in enumerate(pool["gold"])]
                q = {**q, "gold_stages": [x for x in st if x], "support_stages": []}
            first_kind = kinds_of(q)
            if a.fact_stage:
                fs = lambda x: x.split("#")[0] if isinstance(x, str) else x
                st = [fs(x) for x in st]; q = {**q, "gold_stages": [fs(x) for x in q["gold_stages"]], "support_stages": [fs(x) for x in q.get("support_stages", [])]}
            G = set(q["gold_stages"]); SUP = set(q.get("support_stages", [])) - G
            P = pool["fused"][: a.P]; rr = {int(k): v for k, v in pool["rr8"].items()}
            by_stage = {}
            for i in P:
                if st[i] in G: by_stage.setdefault(st[i], []).append(i)
            ev_stage = {g: [i for i in v if first_kind[i] == "ev"] for g, v in by_stage.items()} if a.fact_stage else by_stage
            present = [g for g in q["gold_stages"] if g in by_stage]
            if not present: n_skip += 1; continue
            nongold = [i for i in P if st[i] not in G and st[i] not in SUP and st[i] != "__ignore__"]
            ranked_ng = sorted([i for i in nongold if i in rr], key=lambda i: -rr[i])
            hard = ranked_ng[a.hard_skip: a.hard_skip + 60]
            nongold = [i for i in nongold if i not in set(ranked_ng[: a.hard_skip])]
            # Compute hard-skip BEFORE filtering, so filtering cannot move skipped rounds back in.
            if blocked:
                hard = [i for i in hard if i not in blocked]
                nongold = [i for i in nongold if i not in blocked]
            order = pol.get(q["question_id"]) or sorted(P, key=lambda i: -rr.get(i, -1))
            for _ in range(a.steps * (a.boost if q["question_type"] in ("event_ordering", "summarization", "aggregation", "ordering") else 1)):
                if a.no_set or rng.random() < a.p_empty:
                    S = []
                elif rng.random() < a.p_mixed:
                    C0 = set(rng.sample(present, rng.randint(0, len(present) - 1)))
                    S = [rng.choice(by_stage[g]) for g in C0] + rng.sample(hard, min(rng.choice([0, 1, 2, 4, 8]), len(hard)))
                    rng.shuffle(S)
                else:
                    S = list(order[: rng.randint(0, a.kmax)])
                C = {st[i] for i in S if st[i] in G}
                unc = [g for g in present if g not in C]
                if not unc: continue
                u_ = rng.choice(unc); pos = rng.choice(ev_stage.get(u_) or by_stage[u_])
                redund = [i for g in C for i in by_stage.get(g, []) if i not in S and i not in blocked]
                negs = []
                def take(src, k):
                    src = [x for x in src if x not in negs and x not in S and x != pos]
                    negs.extend(rng.sample(src, min(k, len(src))))
                if a.neg_mix:
                    allowed = set(nongold) - set(S) - {pos}
                    negs, neg_src = sample_mix(entry, allowed, rng)
                elif entry is not None:
                    negs, _ = sample_classic(entry, hard, nongold, set(S) | {pos}, rng, a.K)
                else:
                    take(redund, 3); take(hard, a.K - len(negs) - 4); take(nongold, a.K - len(negs))
                negs = negs[: a.K]
                if len(negs) < (15 if a.neg_mix else 4): continue
                if group_sink is not None:
                    # Observe the unmodified C generator without drawing additional random numbers.
                    import hashlib
                    neg_seed = int.from_bytes(hashlib.sha256(repr(rng.getstate()).encode()).digest()[:8], "big")
                    group_sink({"qid": qid, "positive": pos, "neg_seed": neg_seed,
                                "type": q["question_type"], "nneg": len(negs)})
                    n_out += 1
                    continue
                sel = [(R[i][1], split_round(R[i][2])[0], prev[i]) for i in S]
                docs = [doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in [pos] + negs]
                f.write(json.dumps({"qid": q["question_id"], "type": q["question_type"],
                                    "query": query_str(qtext.get(qid, q["question"]), q["question_date"], sel),
                                    "docs": docs, "label": 0, "nsel": len(S), "ncov": len(C), "ngold": len(present),
                                    "kinds": ["pos"] + ["red" if i in redund else ("hard" if i in hard else "rand") for i in negs],
                                    "teacher": [rr.get(i) for i in [pos] + negs],
                                    **({"neg_src": neg_src} if a.neg_mix else {})}) + "\n")
                n_out += 1
    print("groups", n_out, "skipped questions", n_skip)


def rounds_text(q):
    """(session_idx, date, text) per round exactly as k1.rounds_of, but never touching has_answer (audit 11-02)"""
    R = []
    for si, (sess, date) in enumerate(zip(q["haystack_sessions"], q["haystack_dates"])):
        i = 0
        while i < len(sess):
            turns = [sess[i]]
            if sess[i]["role"] == "user" and i + 1 < len(sess) and sess[i + 1]["role"] == "assistant":
                turns.append(sess[i + 1]); i += 1
            i += 1
            R.append((si, date, "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)))
    return R


def teacher_only(a, qs, rng):
    """Stage-2 teacher groups (SPEC_STAGE23 v3.2 4): selected set empty; 16 candidates drawn without replacement from the
    pool rounds that have the configured teacher scores: 4 of teacher ranks 1-20, 8 of ranks 21-300, 4 more from
    the remaining scored candidates. Questions with fewer than 28 scored candidates are skipped. Uses only the question,
    the pool order and the teacher scores; the gold / stage fields of the pool and the has_answer flags are never read."""
    from fmt import round_prev as _rp
    from teacher_s23 import teacher_maps, group_logits
    usable = []
    for q in qs:
        pp = os.path.join(a.pooldir, q["question_id"] + ".json")
        if not os.path.exists(pp): continue
        pool = json.load(open(pp)); P = pool["fused"][: a.P]
        maps = teacher_maps(pool, q["question_id"], getattr(a, "teacher", "rr8"), getattr(a, "teacher_config", ""))
        available = [i for i in P[:300] if all(i in m for m in maps)]
        rr = dict(zip(available, group_logits(maps, available))) if available else {}
        scored = sorted((i for i in P[:300] if i in rr), key=lambda i: (-rr[i], i))   # retrieval top 300 (old pools score 600, new pools 300: same set either way)
        if len(scored) < 28: continue
        usable.append((q, scored, maps))
    n_skip = len(qs) - len(usable)
    if not usable: print("groups 0 skipped questions", n_skip); return
    order = list(range(len(usable))); rng.shuffle(order)
    n_out = 0
    with open(a.out, "w") as f:
        while n_out < a.n_groups:
            for j in order:
                if n_out >= a.n_groups: break
                q, scored, maps = usable[j]
                R = rounds_text(q); prev = _rp(q)
                pick = rng.sample(scored[:20], 4) + rng.sample(scored[20:300], min(8, len(scored[20:300])))
                rest = [i for i in scored if i not in set(pick)]
                pick += rng.sample(rest, min(16 - len(pick), len(rest)))
                docs = [doc_str(R[i][1], *split_round(R[i][2]), prev=prev[i]) for i in pick]
                f.write(json.dumps({"qid": q["question_id"], "query": query_str(q["question"], q["question_date"], []),
                                    "docs": docs, "nolabel": True, "kinds": ["t"] * len(pick),
                                    "teacher": group_logits(maps, pick), "teacher_format": "logit",
                                    "teacher_name": getattr(a, "teacher", "rr8")}) + "\n")
                n_out += 1
    print("groups", n_out, "skipped questions", n_skip, "usable questions", len(usable))


if __name__ == "__main__":
    main()
