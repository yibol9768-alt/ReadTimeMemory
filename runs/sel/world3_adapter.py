"""v3 atom-aware groups without changing mkdata.py or pool.py.

pool.py outputs are read unchanged. Use this adapter for exact structural quotas
and rounds that cover multiple atoms. Stock mkdata remains usable for its legacy
sampling recipe; v3 does not claim that recipe enforces the new quotas.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import re
import glob
import hashlib
import math
import datetime as dt

from world3_probes import universal_probes, competition_check, ideal_prediction
from world3_structure import distribution
from world3_style import atom_present, anchors_present

PANEL_QUOTAS = {"competing_cluster": 15}

from fmt import clean, doc_str, query_str, round_prev, split_round
from k1 import rounds_of
from world3 import (NEG_QUOTAS, CARRIERS, QTYPES, EVIDENCE, PERSPECTIVES,
                    EVIDENCE_V3, digest, tokenizer_fingerprint, load_tokenizer, read, sample_negatives, write, visible_documents, CORE_CLAUSE, BUILD_CONTRACT)


def mock_pool(q):
    """No scores impersonate a teacher: explicit offline candidate order only."""
    n = len(rounds_of(q))
    return {"fused": list(range(n)), "rr8": {}, "mock": True}


def rendered_query(q, selected, tokenizer=None, max_q=4000):
    R, prev = rounds_of(q), round_prev(q)
    sel = [(R[i][1], split_round(R[i][2])[0], prev[i]) for i in selected]
    full = query_str(q["question"], q["question_date"], sel)
    rendered = training_query(full, tokenizer, max_q)
    selected_text = full.split("<Selected>:\n", 1)[1]
    covered = {a["id"] for a in q["answer_atoms"]
               if atom_present(a,clean(selected_text)) and atom_present(a,clean(rendered))}
    return full, covered


def training_query(text, tokenizer, max_q):
    if tokenizer is None:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_q:
        if max_q <= 200:
            raise ValueError("max_q must exceed train.py's 200-token prefix")
        ids = ids[:200] + ids[-(max_q - 200):]
    return tokenizer.decode(ids)


def groups_for(q, pool, steps=1, no_set=False, seed=0, p=600, strict=True,
               tokenizer=None, max_q=4000, max_d=800):
    token_config = q.get("token_visibility", {})
    if tokenizer is not None and token_config.get("checked"):
        if (token_config.get("tokenizer_fingerprint") != getattr(tokenizer, "world3_fingerprint", None)
                or token_config.get("token_limit") != max_d):
            raise ValueError("tokenizer/max_d differs from question necessity and visibility verification")
    rng = random.Random(f"{seed}:{q['question_id']}")
    R = rounds_of(q)
    prev = round_prev(q)
    documents = visible_documents(q)
    token_docs = visible_documents(q, token_limit=max_d, tokenizer=tokenizer) if tokenizer is not None else documents
    # Distinct round IDs can render to the same candidate. Deduplicate by the
    # actual training-visible document, including tokenizer truncation.
    available, seen_docs = set(), set()
    for i in pool["fused"][:p]:
        if token_docs[i] not in seen_docs:
            available.add(i)
            seen_docs.add(token_docs[i])
    visible_atoms = {str(i): [a["id"] for a in q["answer_atoms"] if atom_present(a,token_docs[i])]
                     for i in available}
    visible_facts = {i for i in available if (clauses := CORE_CLAUSE.findall(R[i][2]))
                     and all(clause in token_docs[i] for clause in clauses)}
    if q.get('evidence_catalog'):
        visible_facts.update(i for i in available if any(atom_present(f,token_docs[i]) for f in q['evidence_catalog']))
    atoms = {a["id"] for a in q["answer_atoms"]}
    meta = [m for sess in q["haystack_sessions"] for m in sess if m["role"] == "user"]
    protected = {int(i) for i, ids in q["round_atoms"].items() if ids} | set(q["support_rounds"])
    reports, groups = [], []
    for step in range(steps):
        selected = []
        if not no_set:
            order = sorted(available, key=lambda i: -float(pool.get("rr8", {}).get(str(i), 0)))
            selected = order[:rng.randint(0, min(30, len(order)))]
        query, covered = rendered_query(q, selected, tokenizer, max_q)
        if tokenizer is not None and clean(q["question"]) not in clean(training_query(query, tokenizer, max_q)):
            reports.append({"qid": q["question_id"], "step": step, "drop": "token_invisible_question"})
            continue
        missing = atoms - covered
        if not missing:
            # An exhausted policy state has no training target. Record, don't
            # relabel an equivalent carrier or silently substitute a fresh state.
            reports.append({"qid": q["question_id"], "step": step, "drop": "all_atoms_covered"})
            continue
        positives = [i for i in available - set(selected) if missing & set(visible_atoms[str(i)])]
        if q.get("revision", 2) >= 3:
            positive_clusters = {indices[0] for indices in q.get("competition_panels", {}).values() if indices}
            positives = [i for i in positives if i in positive_clusters]
        if not positives:
            hidden = tokenizer is not None and any(missing & set(q["round_atoms"].get(str(i), []))
                                                   for i in set(pool["fused"][:p]) - set(selected))
            reports.append({"qid": q["question_id"], "step": step,
                            "drop": "token_invisible_positive" if hidden else "no_uncovered_positive_in_pool"})
            continue
        # Preserve the planner's within-type source-role balance when several
        # equivalent/uncovered carriers are available. In later policy states
        # a required hop may have only another source role: retain its correct
        # label, report the fallback, and audit the actual consumed roles.
        balanced = [i for i in positives
            if (meta[i]["speaker"] == q["asker"]) == q.get("evidence_speaker_is_asker")
            and (meta[i]["speaker"] in {a.get("holder") for a in q["answer_atoms"]
                 if a["id"] in visible_atoms[str(i)]}) == q.get("evidence_speaker_is_holder")]
        eligible=sorted(balanced or positives)
        if q.get('query_spec',{}).get('domain_program'):
            # Native graphs contain different numbers of nodes per role. Sample
            # the required attribute role uniformly before selecting its carrier.
            role_atoms={a['id']:a['attr'] for a in q['answer_atoms']}
            roles=collections.defaultdict(list)
            for i in eligible:
                role=role_atoms[sorted(set(visible_atoms[str(i)]) & missing)[0]]
                roles[role].append(i)
            positive=rng.choice(roles[rng.choice(sorted(roles))])
        else:
            positive=rng.choice(eligible)
        focus = sorted(set(visible_atoms[str(positive)]) & missing)[0]
        candidates = q.get("atom_negative_candidates", {}).get(focus, q["negative_candidates"])
        candidates = {kind: [i for i in ids if kind == "contentless_ack" or i in visible_facts]
                      for kind, ids in candidates.items()}
        requirements = q.get("atom_negative_requirements", {}).get(focus, {})
        allowed = available - protected - set(selected) - {positive}
        if q.get("revision", 2) >= 3:
            panel = q.get("competition_panels", {}).get(focus, [])
            negatives = [i for i in panel[1:] if i in allowed and i in visible_facts]
            labels = ["competing_cluster"] * len(negatives)
            shortage = {"competition_panel": 15 - len(negatives)}
        else:
            negatives, labels, shortage = sample_negatives(candidates, rng, allowed, requirements)
        report = {"qid": q["question_id"], "step": step, "shortfall": shortage,
                  "recipe": q.get("negative_recipe", "legacy_v2_quotas"),
                  "negative_focus_atom": focus,
                  "source_role_fallback": not bool(balanced),
                  "available": {k: len(set(v) & allowed) for k, v in candidates.items()}}
        if strict and any(shortage.values()):
            report["drop"] = "structural_quota_shortfall"
            reports.append(report)
            continue
        reports.append(report)
        indices = [positive] + negatives
        docs = [documents[i] for i in indices]
        token_ok = tokenizer is not None and bool(set(visible_atoms[str(positive)]) & missing)
        ideal = None
        competition = None
        if q.get("revision", 2) >= 3:
            ideal_q = dict(q, _selected_text=training_query(query, tokenizer, max_q).split("<Selected>:\n", 1)[-1])
            try:
                prediction, computed_answer = ideal_prediction(ideal_q, [token_docs[i] for i in indices], covered)
                ideal = dict(predictions=prediction, correct=prediction == [0] and computed_answer == q["answer"],
                    method='verified_anchor_projection_and_question_parser' if q.get('evidence_catalog') else 'visible_text_and_question_parser')
            except (ValueError, KeyError) as exc:
                ideal = dict(correct=False, reason=str(exc))
            competition = competition_check([token_docs[i] for i in indices], 0, q.get('evidence_catalog'))
        atom_authors = []
        for a in q["answer_atoms"]:
            if a["id"] not in visible_atoms[str(positive)]:
                continue
            origin = positive if a["id"] in meta[positive].get("atoms", []) else positive - 1
            atom_authors.append(dict(atom=a["id"], speaker=meta[origin]["speaker"], holder=a.get("holder")))
        groups.append(dict(qid=q["question_id"], type=q["question_type"],
                           version=3, revision=q.get("revision", 2),
                           ideal_rule=ideal, competition=competition,
                           structure_hash=q.get("structure_hash"),
                           query=query, docs=docs,
                           label=0, nsel=len(selected), ncov=len(covered), ngold=len(atoms),
                           kinds=["pos"] + labels, neg_src=labels, negative_shortfall=shortage,
                           teacher=[pool.get("rr8", {}).get(str(i)) for i in indices],
                           round_indices=indices, selected_rounds=selected,
                           positive_atoms=sorted(set(visible_atoms[str(positive)]) & missing),
                           visible_document_count=len({token_docs[i] for i in indices}),
                           negative_focus_atom=focus,
                           negative_subtypes=({kind: [panel[i] for i, k in enumerate(q["panel_subtypes"][focus]) if kind in k and panel[i] in negatives]
                               for kind in ("vetoed", "other_time", "condition_or_future")} if q.get("revision", 2) >= 3 else
                               {name: sorted(set(negatives) & set(ids)) for name, ids in requirements.get("minimal_state_diff", {}).items()}),
                           window_hard_negative_indices=sorted(set(indices[1:]) & set(q.get("window_hard_negatives", []))),
                           covered_atoms=sorted(covered), evidence_count=len(atoms),
                           assistant_only=q["assistant_only"], date_window=q["date_window"],
                           speakers=[meta[i]["speaker"] for i in indices], asker=q["asker"],
                           atom_authors=atom_authors,
                           positive_holders=[a.get("holder") for a in q["answer_atoms"]
                               if a["id"] in visible_atoms[str(positive)]],
                           meaningful_confirmation=meta[positive].get("meaningful_confirmation", False),
                           confirmation_kind=meta[positive].get("confirmation_kind"),
                           domain=q.get("domain"), entity=q["query_spec"]["entity"],
                           value_vocabulary=q.get("value_vocabulary", []),
                           token_visibility=dict(checked=tokenizer is not None, ok=token_ok,
                               max_q=max_q, max_d=max_d,
                               tokenizer_fingerprint=getattr(tokenizer, "world3_fingerprint", None)),
                           carrier=q["carrier"], perspective=q["perspective"], ledger_id=q["ledger_id"],
                           split=q["split"], mock=q.get("mock", False) or pool.get("mock", False)))
    return groups, reports


def blind_scores(groups):
    """Held-out document-only Bernoulli NB: words plus visible event age.

    No question/entity/asker/atom metadata enters features. Hold out complete
    ledgers, or complete question IDs when auditing a single ledger.
    """
    cache = {}
    def features(document):
        if document not in cache:
            fs = set(re.findall(r"[a-z][a-z0-9_-]*|\d+", document.lower()))
            recorded = re.search(r"<Document>:\s*\(([^)]+)\)", document)
            event = re.search(r"On (\d{4}-\d\d-\d\d),", document)
            if recorded and event:
                try:
                    date = dt.datetime.strptime(recorded[1], "%B-%d-%Y").date()
                    fs.add("event_age=" + str((date - dt.date.fromisoformat(event[1])).days))
                except ValueError:
                    pass
            cache[document] = fs
        return cache[document]
    ledgers = {g["ledger_id"] for g in groups}
    multi = len(ledgers) > 1
    folds = [g["ledger_id"] if multi else int(hashlib.sha256(g["qid"].encode()).hexdigest(), 16) % 5 for g in groups]
    rows = [[features(d) for d in g["docs"]] for g in groups]
    result = [None] * len(groups)
    for fold in set(folds):
        counts, sizes = [collections.Counter(), collections.Counter()], [0, 0]
        for i, g in enumerate(groups):
            if folds[i] == fold:
                continue
            for j, fs in enumerate(rows[i]):
                label = int(j == g["label"])
                sizes[label] += 1
                counts[label].update(fs)
        weights = {}
        for feature in counts[0].keys() | counts[1].keys():
            p0, p1 = [(counts[k][feature] + 1) / (sizes[k] + 2) for k in (0, 1)]
            weights[feature] = math.log(p1 / (1 - p1)) - math.log(p0 / (1 - p0))
        for i in range(len(groups)):
            if folds[i] == fold:
                result[i] = [round(sum(weights.get(f, 0.) for f in sorted(fs)), 8) for fs in rows[i]]
    return result, "leave_one_ledger_out" if multi else "held_out_qid_folds_single_ledger"


def shortcut_probes(groups):
    """Unique correct maximum; ties are NOT counted as uniquely identified.

    Rules see only the named inputs. The no-question rule is trained on other
    ledgers/questions; it sees only document words and visible event age.
    Also report tie-randomized accuracy so abstention cannot hide a tie bias.
    """
    names = ("entity_value_vocabulary", "speaker_is_asker", "speaker_is_not_asker", "no_question")
    stats = {name: dict(unique_correct=0, tie_accuracy=0.) for name in names}
    blind, validation = blind_scores(groups)
    for gi, g in enumerate(groups):
        patterns = [re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)", re.I)
                    for v in g["value_vocabulary"]]
        density = [sum(bool(p.search(d)) for p in patterns) for d in g["docs"]]
        scores = {
            "entity_value_vocabulary": [n if anchors_present([g['entity']],d) else 0 for n, d in zip(density, g["docs"])],
            "speaker_is_asker": [int(s == g["asker"]) for s in g["speakers"]],
            "speaker_is_not_asker": [int(s != g["asker"]) for s in g["speakers"]],
            "no_question": blind[gi],
        }
        for name, values in scores.items():
            best = [i for i, v in enumerate(values) if v == max(values)]
            correct = g["label"] in best
            stats[name]["unique_correct"] += int(correct and len(best) == 1)
            stats[name]["tie_accuracy"] += int(correct) / len(best)
    n = len(groups)
    chance = sum(1 / len(g["docs"]) for g in groups) / max(1, n)
    for row in stats.values():
        row.update(n=n, unique_rate=row["unique_correct"] / max(1, n),
                   tie_accuracy=row["tie_accuracy"] / max(1, n), chance=chance, threshold=chance + .10)
        row["pass"] = bool(n) and row["unique_rate"] <= row["threshold"]
    stats["no_question"].update(method="document_only_bernoulli_nb_words_event_age", validation=validation)
    stats.update(universal_probes(groups))
    if any(g.get("revision",2)>=3 for g in groups):
        from world3_extra_probes import extra_probes
        stats.update({"extra_"+name:row for name,row in extra_probes(groups).items()})
    return stats


def evidence_bin(n):
    return "one" if n == 1 else "two" if n == 2 else "three_five" if n <= 5 else "six_eight"


def length_distribution(groups):
    """Empirical CDF distance and relative mean gap, separately per carrier.

    Each group contributes one positive and the average of its 15 negatives.
    This tests distributions, rather than imposing an exact per-message length.
    """
    result={}
    for carrier in sorted({g['carrier'] for g in groups}):
        rows=[g for g in groups if g['carrier']==carrier]
        pos=sorted(len(g['docs'][g['label']]) for g in rows)
        neg=sorted(len(d) for g in rows for i,d in enumerate(g['docs']) if i!=g['label'])
        import bisect
        ks=max((abs(bisect.bisect_right(pos,t)/len(pos)-bisect.bisect_right(neg,t)/len(neg))
                for t in set(pos+neg)),default=1.)
        mean_gap=abs(sum(pos)/len(pos)-sum(neg)/len(neg))/max(1,sum(neg)/len(neg))
        # A sentence is a discrete trimming unit; a 100-character grid keeps
        # one-word/speaker-length differences from turning into a false shift.
        binned_pos=[n//100 for n in pos];binned_neg=[n//100 for n in neg]
        binned_ks=max((abs(bisect.bisect_right(binned_pos,t)/len(pos)-bisect.bisect_right(binned_neg,t)/len(neg))
                       for t in set(binned_pos+binned_neg)),default=1.)
        result[carrier]=dict(n=len(rows),ks=ks,binned_ks=binned_ks,relative_mean_gap=mean_gap,
            **{'pass':binned_ks<=.10+1e-9 and mean_gap<=.05+1e-9})
    return dict(by_carrier=result,bin_chars=100,max_cdf_gap=.10,max_relative_mean_gap=.05,
                **{'pass':bool(groups) and all(r['pass'] for r in result.values())})


def match_length_groups(groups):
    """Drop entire unmatched groups, preserving every structural negative.

    No text or labels are edited. The final audit still checks carrier, date,
    speaker and type quotas and every shortcut probe; callers must resample
    worlds if filtering leaves a deficit. Dropped IDs are always reported.
    """
    kept=list(groups);dropped=[]
    def loss(rows):
        report=length_distribution(rows)
        return sum(max(0,r['binned_ks']-.10)+max(0,r['relative_mean_gap']-.05)
                   for r in report['by_carrier'].values())
    while kept and not length_distribution(kept)['pass']:
        bad={c for c,r in length_distribution(kept)['by_carrier'].items() if not r['pass']}
        best=None
        for i,g in enumerate(kept):
            if g['carrier'] not in bad:continue
            score=loss(kept[:i]+kept[i+1:])
            candidate=(score,str(g['qid']),i)
            if best is None or candidate<best:best=candidate
        if best is None:break
        # Even when every immediate move is flat, a deterministic whole-group
        # removal can reach a match. It may also empty a stratum, reported below.
        _,_,i=best
        g=kept.pop(i);dropped.append(dict(qid=g['qid'],carrier=g['carrier'],drop='length_distribution_mismatch'))
    report=length_distribution(kept)
    report['missing_carriers']=sorted({g['carrier'] for g in groups}-{g['carrier'] for g in kept})
    report['pass'] &= not report['missing_carriers']
    return kept,dropped,report


def final_audit(groups, questions, reports, natural_report=None):
    count = lambda field: dict(collections.Counter(g[field] for g in groups))
    n = len(groups)
    hist = {"carrier": count("carrier"), "type": count("type"), "perspective": count("perspective"),
            "domain": count("domain"), "evidence": dict(collections.Counter(evidence_bin(g["evidence_count"]) for g in groups)),
            "evidence_count": count("evidence_count"), "assistant_only": count("assistant_only"),
            "date_window": dict(collections.Counter(bool(g["date_window"]) for g in groups)),
            "meaningful_confirmation": count("meaningful_confirmation")}
    speakers = {}
    for kind in QTYPES:
        rows = [g for g in groups if g["type"] == kind]
        speakers[kind] = {"n": len(rows),
            "asker": dict(collections.Counter("self" if g["speakers"][g["label"]] == g["asker"] else "other" for g in rows)),
            "holder": dict(collections.Counter("self" if g["speakers"][g["label"]] in g["positive_holders"] else "other" for g in rows)),
            "positive_names": dict(collections.Counter(g["speakers"][g["label"]] for g in rows)),
            "negative_asker": dict(collections.Counter("self" if speaker == g["asker"] else "other"
                for g in rows for i, speaker in enumerate(g["speakers"]) if i != g["label"])),
            "n_evidence_atoms": sum(len(g["atom_authors"]) for g in rows),
            "evidence_asker": dict(collections.Counter("self" if a["speaker"] == g["asker"] else "other"
                for g in rows for a in g["atom_authors"])),
            "evidence_holder": dict(collections.Counter("self" if a["speaker"] == a["holder"] else "other"
                for g in rows for a in g["atom_authors"]))}
    cross = {f"{c}|{t}|{p}": sum(g["carrier"] == c and g["type"] == t and g["perspective"] == p for g in groups)
             for c in CARRIERS for t in QTYPES for p in PERSPECTIVES}
    planned = collections.Counter(f"{q['carrier']}|{q['question_type']}|{q['perspective']}" for q in questions)
    lost = [cell for cell, total in planned.items() if total and not cross[cell]]
    probes = shortcut_probes(groups)
    from natural_stats import summarize
    naturalness=natural_report or dict(world3=summarize(groups),real_baseline_measured=False,
        naturalness_pass=False,status='pending_real_benchmark_pool_comparison')
    natural_ok=bool(naturalness.get('naturalness_pass') and naturalness.get('real_baseline_measured')
        and not naturalness.get('mock') and naturalness.get('groups_semantic_sha256')==digest(groups))
    structures = distribution(questions)
    realized = [set(row["realized_hashes"]) for row in structures.values() if row["types"]]
    modern = any(g.get("revision", 2) >= 3 for g in groups)
    # Small smoke worlds get at most one group's integer rounding allowance.
    tolerance = .10 + 1 / max(1, n)
    checks = {"nonempty": bool(n), "no_lost_cross_cells": not lost,
              "negative_quotas": all(not any(g["negative_shortfall"].values())
                  and collections.Counter(g["neg_src"]) == (PANEL_QUOTAS if g.get("revision", 2) >= 3 else NEG_QUOTAS) and len(g["docs"]) == 16
                  and len(g["teacher"]) == 16 and g["label"] == 0 for g in groups),
              "positive_coverage": all(g["positive_atoms"] and set(g["positive_atoms"]) - set(g["covered_atoms"])
                  and g["ncov"] == len(g["covered_atoms"]) < g["ngold"] and 1 <= g["ngold"] <= 8 for g in groups),
              "unique_visible_documents": all(g["visible_document_count"] == len(g["docs"]) == len(set(g["docs"])) for g in groups),
              "negative_subtypes": all(all(g["negative_subtypes"].get(k) for k in
                  ("vetoed", "other_time", "condition_or_future")) for g in groups),
              "window_negatives": all(g.get("window_hard_negative_indices") for g in groups if g["date_window"]),
              "domain_structure_distinct": not modern or (all(realized) and all(not a & b for i, a in enumerate(realized) for b in realized[:i])),
              "ledger_holdout_validated": not modern or len({g["ledger_id"] for g in groups}) > 1,
              "competition": all(g.get("competition", {}).get("pass") for g in groups if g.get("revision", 2) >= 3),
              "question_alignment": all(q.get("question_alignment",{}).get("ok") for q in questions if q.get("revision",2)>=3),
              "native_share": not modern or all(sum(bool(q.get("query_spec",{}).get("domain_program")) for q in questions if q.get("domain")==d)/max(1,sum(q.get("domain")==d for q in questions))>=.4 for d in {q.get("domain") for q in questions}),
              "ideal_rule": all((g.get("ideal_rule") or {}).get("correct") for g in groups if g.get("revision", 2) >= 3),
              "shortcut_probes": all(r["pass"] for r in probes.values())}
    length_match=length_distribution(groups)
    checks['carrier_length_distributions']=not modern or length_match['pass']
    for field, target in (("carrier", CARRIERS), ("type", QTYPES), ("evidence", EVIDENCE_V3 if modern else EVIDENCE)):
        checks[field] = all(abs(hist[field].get(k, 0) / max(1, n) - v) <= tolerance for k, v in target.items())
    ua = [g for g in groups if g["carrier"] == "assistant"]
    checks["assistant_only"] = bool(ua) and abs(sum(g["assistant_only"] for g in ua) / len(ua) - .25) <= .10 + 1 / len(ua)
    checks["date_window"] = abs(sum(bool(g["date_window"]) for g in groups) / max(1, n) - .15) <= tolerance
    checks["speakers"] = all(row["n"] and all(abs(row[axis].get("self", 0) / row["n"] - .5) <= .15 + 1 / row["n"]
                                               for axis in ("asker", "holder")) for row in speakers.values())
    checks["evidence_speakers"] = all(row["n_evidence_atoms"] and all(
        abs(row[axis].get("self", 0) / row["n_evidence_atoms"] - .5) <= .15 + 1 / row["n_evidence_atoms"]
        for axis in ("evidence_asker", "evidence_holder")) for row in speakers.values())
    token_complete = bool(n) and all(g["token_visibility"]["checked"] and g["token_visibility"]["ok"]
        and g["token_visibility"].get("tokenizer_fingerprint") for g in groups)
    return dict(groups=n, questions=len(questions), quota=PANEL_QUOTAS if any(g.get("revision", 2) >= 3 for g in groups) else NEG_QUOTAS, rows=reports,
        domain_structure=structures, ideal_rule_accuracy=sum(bool((g.get("ideal_rule") or {}).get("correct")) for g in groups) / max(1, n),
        **hist, speakers=speakers, carrier_type_perspective=cross, planned_cross=dict(planned), lost_cells=lost,
        shortcut_probes=probes, thresholds={"marginal_absolute_error": .10, "integer_allowance": "1/n",
            "speaker_self_share": .5, "speaker_absolute_error": .15, "speaker_integer_allowance": "1/n_type",
            "assistant_only_share_within_assistant": .25, "date_window_share": .15,
            "unique_probe_max": "mean(1/candidates)+0.10", "lost_planned_cross_cells": 0,
            "structural_negative_shortfall": 0, "dated_groups_require_in_window_hard_negatives": True,
            "token_visibility_required_for_release": True},
        naturalness=naturalness, naturalness_verified=natural_ok, length_matching=length_match,
        source_role_fallbacks=sum(bool(r.get("source_role_fallback")) for r in reports if not r.get("drop")),
        checks=checks, cpu_contract_pass=all(checks.values()), token_visibility_complete=token_complete,
        release_ready=all(checks.values()) and token_complete and natural_ok and not any(g["mock"] for g in groups))


def export_s23(groups, directory, audit=None, seed=0):
    """Stock training JSONL schema; reserve distinct ledgers for S1/S2/S3.

    S1 is the same group schema as tr_w20.jsonl, NOT chat-completion SFT.
    Only train lineages can enter training. Never cycle a short input to 8k.
    """
    if not groups or any(g["split"] != "train" for g in groups):
        raise ValueError("s23 export requires nonempty train-only groups")
    is_mock = any(g["mock"] for g in groups)
    if not is_mock and (audit is None or not audit["release_ready"]):
        raise ValueError("s23 release requires passing final audit and actual tokenizer visibility")
    ledgers = sorted({g["ledger_id"] for g in groups})
    random.Random(seed).shuffle(ledgers)
    if len(ledgers) < 3:
        raise ValueError("s23 export needs at least three independent train ledgers")
    stages = {ledger: i % 3 for i, ledger in enumerate(ledgers)}
    chunks = [[g for g in groups if stages[g["ledger_id"]] == i] for i in range(3)]
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / name for name in ("tr_world3_s1.jsonl", "crucible_world3_s2.jsonl", "crucible_world3_s3.jsonl")]
    for path, rows in zip(paths, chunks):
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in rows))
        tmp.replace(path)
    combined = root / "crucible_world3_s23.jsonl"
    tmp = combined.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for rows in chunks[1:] for g in rows))
    tmp.replace(combined)
    manifest = dict(mock=is_mock, release_ready=bool(audit and audit["release_ready"] and not is_mock),
        stage_counts={str(i + 1): len(rows) for i, rows in enumerate(chunks)},
        stage_ledgers={str(i + 1): sorted({g["ledger_id"] for g in rows}) for i, rows in enumerate(chunks)},
        paths=[str(p) for p in paths], combined=str(combined), no_repetition=True,
        sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths + [combined]},
        s23_minimum=8000, s23_budget_count_ready=len(chunks[1]) >= 4000 and len(chunks[2]) >= 4000,
        required_s23_source_changes={
            "s1_flops/all_start/budget_stage1_actual": {"tr_w20.jsonl": str(paths[0])},
            "build/build_all/prepare_budget": {"tr_w20.jsonl": str(combined)},
            "build/build_all/prepare_budget_stage_reservation": "read separate S2/S3 files; seeded take(combined) otherwise loses the ledger reservation",
            "preflight/provenance/frozen_budget": "replace matching source keys and hashes; recompute budgets (do not reuse old FLOPs)"})
    write(root / "s23_manifest.json", manifest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qfile", help="question JSON, quoted glob or world-output directory")
    ap.add_argument("pooldir")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=1, help="equal per-question multiplier; no type boost")
    ap.add_argument("--P", type=int, help="real pool default 600; mock default all rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-set", action="store_true")
    ap.add_argument("--mock-pools", action="store_true")
    ap.add_argument("--allow-shortfall", action="store_true")
    ap.add_argument("--allow-audit-failure", action="store_true", help="write diagnostic groups even when final audit fails")
    ap.add_argument("--tokenizer", help="same local tokenizer as train.py")
    ap.add_argument("--max-q", type=int, default=4000)
    ap.add_argument("--max-d", type=int, default=800)
    ap.add_argument("--natural-stats", help="matching real-pool report from natural_stats.py, required for release_ready")
    ap.add_argument("--export-s23", help="also reserve independent train ledgers for stages 1/2/3")
    a = ap.parse_args(argv)
    tokenizer = load_tokenizer(a.tokenizer) if a.tokenizer else None
    files = sorted(Path(a.qfile).glob("q[0-9]*.json")) if Path(a.qfile).is_dir() else [Path(p) for p in sorted(glob.glob(a.qfile))]
    if not files:
        raise ValueError("no question input files")
    questions, gate_status = [], {}
    for file in files:
        questions.extend(read(file))
        gp = file.with_name("g" + file.name[1:])
        if gp.exists():
            gate_status.update({r["qid"]: r.get("completion") == "complete" and r["suff"] == 1 for r in read(gp)})
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("duplicate question identity")
    groups, report = [], []
    for q in questions:
        if q.get("version") != 3 or q.get("revision") != 3 or not q.get("atom_negative_requirements"):
            report.append({"qid": q["question_id"], "drop": "requires_regenerated_v3_revision3"})
            continue
        if not q.get("mock") and not gate_status.get(q["question_id"]):
            report.append({"qid": q["question_id"], "drop": "missing_or_failed_release_gate"})
            continue
        path = Path(a.pooldir) / (q["question_id"] + ".json")
        if not a.mock_pools and not path.exists():
            report.append({"qid": q["question_id"], "drop": "missing_pool"})
            continue
        pool = mock_pool(q) if a.mock_pools else read(path)
        cap = a.P if a.P is not None else len(pool["fused"]) if a.mock_pools else 600
        config = dict(contract=BUILD_CONTRACT, adapter=6, steps=a.steps, no_set=a.no_set, seed=a.seed, p=cap, strict=not a.allow_shortfall,
                      tokenizer=tokenizer_fingerprint(a.tokenizer), max_q=a.max_q, max_d=a.max_d)
        fingerprint = digest([q, pool, config])
        checkpoint = Path(a.out + ".checkpoints") / (q["question_id"] + ".json")
        saved = read(checkpoint) if checkpoint.exists() else {}
        if saved.get("fingerprint") == fingerprint:
            new, rows = saved["groups"], saved["reports"]
        else:
            new, rows = groups_for(q, pool, a.steps, a.no_set, a.seed, cap, not a.allow_shortfall,
                                   tokenizer, a.max_q, a.max_d)
            write(checkpoint, dict(fingerprint=fingerprint, groups=new, reports=rows))
        groups.extend(new)
        report.extend(rows)
    groups,length_drops,length_match=match_length_groups(groups)
    report.extend(length_drops)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in groups))
    tmp.replace(path)
    audit = final_audit(groups, questions, report, read(a.natural_stats) if a.natural_stats else None)
    write(a.out + ".audit.json", audit)
    if not audit["cpu_contract_pass"] and not a.allow_audit_failure:
        raise ValueError("final group audit failed; see " + a.out + ".audit.json")
    if a.export_s23:
        export_s23(groups, a.export_s23, audit, a.seed)
    print("groups", len(groups), "report", a.out + ".audit.json")


if __name__ == "__main__":
    main()
