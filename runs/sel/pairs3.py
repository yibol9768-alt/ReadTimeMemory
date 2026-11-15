"""Counterfactual v3 diagnostics: default 140 pairs, eight balanced interventions.

History variants retain ledger_id and split. pairs.json references standard q*.json
questions, so model evaluation can consume them with the existing pool pipeline.
"""
import argparse
import copy
from pathlib import Path
import random
import re
from world3_domains import vocabulary
from world3_style import anchors_for, assemble_turns

from world3 import (ModelFamilies, NAMES, add_version, build_questions, core, day,
                    gates, make_plan, question_text, read, render, solve,
                    validate_ledger, verify, write, digest, allocation, load_tokenizer, tokenizer_fingerprint, BUILD_CONTRACT, naturalize)

PRESERVE = ("rename", "reporter", "perspective")
CHANGE = ("asked_person", "effective_time", "condition", "approval", "scope")


def diagnostic_plan(world=9000, seed=0, split="audit", qtype="direct", confirmation=False):
    w = make_plan(world, 80, seed, split, revision=2)
    candidates = [c for c in w["cases"] if c["qtype"] == qtype]
    # A simple causal base for state/scope edits; complex structures are kept
    # intact in the multi-hop, aggregate and rule diagnostic bases.
    case = min(candidates, key=lambda c: c["expected_evidence"])
    w["cases"] = [case]
    w["facts"] = [f for f in w["facts"] if f["source"]["case"] == case["id"]]
    initial = next(f for f in w["facts"] if f["id"] == case["targets"][0])
    if confirmation and initial["status"] == "executed":
        case["targets"][0] = initial["prev"]  # approval itself is the effective evidence
        w["facts"] = [f for f in w["facts"] if f["id"] != initial["id"]]
    mapping = {f["id"]: f"f{j}" for j, f in enumerate(w["facts"])}
    for j, f in enumerate(w["facts"]):
        f.update(id=mapping[f["id"]], seq=j, prev=mapping.get(f["prev"]),
                 supersedes=mapping.get(f["supersedes"]))
    for key in ("targets", "negative_atoms", "state_support", "lifecycle_atoms"):
        if key in case:
            case[key] = [mapping[x] for x in case[key]]
    target = next(f for f in w["facts"] if f["id"] == case["targets"][0])
    case.update(old_atom=target["supersedes"], current_atom=target["id"])
    # A condition intervention must change meaning, not only the atom ID.
    emergency = [f for f in w["facts"] if f["entity"] == target["entity"] and f["attr"] == target["attr"]
                 and f["holder"] == target["holder"] and f["scope"] == target["scope"]
                 and f["condition"] and f["condition"] != target["condition"]]
    for f in emergency:
        if f["value"] == target["value"]:
            f["value"] = next(v for v in vocabulary(w["ontology"], target["attr"]) if v != target["value"])
            f["statement"] = core(f)
    # A true holder-exchange diagnostic needs different semantic outcomes,
    # including when a vocabulary legitimately repeats across holders.
    if qtype == "direct":
        for f in w["facts"]:
            if (f["entity"] == target["entity"] and f["attr"] == target["attr"]
                    and f["holder"] != target["holder"] and f["value"] == target["value"]):
                f["value"] = next(val for val in vocabulary(w["ontology"], target["attr"]) if val != target["value"])
    byid = {f["id"]: f for f in w["facts"]}
    for f in w["facts"]:
        if f["supersedes"]:
            f["replacement"] = {k: byid[f["supersedes"]][k] for k in ("value", "event_time", "proposal")}
        f["statement"] = core(f)
    if confirmation:
        target["confirmation"] = True
    w["ledger_id"] = f"diagnostic:{seed}:{world}"
    w["diagnostic_structure"] = "meaningful_confirmation" if confirmation else qtype
    validate_ledger(w["facts"])
    w = naturalize(w)
    w["natural_render"] = True
    w["plan_fingerprint"] = digest(w)
    return w


def rename_text(text, mapping):
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, mapping)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group()], text)


def intervene(w, intervention, serial=0):
    """Mutate only one causal dimension; all answers/gold are subsequently solved."""
    out = copy.deepcopy(w)
    case = out["cases"][0]
    facts = {f["id"]: f for f in out["facts"]}
    current = facts[case["current_atom"]]
    if intervention == "rename":
        def alias(number, width):
            bank={3:['Amy','Ben','Dan','Eli','Eva','Ian','Ivy','Jay','Joe','Kai','Kim','Leo','Max','Mia','Nia','Ray','Sam','Sky'],
                  4:['Amir','Anna','Evan','Finn','Hana','Hugo','Inez','Ivan','Jade','Jana','Jude','Lena','Liam','Luca','Maya','Milo','Nora','Omar'],
                  5:['Alice','Amira','Chloe','Dylan','Elena','Emery','Grace','Hazel','Jonah','Layla','Leila','Logan','Lucia','Mabel','Naomi','Nolan','Oscar','Peter']}[width]
            return bank[number%len(bank)]
        # Same-length names preserve real fmt context boundaries (200 chars),
        # rather than claiming positions survive an arbitrarily longer name.
        mapping = {n: alias(serial * len(NAMES) + i, len(n)) for i, n in enumerate(NAMES)}
        # Rename identities everywhere (ledger, questions, message content and
        # metadata). Atom IDs and normalized evidence labels are invariant.
        def walk(value):
            if isinstance(value, str):
                return rename_text(value, mapping)
            if isinstance(value, list):
                return [walk(v) for v in value]
            if isinstance(value, dict):
                return {k: walk(v) for k, v in value.items()}
            return value
        out = walk(out)
        case = out["cases"][0]
        case["identity_bijection"] = mapping
    elif intervention == "reporter":
        # Change *reporters*, never holders. This applies to statements as well
        # as every piece of equivalent evidence, not only the first occurrence.
        original = case["reporter"]
        replacement = next((name for name in out["team"] if name != original and len(name) == len(original)),
                           {3: "Rae", 4: "Noor", 5: "Quinn"}[len(original)])
        case["reporter"] = replacement
        if case["perspective"] == "self":
            case["asker"] = replacement
        for m in out["rounds"]:
            if m["atoms"]:
                # Assistant remains on the assistant side. Only its attributed
                # human reporter changes; no evidence is moved into user text.
                if m["speaker"] == "Assistant":
                    m["reported_by"] = replacement
                elif m["speaker"] == original:
                    m["speaker"] = replacement
                elif m["speaker"] == replacement:
                    m["speaker"] = original
                for turn in m.get('turns',[]):
                    if turn['speaker']==original:turn['speaker']=replacement
                    elif turn['speaker']==replacement:turn['speaker']=original
                assemble_turns(m,m)
    elif intervention == "perspective":
        case["perspective"] = "external" if case["perspective"] != "external" else "participant"
        case["asker"] = None if case["perspective"] == "external" else NAMES[1]
        # Message facts are re-expressed as a narrated record in the same round
        # positions. No context-only paraphrase is allowed to lose a carrier.
        for m in out["rounds"]:
            if m["atoms"]:
                m["text"] += " This is the narrated discussion record."
                m["expression"] = "third_person_record"
    elif intervention == "asked_person":
        alt = next(f for f in out["facts"] if f["entity"] == current["entity"] and f["attr"] == current["attr"]
                   and f["scope"] == current["scope"] and f["holder"] != current["holder"] and f["status"] == "executed")
        case["query"]["requests"][0]["holder"] = alt["holder"]
    elif intervention == "effective_time":
        case["explicit_as_of"] = True
        case["query"]["as_of"] = facts[case["old_atom"]]["event_time"]
    elif intervention == "condition":
        ont = out["ontology"]
        current_mode = case["query"]["context"][ont["context_key"]]
        case["query"]["context"] = {ont["context_key"]: ont["routine"] if current_mode == ont["emergency"] else ont["emergency"]}
    elif intervention == "scope":
        case["query"]["requests"][0]["scope"] = "team" if current["scope"] == "personal" else "personal"
    elif intervention == "approval":
        # Approved -> vetoed is legal. Replace the final executed event by a
        # veto of that proposal, retaining ID/position for a true minimal edit.
        current["status"] = "vetoed"
        current["confirmation"] = False
        current["statement"] = core(current)
        for m in out["rounds"]:
            if current["id"] in m["atoms"]:
                old_core = m["core"]
                m["core"] = current["statement"]
                if 'anchor_sets' in m:
                    m['anchor_sets'][current['id']]=anchors_for(current)
                m["text"] = m["text"].replace(old_core, current["statement"])
                m["draft"] = m["draft"].replace(old_core, current["statement"])
                for turn in m.get('turns',[]):
                    turn['text']=turn['text'].replace(old_core,current['statement'])
                for key in ("text", "draft"):
                    for phrase in (" I approve this decision and commit to carrying it out as recorded.",
                                   " I confirm that this decision was carried out as recorded."):
                        m[key] = m[key].replace(phrase, "")
                        for turn in m.get('turns',[]):turn['text']=turn['text'].replace(phrase,'')
                assemble_turns(m,m)
                m["meaningful_confirmation"] = False
                m["confirmation_kind"] = None
                m["ok"] = False
        # Revalidation is mandatory because this edit changes semantic labels.
        out["verified"] = False
        case["semantic_ok"] = False
    else:
        raise ValueError(intervention)
    case["question"] = question_text(case)
    validate_ledger(out["facts"])
    return out


def generate_pairs(n_pairs=140, seed=0, models=None, world=9000,
                   checkpoint_dir=None, par=4, tokenizer=None, token_limit=800,
                   tokenizer_id=None):
    if not 120 <= n_pairs <= 200:
        raise ValueError("diagnostic target is 120-200 pairs")
    if models is None:
        raise ValueError("pass real models or explicitly construct ModelFamilies(mock=True)")
    tokenizer_id = tokenizer_id or getattr(tokenizer, "world3_fingerprint", None)
    root = Path(checkpoint_dir) if checkpoint_dir else None
    structures = ["direct"] * 4 + ["multihop", "multihop", "aggregation", "aggregation", "rule", "rule"]
    bases = []
    for i, kind in enumerate(structures):
        plan = diagnostic_plan(world + i, seed + i, qtype=kind, confirmation=i in (3, 9))
        wp = root / f"w{world + i:05d}.json" if root else None
        base = read(wp) if wp and wp.exists() else plan
        if base["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("diagnostic checkpoint plan changed")
        blocks = root / "blocks" / str(world + i) if root else None
        if not base.get("rendered"):
            base = render(base, models, par, blocks)
            if wp:
                write(wp, base)
        elif base.get("renderer_identity") != models.identity:
            raise ValueError("diagnostic renderer family changed on resume")
        if not base.get("verified"):
            base = verify(base, models, blocks)
            if wp:
                write(wp, base)
        elif base.get("verifier_identity") != models.identity:
            raise ValueError("diagnostic verifier family changed on resume")
        bases.append(base)
    questions, pairs = [], []
    kinds = list(PRESERVE + CHANGE)
    random.Random(seed).shuffle(kinds)
    for i in range(n_pairs):
        intervention = kinds[i % len(kinds)]
        # Changes isolate one causal selector on a direct base. Preserving
        # interventions exercise the complete multi-hop/aggregate/rule graph.
        index = (4 + (i // len(kinds)) % 6) if intervention in PRESERVE else (i // len(kinds)) % 4
        base = bases[index]
        path = root / "pairs" / f"pair{i:04d}.json" if root else None
        fingerprint = digest([BUILD_CONTRACT, base["plan_fingerprint"], base["facts"], base["cases"], base["rounds"], intervention,
                              i, models.identity, tokenizer_id, token_limit])
        saved = read(path) if path and path.exists() else {}
        if saved.get("fingerprint") == fingerprint:
            if saved["status"] == "done":
                questions.extend(saved["questions"])
                pairs.append(saved["pair"])
            continue
        left = build_questions(base, tokenizer=tokenizer, token_limit=token_limit)
        variant = intervene(base, intervention, serial=i)
        blocks = root / "variants" / f"pair{i:04d}" if root else None
        variant = verify(variant, models, blocks)
        right = build_questions(variant, tokenizer=tokenizer, token_limit=token_limit)
        if not left or not right:
            if path:
                write(path, dict(fingerprint=fingerprint, status="quarantined",
                                 reason="semantic/token/necessity verification failed"))
            continue
        left, right = left[0], right[0]
        left["question_id"] = f"pair{i:04d}_left"
        right["question_id"] = f"pair{i:04d}_right"
        left_ids = {a["id"] for a in left["answer_atoms"]}
        right_ids = {a["id"] for a in right["answer_atoms"]}
        expected = "preserve" if intervention in PRESERVE else "change"
        if (left_ids == right_ids) != (expected == "preserve"):
            raise ValueError(f"counterfactual failed: {intervention}")
        if expected == "change" and left["answer"] == right["answer"]:
            raise ValueError("changed gold but unchanged semantic answer")
        if expected == "preserve":
            for la, ra in zip(left["answer_atoms"], right["answer_atoms"]):
                if la["equivalent_rounds"] != ra["equivalent_rounds"]:
                    raise ValueError("preserving intervention changed visible evidence positions")
        for q in (left, right):
            q["pair_id"] = f"pair{i:04d}"
            q["pair_intervention"] = intervention
        pair = dict(pair_id=f"pair{i:04d}", intervention=intervention, expected=expected,
                    left=left["question_id"], right=right["question_id"],
                    ledger_id=left["ledger_id"], split="audit", structure=base["diagnostic_structure"],
                    meaningful_confirmation=bool(left["meaningful_confirmations"]),
                    left_atoms=sorted(left_ids), right_atoms=sorted(right_ids),
                    identity_bijection=variant["cases"][0].get("identity_bijection", {}),
                    recomputed=True, intervention_spec={"base_world": base["world"], "kind": intervention, "serial": i})
        questions.extend([left, right])
        pairs.append(pair)
        if path:
            write(path, dict(fingerprint=fingerprint, status="done", questions=[left, right], pair=pair))
    return questions, pairs, bases

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--tokenizer")
    ap.add_argument("--max-d", type=int, default=800)
    ap.add_argument("--timeout", type=float, default=120)
    for flag in ("q27", "gem", "oss"):
        ap.add_argument("--" + flag, default="")
    a = ap.parse_args(argv)
    models = ModelFamilies([a.q27, a.gem, a.oss], mock=a.mock, timeout=a.timeout)
    tokenizer = load_tokenizer(a.tokenizer) if a.tokenizer else None
    questions, pairs, bases = generate_pairs(a.pairs, a.seed, models, a.start,
        Path(a.out) / "checkpoints", a.par, tokenizer, a.max_d, tokenizer_fingerprint(a.tokenizer))
    write(Path(a.out) / "qpairs.json", questions)
    write(Path(a.out) / "pairs.json", pairs)
    for w in bases:
        write(Path(a.out) / f"w{w['world']:05d}.json", w)
    write(Path(a.out) / "manifest.json", {"pairs": len(pairs), "questions": len(questions),
          "independent_ledgers": len(bases), "calls": dict(models.calls), "mock": a.mock,
          "planned_pairs": a.pairs, "quarantined_pairs": a.pairs - len(pairs),
          "interventions": {kind: sum(p["intervention"] == kind for p in pairs) for kind in PRESERVE + CHANGE},
          "structures": sorted({p["structure"] for p in pairs}),
          "token_visibility_complete": bool(questions) and all(q["token_visibility"]["ok"] for q in questions),
          "release_ready": not a.mock and len(pairs) == a.pairs and tokenizer is not None,
          "note": "Pair variants share lineage and must remain in the audit split."})
    print("pairs", len(pairs), "questions", len(questions), "calls", dict(models.calls))


if __name__ == "__main__":
    main()
