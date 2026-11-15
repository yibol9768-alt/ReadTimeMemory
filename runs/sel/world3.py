"""Executable Crucible v3 ledger, rendering, and compatible question export.

CPU/offline smoke: python world3.py all --mock --out /tmp/ws3 --questions 20
Real rendering uses exactly gen.Pool(spec).chat(...)[0]; no inference server is started.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import hashlib
import threading

from fmt import clean, doc_str, round_prev, split_round
from world3_style import (configure as configure_style, mock_message, valid_message,
    ASSISTANT_PROMPT, STATUS_SYNONYMS, anchors_for, anchors_present, atom_present, render_instructions,
    match_lengths, dedupe_fillers, message_failures, assemble_turns, augment_group, signature_for, leaks_anchors, filler_keys, collapse_whitespace, normalize as normalize_filler)
from k1 import ntok, rounds_of
from world3_domains import DOMAINS, vocabulary, all_values, natural_names
from world3_structure import STRUCTURES, RELATION_TYPES, INSTRUCTIONS, parents, solve_relations, structure_hash, distribution, observed_structure

CARRIERS = {"group": .40, "assistant": .30, "dyad": .15, "record": .15}
QTYPES = {"direct": .20, "preference": .10, "rule": .10, "update": .15,
          "multihop": .15, "aggregation": .15, "temporal": .10, "recommendation": .05}
EVIDENCE = {"one": .40, "two": .30, "three_five": .25, "six_eight": .05}
EVIDENCE_V3 = {"one": .25, "two": .30, "three_five": .40, "six_eight": .05}
NEG_QUOTAS = {"same_speaker_other_topic": 4, "same_topic_wrong_scope": 4,
              "minimal_state_diff": 3, "contentless_ack": 2, "random": 2}
PERSPECTIVES = ("self", "participant", "external")
TRANSITIONS = {None: {"proposed"}, "proposed": {"approved", "vetoed"},
               "approved": {"executed", "vetoed"}, "executed": {"rolled_back"},
               "vetoed": set(), "rolled_back": set()}
NAMES = ("Ari", "Bela", "Cleo", "Davi", "Esme", "Farah")
DAY0 = dt.date(2025, 1, 20)
BUILD_CONTRACT = "v3r8_single_message_natural_status"
VERIFY_CONTRACT = "2026-11-08_message_required_carriers"
NATIVE_TASKS = {
    'aggregation': 'Summarize all reachable outcomes, following the listed native attribute types from the scoped root.',
    'temporal': 'Report reachable outcomes in event-time order, including each event date and the interval in days between the first and last events. Break ties by entity name and then attribute name.',
    'multihop': 'Report the derived route and its conclusions.',
    'direct': 'Report the derived route and its conclusions.',
}
EFFECTIVE = ("approved", "executed") + tuple(x["live"] for x in STRUCTURES.values())
CORE_CLAUSE = re.compile(r"On \d{4}-\d\d-\d\d, [^\n]*? was (?:[a-z_]+) "
                         r"\([^\n]*?\)\. Decision [A-Z]{8}\.(?: It replaces decision [A-Z]{8}\.)?(?: Links: [^\n]*?\.)?")
# Controlled natural source prose remains parseable in offline oracle tests.
NATURAL_CLAUSE = re.compile(r"On \d{4}-\d\d-\d\d, [^\n]*? had [^\n]*?; its status was [^\n]*?\."
    r"(?: This applied [^\n]*?\.)?(?: This was the [^\n]*? update\.)?"
    r"(?: It replaced the [^\n]*? update\.| It replaced the entry from \d{4}-\d\d-\d\d\.)?(?: It depended on [^\n]*?\.)?")
CORE_CLAUSE = re.compile('(?:' + CORE_CLAUSE.pattern + ')|(?:' + NATURAL_CLAUSE.pattern + ')')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_vote(models, family, instructions, payload, cached=None, blocked=False):
    if blocked:
        return {"ok": False, "reason": "quarantined render message"}
    if cached is not None:
        return cached
    try:
        return models.ask(family, instructions, payload)
    except ValueError as exc:
        return {"ok": False, "reason": str(exc)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def day(offset):
    return (DAY0 + dt.timedelta(days=offset)).isoformat()


def allocation(n, proportions):
    sizes = {k: math.floor(n * p) for k, p in proportions.items()}
    order = sorted(proportions, key=lambda k: -(n * proportions[k] - sizes[k]))
    for k in order[:n - sum(sizes.values())]:
        sizes[k] += 1
    return sizes


def schedule(n, proportions, rng):
    out = [k for k, size in allocation(n, proportions).items() for _ in range(size)]
    rng.shuffle(out)
    return out


def slot(f):
    return f["holder"], f["scope"], f["entity"], f["attr"], tuple(sorted(f["condition"].items()))


def validate_ledger(facts):
    """A chain is a proposal's lifecycle; supersedes connects distinct versions."""
    seen, latest, proposals = {}, {}, {}
    for f in facts:
        if f["id"] in seen:
            raise ValueError("duplicate atom")
        previous = seen.get(f.get("prev"))
        old = previous["status"] if previous else None
        if f.get("prev") and not previous:
            raise ValueError("missing predecessor")
        machine = TRANSITIONS
        if f.get("machine"):
            if f["attr"] not in RELATION_TYPES[f["machine"]] or f["value"] not in STRUCTURES[f["machine"]]["values"]:
                raise ValueError("native attribute/value type mismatch")
            edges = STRUCTURES[f["machine"]]["machine"]
            machine = {None: {edges[0][0]}}
            for a, b in edges:
                machine.setdefault(a, set()).add(b)
            for a, b in edges:
                machine.setdefault(b, set())
        if f["status"] not in machine.get(old, set()):
            raise ValueError(f"illegal transition {old} -> {f['status']}")
        if previous:
            if f.get("machine") != previous.get("machine") or f.get("links") != previous.get("links"):
                raise ValueError("lifecycle changed relation/machine")
            if latest[f["chain"]] != previous["id"] or slot(previous) != slot(f):
                raise ValueError("forked lifecycle or changed slot")
            if previous["chain"] != f["chain"] or previous["value"] != f["value"]:
                raise ValueError("lifecycle changed value/chain")
            if previous.get("supersedes") != f.get("supersedes"):
                raise ValueError("lifecycle changed supersedes")
            if previous["proposal"] != f["proposal"]:
                raise ValueError("lifecycle changed proposal name")
            if previous["event_time"] > f["event_time"] or previous["record_time"] > f["record_time"]:
                raise ValueError("backwards lifecycle")
        elif f["chain"] in latest:
            raise ValueError("duplicate chain root")
        if f["proposal"] in proposals and proposals[f["proposal"]] != f["chain"]:
            raise ValueError("duplicate proposal name")
        proposals[f["proposal"]] = f["chain"]
        if f["event_time"] > f["record_time"]:
            raise ValueError("record precedes event")
        sup = f.get("supersedes")
        if sup and (sup not in seen or slot(seen[sup]) != slot(f) or seen[sup]["chain"] == f["chain"]):
            raise ValueError("invalid supersession")
        if sup and seen[sup]["event_time"] >= f["event_time"]:
            raise ValueError("supersession must be later")
        if sup and seen[sup]["status"] not in EFFECTIVE:
            raise ValueError("supersession must reference an effective version")
        if sup and seen[sup]["record_time"] > f["record_time"]:
            raise ValueError("supersession references an unknown record")
        if sup and f.get("replacement") != {k: seen[sup][k] for k in ("value", "event_time", "proposal")}:
            raise ValueError("rendered replacement relation disagrees with supersedes")
        if not sup and f.get("replacement"):
            raise ValueError("rendered replacement without supersedes")
        if f.get("statement") != core(f):
            raise ValueError("ledger statement drift")
        seen[f["id"]] = f
        latest[f["chain"]] = f["id"]
    return True


def validate_types(w):
    for f in w['facts']:
        try: allowed=vocabulary(w['ontology'],f['attr'])
        except KeyError as exc: raise ValueError('unknown attribute type') from exc
        if f['value'] not in allowed: raise ValueError('attribute value outside its domain')
        if f['attr'] in RELATION_TYPES[w['domain']] and f.get('machine')!=w['domain']:
            raise ValueError('native attribute requires its own machine')
        if f.get('machine') and f['attr'] not in RELATION_TYPES[f['machine']]:
            raise ValueError('machine does not belong to attribute')
    return True


def resolve(facts, selector, as_of, known_at, context=None, accepted=EFFECTIVE):
    """Bitemporal snapshot, strict holder/scope; a rollback restores the older live version.

    A proposal/veto never replaces an approved value. More specific applicable
    conditions override a general rule. No implicit personal -> team inheritance.
    """
    context = context or {}
    chains = {}
    for f in facts:
        if any(f.get(k) != v for k, v in selector.items()):
            continue
        if f["event_time"] > as_of or f["record_time"] > known_at:
            continue
        if not all(context.get(k) == v for k, v in f["condition"].items()):
            continue
        prior = chains.get(f["chain"])
        if prior is None or (f["event_time"], f["record_time"], f["seq"]) > (prior["event_time"], prior["record_time"], prior["seq"]):
            chains[f["chain"]] = f
    live = [f for f in chains.values() if f["status"] in accepted]
    # An active replacement hides all its ancestors, even when the ancestor
    # has a later lifecycle timestamp. Rolled-back/vetoed replacements do not.
    byid = {f["id"]: f for f in facts}
    byproposal = {f["proposal"]: f for f in facts if f["event_time"] <= as_of and f["record_time"] <= known_at}
    hidden = set()
    for f in live:
        current, visited = f, set()
        while current.get("supersedes"):
            ancestor = byid.get(current["supersedes"]) or byproposal.get(current.get("replacement", {}).get("proposal"))
            if ancestor is None:
                break
            if ancestor["chain"] in visited:
                raise ValueError("supersession cycle")
            visited.add(ancestor["chain"])
            hidden.add(ancestor["chain"])
            current = ancestor
    live = [f for f in live if f["chain"] not in hidden]
    if not live:
        return None
    specificity = max(len(f["condition"]) for f in live)
    live = [f for f in live if len(f["condition"]) == specificity]
    if len(live) > 1:
        raise ValueError("ambiguous live versions without supersedes")
    return live[0]


def core(f):
    if f.get('natural_surface'):
        policy = f['natural_surface'] if isinstance(f['natural_surface'],dict) else {}
        entity=f['entity']+(f" ({policy['version_date']})" if policy.get('version_date') else '')
        text = (f"On {f['event_time']}, {f['holder']}'s {f['scope']} {entity} had "
            f"{f['attr'].replace('_',' ')} {f['value']}; its status was {f['status'].replace('_',' ')}.")
        if policy.get('condition', True) and f['condition']:
            cond = ' and '.join(f"the {k} was {v}" for k,v in sorted(f['condition'].items()))
            text += f" This applied when {cond}."
        if policy.get('update', True):
            text += f" This was the {f['proposal']} update."
        if f.get('replacement'):
            if policy.get('replacement_named',True):
                text += f" It replaced the {f['replacement']['proposal']} update."
            else:
                text += f" It replaced the entry from {f['replacement']['event_time']}."
        if f.get('links'):
            text += ' It depended on ' + ', '.join(f['links']) + '.'
        return text
    cond = ", ".join(f"{k}={v}" for k, v in sorted(f["condition"].items())) or "always"
    text = (f"On {f['event_time']}, {f['holder']}'s {f['scope']} {f['entity']} "
            f"{f['attr']}={f['value']} was {f['status']} ({cond}). Decision {f['proposal']}.")
    if f.get("replacement"):
        # Name the predecessor proposal, without restating its value.
        # Otherwise the rollback alone leaks the restored answer and makes
        # the supposedly necessary predecessor message redundant.
        text += f" It replaces decision {f['replacement']['proposal']}."
    if "links" in f:
        text += " Links: " + (", ".join(f["links"]) or "none") + "."
    return text


def naturalize(w):
    """Change surface names only after structural sampling, preserving its RNG."""
    mapping = natural_names([name for f in w['facts'] for name in [f['entity']]+f.get('links',[])],w['domain'],w['seed'],w['world'])
    proposals=[name for f in w['facts'] for name in [f['proposal']]+([f['replacement']['proposal']] if f.get('replacement') else [])]
    mapping.update(natural_names(proposals,w['domain'],w['seed'],w['world'],'update',mapping.values()))
    w = rename_values(w,mapping)
    cases = {c['id']:c for c in w['cases']}
    by_entity = collections.defaultdict(list)
    value_chains=collections.defaultdict(set)
    chain_dates={}
    for f in w['facts']:
        by_entity[f['entity']].append(f)
        value_chains[slot(f)+(f['value'],)].add(f['chain'])
        chain_dates[f['chain']]=min(chain_dates.get(f['chain'],f['event_time']),f['event_time'])
    dated_chains=collections.defaultdict(set)
    for f in w['facts']:
        dated_chains[slot(f)+(f['value'],chain_dates[f['chain']])].add(f['chain'])
    for f in w['facts']:
        c = cases[f['source']['case']]
        peers = by_entity[f['entity']]
        # Keep qualifications wherever the query or a competing slot needs
        # them. Chain ambiguity is structural; no candidate labels are used.
        condition = bool(c['query'].get('context')) or len({tuple(sorted(x['condition'].items())) for x in peers}) > 1
        f['natural_surface'] = dict(condition=condition,
            update=c['qtype']=='update' or any(x['status']=='rolled_back' for x in peers) or
                len(dated_chains[slot(f)+(f['value'],chain_dates[f['chain']])])>1,
            holder_anchor=len({x['holder'] for x in peers}) > 1)
        if not f['natural_surface']['update'] and len(value_chains[slot(f)+(f['value'],)])>1:
            f['natural_surface']['version_date']=chain_dates[f['chain']]
    byid={f['id']:f for f in w['facts']}
    for f in w['facts']:
        if f.get('supersedes'):
            f['natural_surface']['replacement_named']=byid[f['supersedes']]['natural_surface']['update']
        f['statement'] = core(f)
    for c in w['cases']:
        c['question'] = question_text(c)
    return w


def add_version(w, case, holder, scope, entity, attr, value, condition=None,
                when=None, statuses=("proposed", "approved", "executed"), supersedes=None):
    chain = f"c{len(w['facts'])}"
    occupied = {f.get("proposal") for f in w["facts"]}
    salt = 0
    while True:
        key = digest([w.get("world"), w.get("seed"), case["id"], chain, holder, scope, entity, attr, value, salt])
        proposal = "".join(chr(65 + int(key[i:i + 2], 16) % 26) for i in range(0, 16, 2))
        if proposal not in occupied:
            break
        salt += 1
    prev, result = None, []
    when = case["offset"] - 9 if when is None else when
    for step, status in enumerate(statuses):
        fid = f"f{len(w['facts'])}"
        f = dict(id=fid, chain=chain, proposal=proposal, seq=len(w["facts"]), holder=holder, scope=scope,
                 entity=entity, attr=attr, value=value, condition=condition or {}, status=status,
                 event_time=day(when + step), record_time=case["record_time"],
                 source={"case": case["id"], "kind": "ledger"}, prev=prev, supersedes=supersedes)
        if supersedes:
            old = next(f for f in w["facts"] if f["id"] == supersedes)
            f["replacement"] = {k: old[k] for k in ("value", "event_time", "proposal")}
        f["statement"] = core(f)
        w["facts"].append(f)
        result.append(fid)
        prev = fid
    return result


def solve(w, query):
    """Return necessary ledger atoms and a deterministic answer, never ask an LLM."""
    facts = w["facts"]
    if query.get("date_window") is not None:
        lo, hi = query["date_window"]
        if dt.date.fromisoformat(lo) > dt.date.fromisoformat(hi):
            raise ValueError("reversed record date window")
        # Filter the entire evidence universe before snapshot resolution, native
        # edge traversal, supersession and reversal collection. Inclusive RECORD
        # dates match the dates visible in candidate document headers.
        facts = [f for f in facts if lo <= f["record_time"] <= hi]
    selected = []
    for request in query["requests"]:
        f = resolve(facts, request, query["as_of"], query["known_at"], query.get("context"),
                    tuple(query.get("accepted", EFFECTIVE)))
        if f is None:
            raise ValueError(f"unanswerable request: {request}")
        selected.append(f)
    if query.get('domain_program'):
        domain=query['domain_program']['domain']
        root=selected[0]; selected=[]; visited=set()
        def visit(f):
            if f['entity'] in visited:return
            visited.add(f['entity']);selected.append(f)
            for entity in f.get('links',[]):
                matches=[]
                for attr in RELATION_TYPES[domain]:
                    row=resolve(facts,dict(holder=root['holder'],scope=root['scope'],entity=entity,attr=attr),
                        query['as_of'],query['known_at'],query.get('context'))
                    if row is not None:matches.append(row)
                if len(matches)!=1:raise ValueError('missing or ambiguous typed relation edge endpoint: '+entity)
                visit(matches[0])
        visit(root)
        answer=solve_relations(domain,selected,query['domain_program'])
        if query['operation']=='temporal':
            ordered=sorted(selected,key=lambda f:(f['event_time'],f['entity'],f['attr']))
            span=(dt.date.fromisoformat(ordered[-1]['event_time'])-dt.date.fromisoformat(ordered[0]['event_time'])).days
            answer+='; Event order: '+'; '.join(f"{f['entity']} / {f['attr']}={f['value']} on {f['event_time']}" for f in ordered)+f'; interval {span} days'
        return {'atom_ids':[f['id'] for f in selected],'answer':answer}
    if query.get("include_reversals"):
        for current in list(selected):
            reversals = [f for f in facts if slot(f) == slot(current) and f["status"] == "rolled_back"
                         and f.get("replacement", {}).get("proposal") == current["proposal"]
                         and f["event_time"] <= query["as_of"] and f["record_time"] <= query["known_at"]]
            selected.extend(reversals)
    if query["operation"] == "multihop":
        # A route identifies the person; the second hop is computed from that value.
        target = resolve(facts, {"holder": selected[0]["value"], "scope": "personal",
                         "entity": query["entity"], "attr": query.get("location_attr", "location")},
                         query["as_of"], query["known_at"], query.get("context"))
        if target is None:
            raise ValueError("missing second hop")
        selected.append(target)
    if query["operation"] == "recommendation":
        # Actual event requirement + person skill and availability, all necessary.
        need = selected[0]["value"]
        person = selected[1]["holder"]
        available = query.get("available_value", "available")
        cleared = query.get("clearance_value", "cleared")
        if any(f["holder"] != person for f in selected[1:]):
            raise ValueError("mixed recommendation candidates")
        eligible = (selected[1]["value"] == need and selected[2]["value"] == available
                    and all(f["value"] == cleared for f in selected[3:]))
        # Report every constraint, including failed ones. This keeps all rows
        # necessary and samples availability/permissions from their whole pool.
        answer = (f"Recommend {person}. " if eligible else f"Do not recommend {person}. ") + f"Required: {need}; "
        answer += "; ".join(f"{f['attr']}={f['value']}" for f in selected[1:])
    elif query["operation"] == "temporal":
        selected.sort(key=lambda f: (f["event_time"], f["seq"]))
        span = (dt.date.fromisoformat(selected[-1]["event_time"]) - dt.date.fromisoformat(selected[0]["event_time"])).days
        answer = "; ".join(f"{f['attr']}={f['value']} on {f['event_time']}" for f in selected) + f"; interval {span} days"
    elif query["operation"] in ("preference", "rule"):
        answer = "; ".join(f"Use {f['value']} for {f['attr']}" for f in selected)
    else:
        answer = "; ".join((f"Rollback of {f['value']} for {f['holder']} / {f['attr']}"
                             if f["status"] == "rolled_back" else
                             f"{f['holder']} / {f['scope']} / {f['attr']}: {f['value']}") for f in selected)
    return {"atom_ids": list(dict.fromkeys(f["id"] for f in selected)), "answer": answer}


def question_text(case):
    q = case["query"]
    req = q["requests"]
    who = req[0]["holder"]
    entity = q["entity"]
    attrs = ", ".join(r["attr"] for r in req)
    scope = req[0]["scope"]
    candidate = req[1]["holder"] if q["operation"] == "recommendation" else who
    texts = {
        "direct": f"What are {who}'s {scope} {attrs} for {entity}?",
        "preference": f"Prepare {entity} for {who} using their personal preferences for {attrs}. What should be used?",
        "rule": f"Which of {who}'s team rules for {attrs} should be applied to {entity} under the stated conditions?",
        "update": f"After the decisions and reversals, what are {who}'s effective {scope} {attrs} for {entity}?",
        "multihop": f"Trace {who}'s team assignment and milestones for {entity}, and locate the assigned person.",
        "aggregation": f"Summarize {who}'s effective {scope} outcomes for only these attributes: {attrs}, for {entity}.",
        "temporal": f"In event order, what were {who}'s effective {scope} milestones for only these attributes: {attrs}, for {entity}, and how many days separate the first and last events?",
        "recommendation": f"Should {candidate} handle {entity} under {who}'s team requirements? Report the required skill, this candidate's skill, availability and the listed clearance constraints, then give the recommendation.",
    }
    text = texts[q["operation"]]
    if not q.get('domain_program'):
        text += ' Evaluate only these scoped attributes: ' + '; '.join(
            f"{r['holder']}'s {r['scope']} {r['attr']} for {r['entity']}" for r in req) + '.'
    if q.get("domain_program"):
        program = q["domain_program"]
        text = (f"For {who}, evaluate {program['relation']} using these scoped relations: "
                + "; ".join(f"{r['holder']}'s {r['scope']} {r['attr']} for {r['entity']}" for r in req)
                + ". " + NATIVE_TASKS[q["operation"]] + " " + INSTRUCTIONS[program["domain"]]
                + " At every linked entity, use only the unique effective native attribute for the root's holder and scope under the stated conditions and snapshot."
                + " Native attribute types: " + ", ".join(RELATION_TYPES[program['domain']]) + "."
                + (f" Sample measurement: {program['sample']}." if "sample" in program else "")
                + (f" Quorum: {program['quorum']}." if "quorum" in program else ""))
    # Snapshot time is semantically necessary, but is not an explicit date *window*.
    if q["operation"] in ("temporal", "update") or case.get("explicit_as_of"):
        text += f" Evaluate the state effective on {q['as_of']}, using records available by {q['known_at']}."
    else:
        text += " Use the effective decisions documented for this task."
    if q.get("context"):
        text += " Conditions: " + ", ".join(f"{k}={v}" for k, v in sorted(q["context"].items())) + "."
    window = q.get("date_window", case.get("date_window"))
    if window:
        text += f" Use records between {window[0]} and {window[1]}."
    if case["perspective"] == "self":
        text = f"I ({case['asker']}) reported on this. " + text
    elif case["perspective"] == "participant":
        text = f"I ({case['asker']}) took part in the discussion. " + text
    return text


def rename_values(value, mapping):
    """Bijective renaming; compile once, preferring complete longer names."""
    if not mapping:return value
    pattern=re.compile(r"\b(?:"+'|'.join(map(re.escape,sorted(mapping,key=len,reverse=True)))+r")\b")
    def visit(item):
        if isinstance(item,str):return pattern.sub(lambda m:mapping[m.group()],item)
        if isinstance(item,list):return [visit(x) for x in item]
        if isinstance(item,dict):return {key:visit(x) for key,x in item.items()}
        return item
    return visit(value)


def relation_version(w, case, holder, scope, entity, attr, value, condition, when, domain, links, supersedes=None, terminal=None):
    spec = STRUCTURES[domain]
    if attr not in RELATION_TYPES[domain] or value not in spec["values"]:
        raise ValueError("native attribute/value type mismatch")
    path = [spec['machine'][0][0]]
    terminal = terminal or spec['live']
    while path[-1] != terminal:
        options = [b for a, b in spec['machine'] if a == path[-1]]
        # Prefer the unique path to the requested live/inactive state.
        def reaches(node, goal):
            return node == goal or any(reaches(b, goal) for a, b in spec['machine'] if a == node)
        path.append(next(b for b in options if reaches(b, terminal)))
    ids = add_version(w, case, holder, scope, entity, attr, value, condition,
                      when=when - len(path) + 1, statuses=path, supersedes=supersedes)
    byid = {f['id']: f for f in w['facts']}
    for fid in ids:
        byid[fid].update(machine=domain, links=links)
        byid[fid]['statement'] = core(byid[fid])
    return ids[-1]


def install_domain_relations(w, rng):
    """Domain-specific multi-hop operands and real native lifecycles replace scalar routes."""
    domain, ont = w['domain'], w['ontology']
    spec = STRUCTURES[domain]
    attr = spec['relation']
    for relation in RELATION_TYPES[domain]:
        ont['attributes'][relation] = spec['values']
    w['value_vocabulary'] = all_values(ont)
    for case in w['cases']:
        if not case.get('native_program',case['qtype'] in ('multihop','aggregation','temporal')):
            continue
        count = case['expected_evidence']
        graph = parents(domain, count)
        depths={0:0}
        def set_depth(i):
            for child in graph[i]:depths[child]=depths[i]+1;set_depth(child)
        set_depth(0)
        original = next(f for f in w['facts'] if f['id'] == case['targets'][0])
        entities = [original['entity'] + ' ' + ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8)) for _ in range(count)]
        case['targets'] = []
        case.pop('state_support', None)
        case['query']['requests'] = []
        for i in range(count):
            attr = RELATION_TYPES[domain][min(depths[i], 2)]
            value = rng.choice(spec['values'])
            old_value = rng.choice([v for v in spec['values'] if v != value])
            links = [entities[j] for j in graph[i]]
            when = case['offset'] - rng.randint(7, 12)
            prior = relation_version(w, case, original['holder'], original['scope'], entities[i], attr,
                old_value, case['query']['context'], when - 8, domain, links)
            fid = relation_version(w, case, original['holder'], original['scope'], entities[i], attr,
                value, case['query']['context'], when, domain, links, supersedes=prior)
            case['targets'].append(fid)
            case['negative_atoms'].append(prior)
            # Explicit inapplicable/newer and inactive lifecycles in every native ontology.
            inactive = next(b for a, b in spec['machine'] if b != spec['live'] and not any(x == b for x, y in spec['machine']))
            for inactive in sorted({b for a, b in spec['machine'] if b != spec['live'] and not any(x == b for x, y in spec['machine'])}):
                case['negative_atoms'].append(relation_version(w, case, original['holder'], original['scope'], entities[i], attr,
                    rng.choice(spec['values']), case['query']['context'], when + 2, domain, links, supersedes=prior, terminal=inactive))
            case['negative_atoms'].append(relation_version(w, case, original['holder'], original['scope'], entities[i], attr,
                rng.choice(spec['values']), {ont['context_key']: next(v for v in (ont['routine'], ont['emergency']) if v != case['query']['context'][ont['context_key']])},
                when + 3, domain, links))
            case['query']['requests'].append(dict(holder=original['holder'], scope=original['scope'], entity=entities[i], attr=attr))
        case['query']['requests']=case['query']['requests'][:1]
        case['query']['domain_program'] = dict(domain=domain, relation=spec['relation'], constraint=spec['constraint'])
        if domain == 'laboratory': case['query']['domain_program']['sample'] = rng.randint(2, 10)
        if domain == 'school': case['query']['domain_program']['quorum'] = rng.randint(1, count)
        case['question'] = question_text(case)
        for f in w['facts']:
            if f['source']['case'] == case['id'] and f['attr'] in RELATION_TYPES[domain]:
                f['source']['reporter'] = 'Assistant' if case['assistant_only'] else case['reporter']
                f['confirmation'] = False


def prune_unqueried_slots(w):
    """Replace the old negative bank, retaining every version of every queried slot."""
    byid={f['id']:f for f in w['facts']}
    needed={slot(byid[fid]) for c in w['cases'] for fid in solve(w,c['query'])['atom_ids']}
    kept=[f for f in w['facts'] if slot(f) in needed]
    mapping={f['id']:f'f{i}' for i,f in enumerate(kept)}
    chains={}
    for i,f in enumerate(kept):
        old=f['id'];chains.setdefault(f['chain'],f'c{i}')
        f.update(id=mapping[old],seq=i,chain=chains[f['chain']],
            prev=mapping.get(f.get('prev')),supersedes=mapping.get(f.get('supersedes')))
    w['facts']=kept
    for c in w['cases']:
        for key in ('targets','negative_atoms','state_support','lifecycle_atoms'):
            if key in c:c[key]=[mapping[fid] for fid in c[key] if fid in mapping]
    validate_ledger(w['facts'])


def install_competition_panels(w, rng):
    """Random balanced partitions, sampled dimension combinations and 1-4 record documents.

    All sixteen anchors, including the queried one, share the same document law.
    No Cartesian cube is enumerated. Native distractors stay in native attribute types.
    """
    prune_unqueried_slots(w)
    ont = w['ontology']
    for case in w['cases']:
        case['competition_panels'], case['panel_subtypes'], case['competition_documents'] = {}, {}, {}
        required = solve(w, case['query'])['atom_ids']
        for fid in required:
            byid = {f['id']: f for f in w['facts']}
            target = byid[fid]
            native = target.get('machine')
            if native:
                choices=list(RELATION_TYPES[native])
            elif target['attr'] in DOMAINS[w['domain']]['attributes']:
                # The question samples these eight attributes uniformly. Do not
                # make low-prior generic attributes an easy distractor marker.
                choices=list(DOMAINS[w['domain']]['attributes'])
            else:
                choices=[ont[k] for k in ('requirement','skill','availability')]
                choices += [ont['clearance']+' / '+k for k in ('equipment','room','schedule','archive','materials')]
                if target['attr'] not in choices:choices.append(target['attr'])
            attrs = [target['attr']] + rng.sample([a for a in choices if a != target['attr']],
                min(len(choices)-1, rng.choice([1, 2, 3])))
            people=NAMES[:2] if case['carrier']=='dyad' else NAMES
            holders = [target['holder']] + rng.sample([n for n in people if n != target['holder']],rng.randint(1,min(3,len(people)-1)))
            entities = [target['entity']]
            for _ in range(rng.choice([1, 2, 3])):
                name = f"{w['project']} {rng.choice(['Orion','Maple','River','Atlas'])} " + ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8))
                if native: name += ' ' + ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8))
                entities.append(name)
            conditions = [target['condition'], {ont['context_key']: next(v for v in (ont['routine'],ont['emergency'])
                if (target['condition'] or case['query']['context']).get(ont['context_key']) != v)}]
            scopes = [target['scope'], 'team' if target['scope'] == 'personal' else 'personal']
            def partition(values):
                # Equal or near equal marginal counts; category labels are randomized.
                codes = [i % len(values) for i in range(16)]
                rng.shuffle(codes)
                chosen = codes[0]
                labels = [v for v in values[1:]]; rng.shuffle(labels)
                mapping = {chosen: values[0]}
                mapping.update(zip([i for i in range(len(values)) if i != chosen],labels))
                return [mapping[i] for i in codes]
            occupied={(f['entity'],f['holder'],f['scope'],f['attr'],tuple(sorted(f['condition'].items()))) for f in w['facts']}
            for attempt in range(40000):
                # An existing dense ledger can exhaust a small sampled layout.
                # Grow only after the original rejection budget, up to sixteen
                # distinct entities; then every candidate has a free anchor.
                if attempt and attempt%10000==0:
                    wanted=min(8 if case.get('date_window') else 16,2*len(entities))
                    while len(entities)<wanted:
                        name=f"{w['project']} {rng.choice(['Orion','Maple','River','Atlas'])} "+''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8))
                        if native:name+=' '+''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8))
                        if name not in entities:entities.append(name)
                dims = [partition(v) for v in (entities,holders,scopes,attrs,conditions)]
                anchors = list(zip(*dims))
                keys = [(e,h,s,a,tuple(sorted(c.items()))) for e,h,s,a,c in anchors]
                # One attribute per candidate projection prevents a queried
                # entity from being the only center with a missing attribute peer.
                projections=[key[:3]+(key[4],) for key in keys]
                if len(set(projections))!=16:continue
                if max(collections.Counter((key[0],key[3]) for key in keys).values())>5:continue
                if case.get('date_window'):
                    wanted_attrs={byid[x]['attr'] for x in required if byid[x]['entity']==target['entity']}
                    if not any(key[0]==target['entity'] and key[3] in wanted_attrs for key in keys[1:]):continue
                # Target must never be copied into a second candidate or another required slot.
                requested = {(byid[x]['entity'],byid[x]['holder'],byid[x]['scope'],byid[x]['attr'],tuple(sorted(byid[x]['condition'].items()))) for x in required}
                if len(set(keys)) == 16 and not (set(keys[1:]) & (requested | occupied)): break
            else: raise ValueError('cannot sample distinct competition anchors')
            capacity=max(1,min(4,2500//(len(target['statement'])+150)))
            lengths = [rng.randint(1,capacity) for _ in range(16)]
            kinds = ['other_time','vetoed','condition_or_future']
            # At least three negative documents carry the three audited contrasts.
            forced = rng.sample(range(1,16),3)
            documents=[]; subtype_lists=[]
            when = (dt.date.fromisoformat(target['event_time'])-DAY0).days
            prior_time = (dt.date.fromisoformat(byid[target['supersedes']]['event_time'])-DAY0).days if target.get('supersedes') else when-8
            def make(h,scope,entity,attr,value,cond,t,kind='current',sup=None,links=None):
                if native:
                    spec=STRUCTURES[native]
                    terminal=next(b for a,b in spec['machine'] if b != spec['live'] and not any(x==b for x,y in spec['machine'])) if kind=='vetoed' else None
                    return relation_version(w,case,h,scope,entity,attr,value,cond,t,native,links or [],supersedes=sup,terminal=terminal)
                prior_status=byid[target['supersedes']]['status'] if target.get('supersedes') else 'executed'
                endpoint_status=prior_status if kind=='prior' else target['status']
                states=('proposed','vetoed') if kind=='vetoed' else ('proposed','approved') if endpoint_status=='approved' else ('proposed','approved','executed')
                if kind=='current' and target['status']=='rolled_back': states+=('rolled_back',)
                if kind=='condition_or_future' and case['qtype']=='update': states=('proposed',)
                return add_version(w,case,h,scope,entity,attr,value,cond,when=t-len(states)+1,statuses=states,supersedes=sup)[-1]
            for i,(entity,h,scope,attr,cond) in enumerate(anchors):
                links=[]
                if native:
                    old_prefix,new_prefix=target['entity'].rsplit(' ',1)[0],entity.rsplit(' ',1)[0]
                    links=[x.replace(old_prefix,new_prefix,1) for x in target['links']]
                values=vocabulary(ont,attr)
                if i==0: current=target
                else:
                    value=rng.choice(values)
                    prior=make(h,scope,entity,attr,rng.choice([v for v in values if v!=value]),cond,prior_time,kind='prior',links=links)
                    mid=make(h,scope,entity,attr,value,cond,when,sup=prior,links=links)
                    current=w['facts'][-1]
                document=[current['id']]; tags=[]
                count=lengths[i]-1
                if i in forced: count=max(1,count)
                for j in range(count):
                    kind=kinds[forced.index(i)] if i in forced and j==0 else rng.choice(kinds)
                    earlier=current.get('prev') if current['status']=='rolled_back' else current.get('supersedes')
                    if kind=='other_time' and earlier and earlier not in required:
                        endpoint=earlier
                    else:
                        if kind=='other_time': kind='vetoed'
                        t=case['offset'] if kind=='condition_or_future' else min(case['offset']-1,when+2)
                        endpoint=make(h,scope,entity,attr,rng.choice(values),cond,t,kind,
                            sup=(current['id'] if current['status'] in EFFECTIVE else current.get('supersedes')) if kind=='condition_or_future' else None,links=links)
                    if endpoint not in document: document.append(endpoint);tags.append(kind)
                documents.append(document);subtype_lists.append(tags)
                people=NAMES[:2] if case['carrier']=='dyad' else NAMES
                speaker=('Assistant' if case['assistant_only'] else case['reporter']) if i==0 else (
                    'Assistant' if case['assistant_only'] else h if case['reporter']==target['holder'] else rng.choice([n for n in people if n!=h]))
                for endpoint in document:
                    f=next(f for f in w['facts'] if f['id']==endpoint)
                    f['source']['reporter']=speaker
                    f['confirmation']=target.get('confirmation',False) if f['status']==target['status'] else f['status']=='approved'
            # Preserve v4's ledger and RNG stream; carrier layout only chooses
            # which fixed assertions share a visible message. Never resample facts
            # or category partitions just because a carrier needs shorter prose.
            case.setdefault('competition_anchors',{})[fid]=[document[0] for document in documents]
            if case['carrier'] in ('group','dyad','assistant'):
                limit=1 if case['carrier'] in ('group','dyad') else max(1,min(2,650//(len(target['statement'])+100)))
                for i,document in enumerate(documents):
                    if limit==1:
                        documents[i]=[document[1] if i in forced else document[0]]
                        subtype_lists[i]=[kinds[forced.index(i)]] if i in forced else []
                    else:
                        documents[i]=document[:limit]
                        subtype_lists[i]=subtype_lists[i][:limit-1]
            case['competition_documents'][fid]=documents
            case['competition_panels'][fid]=[x for doc in documents for x in doc]
            case['panel_subtypes'][fid]=subtype_lists
            case['negative_atoms'].extend(x for doc in documents for x in doc if x!=fid)


def make_plan(world=0, n_questions=80, seed=0, split="train", project=None,
              domain=None, domain_splits=None, revision=3):
    if n_questions < 8:
        raise ValueError("use at least 8 questions for a meaningful quota plan")
    rng = random.Random(f"{seed}:{world}")
    if domain_splits is not None:
        validate_domain_splits(domain_splits)
    allowed = list(DOMAINS) if domain_splits is None else domain_splits[split]
    if not allowed or any(d not in DOMAINS for d in allowed):
        raise ValueError("empty/unknown domain split")
    if domain is not None and domain not in allowed:
        raise ValueError("domain excluded by domain split")
    domain = domain or rng.choice(allowed)
    ont = DOMAINS[domain]
    w = dict(version=3, revision=revision, world=world, seed=seed, split=split,
             ledger_id=f"v3:{seed}:{world}", project=project or ont["entity"], domain=domain,
             ontology=copy.deepcopy(ont), value_vocabulary=all_values(ont),
             domain_splits=domain_splits, name=NAMES[0], team=list(NAMES),
             facts=[], cases=[], rounds=[], verified=False)
    carriers = schedule(n_questions, CARRIERS, rng)
    types = schedule(n_questions, QTYPES, rng)
    bins = schedule(n_questions, EVIDENCE_V3 if revision >= 3 else EVIDENCE, rng)
    counts = [1 if b == "one" else 2 if b == "two" else rng.randint(3, 5) if b == "three_five" else rng.randint(6, 8) for b in bins]
    minimum = {"multihop": 2, "aggregation": 2, "temporal": 2, "recommendation": 3, "update": 2}
    if revision >= 3:
        minimum.update(multihop=3, aggregation=3, temporal=3)
    native_indices={i for i,t in enumerate(types) if t in ('multihop','aggregation','temporal')} if revision>=3 else set()
    extra=max(0,math.ceil(.4*n_questions)-len(native_indices)) if revision>=3 else 0
    if extra: native_indices.update(rng.sample([i for i,t in enumerate(types) if t=='direct'],extra))
    def need(k): return 3 if k in native_indices else minimum.get(types[k],1)
    high=sum(need(i)>=3 for i in range(n_questions))
    short=max(0,high-sum(c>=3 for c in counts))
    for i in [i for i,c in enumerate(counts) if c==2][:short]: counts[i]=3
    ordered = sorted(range(n_questions), key=lambda k: -need(k))
    available, chosen = sorted(counts), {}
    for k in ordered:
        eligible = [x for x in available if x >= need(k)]
        if not eligible:
            raise ValueError("question count too small for joint quotas")
        count = rng.choice(eligible)
        available.remove(count)
        chosen[k] = count
    dated = set(rng.sample(range(n_questions), round(.15 * n_questions)))
    ua = [k for k, c in enumerate(carriers) if c == "assistant"]
    assistant_only = set(rng.sample(ua, round(len(ua) / 4)))
    # Balance both reporter=holder and reporter=asker WITHIN every task type.
    # Assistant-only evidence is necessarily in the 'other' half.
    own_asker, own_holder = set(), set()
    for type_index,kind in enumerate(QTYPES):
        indices = [k for k, t in enumerate(types) if t == kind]
        human = [k for k in indices if k not in assistant_only]
        if revision>=3:
            # Odd within-type quotas alternate across independent ledgers, rather
            # than accumulating a binomial imbalance in one-question task cells.
            phase,domain_phase=divmod(world,len(DOMAINS))
            n_asker=len(indices)//2+((phase+domain_phase+type_index+seed)%2 if len(indices)%2 else 0)
            n_holder=len(indices)//2+((phase//2+domain_phase+type_index+seed//2)%2 if len(indices)%2 else 0)
        else:
            n_asker = len(indices) // 2 + (rng.randrange(2) if len(indices) % 2 else 0)
            n_holder = len(indices) // 2 + (rng.randrange(2) if len(indices) % 2 else 0)
        own_asker.update(rng.sample(human, min(len(human), n_asker)))
        own_holder.update(rng.sample(human, min(len(human), n_holder)))
    perspectives = collections.Counter()
    used_entities = set()
    def entity_name():
        while True:
            token = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(8))
            name = f"{w['project']} {rng.choice(['Orion', 'Maple', 'River', 'Atlas'])} {token}"
            if name not in used_entities:
                used_entities.add(name)
                return name
    for k in range(n_questions):
        carrier, kind, count = carriers[k], types[k], chosen[k]
        perspective = PERSPECTIVES[perspectives[carrier] % 3] if carrier in ("group", "dyad") else "external"
        perspectives[carrier] += 1
        participants = NAMES[:2] if carrier == "dyad" else NAMES
        reporter = rng.choice(participants)
        other = rng.choice([n for n in participants if n != reporter])
        holder = reporter if k in own_holder else other
        asker = reporter if k in own_asker else other
        case = dict(id=f"task{k}", offset=k * 3,
                    record_time=day(k * 3 + (2 if carrier == "record" else 0)), carrier=carrier,
                    qtype=kind, native_program=k in native_indices, perspective=perspective, asker=asker, reporter=reporter,
                    evidence_speaker_is_asker=k in own_asker,
                    evidence_speaker_is_holder=k in own_holder,
                    assistant_only=k in assistant_only, targets=[], negative_atoms=[],
                    expected_evidence=count, date_window=[day(k * 3 + (2 if carrier == "record" else 0))] * 2 if k in dated else None)
        entity = entity_name()
        mode = rng.choice([ont["routine"], ont["emergency"]])
        other_mode = ont["emergency"] if mode == ont["routine"] else ont["routine"]
        context = {ont["context_key"]: mode}
        query = dict(operation=kind, entity=entity, as_of=day(k * 3 - 1), known_at=case["record_time"],
                     context=context, requests=[], location_attr=ont["location"],
                     available_value=ont["available"], clearance_value=ont["cleared"])
        if case["date_window"]:
            query["date_window"] = list(case["date_window"])
        case["query"] = query
        scope = ("personal" if kind == "preference" else "team" if kind in ("rule", "recommendation") else rng.choice(["team", "personal"])) if revision >= 3 else ("team" if kind in ("rule", "recommendation", "multihop") else "personal")
        attrs = rng.sample(list(ont["attributes"]), count)
        values = [rng.choice(vocabulary(ont, attr)) for attr in attrs]
        if kind == "multihop":
            attrs[0], values[0] = "coordinator", rng.choice(NAMES)
        elif kind == "recommendation":
            attrs[:3] = [ont["requirement"], ont["skill"], ont["availability"]]
            skill = rng.choice(ont["skills"])
            values[:3] = [skill, rng.choice(ont["skills"]), rng.choice([ont["available"], ont["unavailable"]])]
            permissions = ["equipment", "room", "schedule", "archive", "materials"]
            rng.shuffle(permissions)
            for extra in range(3, count):
                attrs[extra], values[extra] = ont["clearance"] + " / " + permissions[extra - 3], rng.choice([ont["cleared"], ont["uncleared"]])
        for j in range(count - 1 if kind == "update" else count):
            if kind == "multihop" and j == count - 1:
                h, s, attr, val = values[0], "personal", ont["location"], rng.choice(ont["places"])
            else:
                h = other if kind == "recommendation" and j >= 1 else holder
                s = "personal" if kind == "recommendation" and j >= 1 else scope
                attr, val = attrs[j], values[j]
            req = dict(holder=h, scope=s, entity=entity, attr=attr)
            condition = context if revision >= 3 or kind == "rule" else {}
            # Every target slot has a genuine older value, not just the first
            # attribute. Conditional hard negatives are matched to the group's
            # actual positive atom in the adapter.
            old_value = rng.choice([v for v in vocabulary(ont, attr) if v != val])
            terminal = k * 3 - rng.randint(5, 14)
            supersedes = add_version(w, case, h, s, entity, attr, old_value, condition,
                                     when=terminal - rng.randint(4, 9) - 2)[-1]
            case["negative_atoms"].append(supersedes)
            approval = rng.random() < .12
            states = ("proposed", "approved") if approval else ("proposed", "approved", "executed")
            ids = add_version(w, case, h, s, entity, attr, val, condition,
                              when=terminal - len(states) + 1, supersedes=supersedes, statuses=states)
            target = ids[-1]
            if approval:
                w["facts"][-1]["confirmation"] = True
            if kind == "update" and j == 0:
                query["include_reversals"] = True
                changed_value = rng.choice([v for v in vocabulary(ont, attr) if v != val])
                changed = add_version(w, case, h, s, entity, attr, changed_value, condition,
                                      when=k * 3 - 4, statuses=("proposed", "approved", "executed", "rolled_back"), supersedes=target)
                case["negative_atoms"].append(changed[0])
                case["state_support"] = [changed[-1]]
                case["lifecycle_atoms"] = changed[1:3]
            case["targets"].append(target)
            if not (kind == "multihop" and j == count - 1):
                query["requests"].append(req)
        for target_id in case["targets"]:
            target_fact = next(f for f in w["facts"] if f["id"] == target_id)
            pool = vocabulary(ont, target_fact["attr"])
            terminal = (dt.date.fromisoformat(target_fact["event_time"]) - DAY0).days
            other_holders = rng.sample([n for n in NAMES if n != target_fact["holder"]], 3)
            for j in range(4):
                h = other_holders[j] if j < 3 else target_fact["holder"]
                s = target_fact["scope"] if j < 3 else ("team" if target_fact["scope"] == "personal" else "personal")
                val = rng.choice(pool)
                prior = add_version(w, case, h, s, entity, target_fact["attr"],
                    rng.choice([v for v in pool if v != val]), target_fact["condition"], when=terminal - rng.randint(4, 9) - 2)[-1]
                states = ("proposed", "approved") if rng.random() < .12 else ("proposed", "approved", "executed")
                ids = add_version(w, case, h, s, entity, target_fact["attr"], val, target_fact["condition"], supersedes=prior,
                    when=terminal - len(states) + 1, statuses=states)
                case["negative_atoms"].append(ids[-1])
                third = next(n for n in NAMES if n not in (reporter, other))
                source_speaker = [asker, reporter, third if carrier == "group" else other, other][j]
                for fid in ids:
                    next(f for f in w["facts"] if f["id"] == fid)["source"]["reporter"] = source_speaker
            case["negative_atoms"].extend([
                add_version(w, case, target_fact["holder"], target_fact["scope"], entity, target_fact["attr"], rng.choice(pool), target_fact["condition"], when=terminal - 1, statuses=("proposed", "vetoed"), supersedes=target_fact["supersedes"])[-1],
                add_version(w, case, target_fact["holder"], target_fact["scope"], entity, target_fact["attr"], rng.choice(pool), {ont["context_key"]: other_mode}, when=terminal - 2)[-1],
                add_version(w, case, target_fact["holder"], target_fact["scope"], entity, target_fact["attr"], rng.choice(pool), target_fact["condition"], when=k * 3, statuses=("proposed",), supersedes=target_fact["supersedes"])[-1]])
        first = next(f for f in w["facts"] if f["id"] == case["targets"][0])
        for j in range(6):
            attr = rng.choice(list(ont["attributes"]))
            vals = vocabulary(ont, attr)
            val = rng.choice(vals)
            side_entity = entity_name()
            terminal = k * 3 - rng.randint(5, 14)
            prior = add_version(w, case, first["holder"], first["scope"], side_entity,
                attr, rng.choice([v for v in vals if v != val]), when=terminal - rng.randint(4, 9) - 2)[-1]
            states = ("proposed", "approved") if rng.random() < .12 else ("proposed", "approved", "executed")
            ids = add_version(w, case, first["holder"], first["scope"], side_entity,
                              attr, val, supersedes=prior, when=terminal - len(states) + 1, statuses=states)
            for fid in ids:
                f = next(f for f in w["facts"] if f["id"] == fid)
                f["source"]["reporter"] = ("Assistant" if case["assistant_only"] else reporter) if j < 4 else other
            case["negative_atoms"].append(ids[-1])
        # All fact styles share templates, including meaningful confirmations.
        # Speakers are source metadata, independent of the target holder.
        target_chains = {f["chain"] for f in w["facts"] if f["id"] in case["targets"] + case.get("state_support", [])}
        for f in w["facts"]:
            if f["source"]["case"] != case["id"]:
                continue
            if f["chain"] in target_chains:
                sp = "Assistant" if case["assistant_only"] else reporter
            else:
                sp = rng.choice([asker, reporter, other])
            f["source"].setdefault("reporter", sp)
            f.setdefault("confirmation", f["status"] == "approved" or
                         (rng.random() < .12 and f["status"] == "executed"))
        case["question"] = question_text(case)
        w["cases"].append(case)
    if revision >= 3:
        install_domain_relations(w, rng)
        for case in w["cases"]:
            if case["qtype"] != "update":
                case["query"]["as_of"] = day(case["offset"] - 4)
        install_competition_panels(w, rng)
        for case in w['cases']:
            case['explicit_as_of'] = True
            case['question'] = question_text(case)
    identities = list(NAMES)
    rng.shuffle(identities)
    w = rename_values(w, dict(zip(NAMES, identities)))
    if revision >= 3:
        w = naturalize(w)
    validate_ledger(w["facts"])
    validate_types(w)
    for case in w["cases"]:
        solve(w, case["query"])
    w["plan_fingerprint"] = digest(w)
    return w

class MockPool:
    """Deterministic Pool.chat stub; used only when --mock was explicit."""
    def __init__(self, family="mock"):
        self.family = family
        self.calls = 0

    def chat(self, prompt, **kwargs):
        self.calls += 1
        payload = json.loads(prompt.split("\nPAYLOAD\n", 1)[1])
        if payload["task"] == "render":
            return json.dumps({"messages": [mock_message(m) for m in payload["messages"]]}), self.family
        if payload["task"] == "verify":
            return json.dumps({"ok": True, "no_extra_facts": True, "quotes": {
                fid: (m['text'] if m.get('anchor_sets') is not None else m['core'])
                for m in payload['messages'] if m['core']
                for fid in ([m['id']+':'+fid for fid in m['anchor_sets']] if m.get('anchor_sets') is not None else [m['id']])}}), self.family
        if payload["task"] == "gate":
            supported = {}
            for atom in payload["atoms"]:
                supported[atom["id"]] = [{"round": i, "quote": next((span for span in atom.get('evidence_spans',[]) if span in doc), atom['statement'] if atom['statement'] in doc else doc)}
                                         for i, doc in payload["documents"].items() if atom_present(atom,doc)]
            return json.dumps({"question_matches": True, "answer_correct": True, "supported": supported}), self.family
        raise ValueError(payload["task"])


def parse_model_response(text, task='render', expected_ids=None):
    """Decode one JSON value, tolerating fences and surrounding model prose.

    A truncated render array may retain complete rows. Incomplete strings/rows
    are never repaired; those IDs must be retried through the normal budget.
    """
    # Only remove a leading reasoning section. Markup inside JSON strings is
    # evidence and must survive byte-for-byte for verifier quotations.
    text = re.sub(r'^\s*<think>.*?</think>\s*', '', text, flags=re.S).strip()
    decoder = json.JSONDecoder()
    def rendered(value):
        rows=value if isinstance(value,list) else value['messages'] if 'messages' in value else [value]
        # Ignore unsolicited summaries and reply-only pseudo-rows. No reply is
        # attributed to a source message by position or guesswork.
        return {'messages':[r for r in rows if isinstance(r,dict) and isinstance(r.get('id'),str)
            and (isinstance(r.get('text'),str) or isinstance(r.get('turns'),list)) and (expected_ids is None or r['id'] in expected_ids)]}
    for match in re.finditer(r'[\[{]',text):
        try:
            value,_ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if task=='render':
            if isinstance(value,list):return rendered(value)
            if isinstance(value,dict) and isinstance(value.get('messages'),list):return rendered(value)
            if (isinstance(value,dict) and isinstance(value.get('id'),str)
                    and '[' not in text[:match.start()]):return rendered(value)
        elif isinstance(value,dict):return value
    if task=='render':
        start = re.search(r'(?:"messages"\s*:\s*)?\[\s*',text)
        if start:
            pos=start.end();rows=[]
            while pos<len(text):
                try:row,end=decoder.raw_decode(text,pos)
                except ValueError:break
                if not isinstance(row,dict):break
                rows.append(row);pos=end
                sep=re.match(r'\s*,\s*',text[pos:])
                if not sep:break
                pos+=sep.end()
            if rows:return rendered(rows)
    raise ValueError('no complete JSON response')


def verification_quote(vote, message, atom_id):
    """Tolerate common verifier quote maps without weakening quote checks."""
    quotes=vote.get('quotes',{})
    if not isinstance(quotes,dict):return None
    for key in (message['id']+':'+atom_id,atom_id,message['id']):
        value=quotes.get(key)
        if isinstance(value,str):return value
        if isinstance(value,dict) and isinstance(value.get(atom_id),str):return value[atom_id]
    return None


class ModelFamilies:
    def __init__(self, specs=None, mock=False, pools=None, timeout=120):
        if pools is not None:
            self.pools = pools
        elif mock:
            self.pools = [MockPool(f"mock-{i}") for i in range(3)]
        else:
            if not specs or len(specs) != 3 or not all(specs):
                raise ValueError("provide all --q27, --gem, --oss, or explicitly use --mock")
            # Local checkout first, same Pool protocol as world.py. Import lazily:
            # offline planning and tests require no model SDK or server.
            paths = [Path(__file__).resolve().parent.parent / "env" / "gen.py",
                     Path.home() / "data/rtm/env/gen.py"]
            gen_path = next((p for p in paths if p.exists()), None)
            if gen_path is None:
                raise ValueError("local gen.Pool implementation not found")
            spec = importlib.util.spec_from_file_location("crucible_v3_gen", gen_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Replace only this privately imported module's transport. No
            # hidden nested retries at gen.Pool's stock 600-second timeout.
            import urllib.request
            def post_once(url, body, timeout=timeout):
                request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read())
            module._post = post_once
            Pool = module.Pool
            self.pools = [Pool(s) for s in specs]
        if len(self.pools) != 3:
            raise ValueError("three families required")
        self.mock = mock
        self.identity = list(specs) if specs and any(specs) else [type(p).__name__ for p in self.pools]
        self.calls = collections.Counter()
        self.lock = threading.Lock()

    @staticmethod
    def response_schema(result, payload):
        if payload['task']=='render':
            rows = result if isinstance(result,list) else result.get('messages') if isinstance(result,dict) else None
            return isinstance(rows,list) and all(isinstance(r,dict) and isinstance(r.get('id'),str)
                and (isinstance(r.get('text'),str) or isinstance(r.get('turns'),list)) for r in rows)
        if not isinstance(result, dict):
            return False
        if payload["task"] == "reader":
            return isinstance(result.get("answer"), str) and isinstance(result.get("abstain"), bool)
        if payload["task"] == "verify":
            if not isinstance(result.get("ok"), bool):
                return False
            if result["ok"] is False:
                result.update(no_extra_facts=False, quotes={})
                return True
            if not isinstance(result.get("no_extra_facts"), bool):
                return False
            if result["no_extra_facts"] is False:
                result["quotes"] = {}
                return True
            quotes = result.get("quotes")
            return isinstance(quotes,dict) and all(
                all(isinstance(verification_quote(result,m,fid),str) for fid in m['anchor_sets'])
                if m.get('anchor_sets') is not None else isinstance(quotes.get(m['id']),str)
                for m in payload['messages'] if m['core'])
        if payload["task"] == "gate":
            if not all(isinstance(result.get(k), bool) for k in ("question_matches", "answer_correct")):
                return False
            if not result["question_matches"] or not result["answer_correct"]:
                result["supported"] = {}
                return True
            support = result.get("supported")
            if not isinstance(support, dict):
                return False
            for rows in support.values():
                if not isinstance(rows, list):
                    return False
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get("quote"), str):
                        return False
                    try:
                        if str(int(row["round"])) not in payload["documents"]:
                            return False
                    except (ValueError, TypeError, KeyError):
                        return False
        return True

    def ask(self, family, instructions, payload, validator=None, row_validator=None, failure_collector=None):
        if (payload['task']=='render' and payload.get('carrier')=='assistant'
                and len(payload['messages'])==1 and payload['messages'][0].get('length_contract')
                and not payload['messages'][0].get('render_side')):
            return render_assistant(self, family, payload, row_validator, failure_collector)
        family %= 3
        accepted={};original=payload;feedback={}
        is_render=payload['task']=='render' and bool(payload['messages'])
        reason='invalid response'
        failures={m['id']:[] for m in payload.get('messages',[]) if isinstance(m,dict) and 'id' in m}
        def record_failure(mid,reasons):
            failures[mid].extend(reasons)
            if failure_collector is not None: failure_collector.setdefault(mid,[]).extend(reasons)
        for attempt in range(3):
            if is_render:
                payload=dict(original,messages=[m for m in original['messages'] if m['id'] not in accepted])
                if attempt:
                    payload['retry_feedback']={m['id']:feedback.get(m['id'],'Missing or incomplete JSON row; return this ID only.') for m in payload['messages']}
            with self.lock:
                self.calls[f"family{family}:{payload['task']}"] += 1
            try:
                max_tokens=6000
                if payload["task"]=="render":
                    requested=sum(m.get('text_chars',0)+m.get('reply_chars',0) for m in payload['messages'])
                    max_tokens=max(max_tokens,math.ceil(requested/3)+1000)
                text = self.pools[family].chat(instructions + "\nPAYLOAD\n" + json.dumps(payload),
                                               max_tokens=max_tokens, temperature=.6 if payload["task"] == "render" else 0)[0]
                result = parse_model_response(text,payload['task'],{m['id'] for m in payload['messages']} if is_render else None)
                if is_render:
                    rows=result.get('messages',[])
                    counts=collections.Counter(r.get('id') for r in rows if isinstance(r,dict) and isinstance(r.get('id'),str))
                    expected={m['id']:m for m in payload['messages']}
                    for row in rows:
                        if not isinstance(row,dict) or not isinstance(row.get('id'),str):continue
                        mid=row['id']
                        if mid not in expected:continue
                        assemble_turns(row,expected[mid])
                        if counts[mid]==1 and (not row_validator or row_validator(row,expected[mid])) and isinstance(row.get('text'),str):
                            accepted[mid]=row
                        elif mid in expected:
                            source=expected[mid]
                            reasons=row.pop('_validation_failures',None) or message_failures(row,source)
                            if counts[mid]!=1:reasons.append('duplicate JSON id')
                            if not reasons:reasons=['local validation rejected row (unassigned fact or duplicate filler)']
                            record_failure(mid,reasons)
                            feedback[mid]=dict(reasons=reasons,observed_text_chars=len(row.get('text','')) if isinstance(row.get('text'),str) else 0,
                                target_text_chars=source.get('text_chars'),
                                observed_reply_chars=len(row.get('assistant_reply','')) if isinstance(row.get('assistant_reply',''),str) else 0,
                                target_reply_chars=source.get('reply_chars'),
                                missing_anchors=[span for spans in source.get('anchor_sets',{}).values() for span in spans
                                    if not anchors_present([span],row.get('text','') if isinstance(row.get('text'),str) else '')],
                                instruction='Rewrite this ID, fixing the listed reasons and preserving meaning and carrier form. Use natural length. Return one row for this ID.')
                    for mid in expected.keys()-counts.keys():
                        record_failure(mid,['missing JSON row'])
                        feedback[mid]=dict(reasons=['missing JSON row'])
                    if len(accepted)==len(original['messages']):
                        merged={'messages':[accepted[m['id']] for m in original['messages']]}
                        if not validator or validator(merged):return merged
                        accepted.clear()
                    reason='failed message validation'
                    continue
                if not self.response_schema(result, payload) or (validator and not validator(result)):
                    raise ValueError("response schema/core validation")
                return result
            except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                reason = type(exc).__name__+': '+str(exc)
                if is_render:
                    for source in payload['messages']:
                        record_failure(source['id'],[reason])
                        feedback[source['id']]=dict(reasons=[reason])
        if is_render and accepted:
            return {'messages':list(accepted.values()),'failed_ids':[m['id'] for m in original['messages'] if m['id'] not in accepted]}
        raise ValueError(f"{payload['task']} exhausted 3 attempts ({reason})")


def render_assistant(models, family, payload, row_validator=None, failure_collector=None):
    """Render each side independently and assemble the pair in the program.

    Successful sides are retained while the missing/invalid side uses its own
    three-attempt budget. The reply call sees only the rendered user turn.
    """
    m=payload['messages'][0]
    user=m.get('user_speaker',m['participants'][0]) if m['speaker']=='Assistant' else m['speaker']
    turns=[]
    for side,speaker in (('user',user),('assistant','Assistant')):
        factual=speaker==m['speaker']
        target=m['text_chars'] if factual else m.get('reply_chars',len(ASSISTANT_PROMPT))
        source=dict(id=m['id'],speaker=speaker,channel='record',render_side=side,
                    core=m['core'] if factual else '',
                    draft=m.get('draft',m['core']) if factual else ASSISTANT_PROMPT if side=='user' else '',
                    anchor_sets=m.get('anchor_sets',{}) if factual else {},
                    length_contract=m['length_contract'],text_chars=target,
                    style_key=m['id'] if factual or side=='user' else m['id']+':reply')
        if factual:
            source['status_alternatives']=status_alternatives(source)
        side_payload=dict(task='render',carrier='assistant',project=payload.get('project'),messages=[source])
        if side=='assistant':side_payload['user_turn']=turns[0]['text']
        elif factual and payload.get('context'):side_payload['context']=payload['context']
        if payload.get('avoid_filler'):side_payload['avoid_filler']=payload['avoid_filler']
        prompt=(render_instructions('assistant')+
                ' Render only the '+side+' side as one JSON object {id,text}. ')
        if not factual and side=='assistant':
            prompt+=('Respond helpfully to user_turn without repeating ledger facts, names, values or status anchors. '
                     'Discuss reading or wording in general terms. ')
            if m.get('core') and target>=900:
                prompt+='A longer explanation is requested for this reply; a few useful paragraphs are appropriate. '
        elif factual:
            prompt+='Paraphrase the core naturally; status anchors accept any phrase in status_alternatives. Preserve meaningful confirmations in the draft. '
        def valid_side(row, spec):
            # Tolerate a one-sided structured response as well as plain text.
            if isinstance(row.get('turns'),list):
                matching=[t for t in row['turns'] if isinstance(t,dict) and t.get('speaker')==speaker and isinstance(t.get('text'),str)]
                if len(matching)==1:row['text']=matching[0]['text']
            reasons=message_failures(row,spec)
            if not factual and side=='assistant' and any(
                    anchors_present([anchor],row.get('text','')) for spans in m.get('anchor_sets',{}).values() for anchor in spans):
                reasons.append('assistant reply contains fact anchor')
            if not reasons and row_validator and (factual or side=='assistant'):
                # Apply the caller's leakage/uniqueness checks inside this
                # side's retry budget as well. A first-side fact can be checked
                # with a temporary, fact-free reply that is never emitted.
                trial_turns=turns+[dict(speaker=speaker,text=row['text'])]
                if side=='user':
                    trial_turns.append(dict(speaker='Assistant',text='I can help read the wording carefully.'))
                trial=dict(id=m['id'],turns=trial_turns)
                assemble_turns(trial,m)
                if not row_validator(trial,m):
                    reasons.extend(trial.get('_validation_failures') or
                                   ['assembled side rejected (unassigned fact or duplicate filler)'])
            if reasons:row['_validation_failures']=reasons
            return not reasons
        result=models.ask(family,prompt,side_payload,row_validator=valid_side,failure_collector=failure_collector)
        turns.append(dict(speaker=speaker,text=result['messages'][0]['text']))
    row=dict(id=m['id'],turns=turns)
    assemble_turns(row,m)
    if row_validator and not row_validator(row,m):
        reasons=row.pop('_validation_failures',None) or message_failures(row,m)
        if failure_collector is not None:failure_collector.setdefault(m['id'],[]).extend(reasons)
        raise ValueError('assembled assistant validation: '+str(reasons))
    return dict(messages=[row])


def _message(w, case, fid, k, speaker=None, repeat=False, state_support=False):
    names = w["team"]
    facts = w.get("_render_facts") or {f["id"]: f for f in w["facts"]}
    f = facts.get(fid)
    speaker = speaker or (f["source"].get("reporter") if f else case["reporter"])
    channel = w.get("domain", "discussion") if k % 2 else "planning"
    thread = f"{case['id']}:thread{(k // 2) % 2}"
    if not f and k % 2:
        speaker = names[1] if case["reporter"] == names[0] else names[0]
    body = f["statement"] if f else ["Thanks, I can see the thread.", "I am scrolling back through the notes."][k % 2]
    # Complete clauses come first, within the formatter's first 200 characters.
    # Natural trailing context also stops unrelated fact-bearing rounds being
    # interpreted as short follow-ups by fmt.round_prev.
    if f:
        if f.get("confirmation"):
            body += (" I approve this decision and commit to carrying it out as recorded." if f["status"] == "approved"
                     else " I confirm that this decision was carried out as recorded.")
        elif case["qtype"] == "preference" and speaker != f["holder"]:
            body += f" {f['holder']} said they prefer {f['value']} in this setting."
        body += " I wanted to put that in writing while we are going through the notes, so everyone following this part of the discussion has the same context."
    expression = "equivalent_restatement" if repeat else "third_person_record" if case["carrier"] == "record" else "reported" if f and speaker != f["holder"] else "direct"
    return dict(id=f"m{len(w['rounds']) + k}", case=case["id"], speaker=speaker,
                recipients=[case["asker"]] if speaker == "Assistant" else list(names[:2] if case["carrier"] == "dyad" else names),
                channel=channel if case["carrier"] == "group" else case["carrier"], thread=thread,
                atoms=[fid] if fid else [], new_fact=bool(fid and not repeat), expression=expression,
                record_time=case["record_time"], core=f["statement"] if f else "", draft=body,
                meaningful_confirmation=bool(f and f.get("confirmation")),
                confirmation_kind=("approval_commitment" if f and f.get("confirmation") and f["status"] == "approved"
                                   else "execution_confirmation" if f and f.get("confirmation") else None),
                kind="state_support" if state_support else "repeat" if repeat else "ev" if f else "ack")


def block_ids(n, carrier=None):
    if carrier in ("record", "assistant"):
        return [j // 2 for j in range(n)]
    blocks = [j // 6 for j in range(n)]
    if n % 6 in (1, 2) and n > 6:
        for j in range(n - n % 6, n):
            blocks[j] -= 1
    return blocks


def render_input(message):
    """Authoring constraints only: no future-query roles or candidate labels."""
    fields=('id','speaker','recipients','channel','thread','expression','record_time',
            'core','draft','meaningful_confirmation','confirmation_kind','length_contract',
            'text_chars','reply_chars','max_chars','anchor_sets','participants','user_speaker')
    source={key:message[key] for key in fields if key in message}
    source['status_alternatives']=status_alternatives(message)
    return source


def status_alternatives(message):
    return {span:STATUS_SYNONYMS[span] for spans in message.get('anchor_sets',{}).values()
            for span in spans if span in STATUS_SYNONYMS}


def render(w, models, par=1, checkpoint_dir=None):
    if par < 1:
        raise ValueError("par must be positive")
    w = copy.deepcopy(w)
    w["rounds"] = []
    w["_render_facts"]={f["id"]:f for f in w["facts"]}
    jobs = []
    for ci, case in enumerate(w["cases"]):
        # All candidates go through the same renderer. No positive/negative label
        # or future question is sent to the model.
        ids = list(dict.fromkeys(case["targets"] + case.get("state_support", []) + case.get("lifecycle_atoms", []) + case["negative_atoms"]))
        # Interleave two local threads; each block below has 3-8 consecutive
        # messages. Topic blocks retain neighbouring discussion history.
        if case.get("competition_panels"):
            clustered = {fid for panel in case["competition_panels"].values() for fid in panel}
            local = [_message(w, case, fid, k, state_support=fid in case.get("state_support", []))
                for k, fid in enumerate(ids) if fid not in clustered]
            byid = {f["id"]: f for f in w["facts"]}
            for documents in case['competition_documents'].values():
                for document in documents:
                    first=document[0]
                    m=_message(w,case,first,len(local),state_support=first in case.get('state_support',[]))
                    f=byid[first];tail=m['draft'][len(f['statement']):]
                    m.update(atoms=document,core='\n'.join(byid[x]['statement'] for x in document),max_chars=5000)
                    m['draft']=m['core']+' Regarding the first entry,'+tail
                    local.append(m)
        else:
            local = [_message(w, case, fid, k, state_support=fid in case.get("state_support", [])) for k, fid in enumerate(ids)]
        # Equivalent restatement is no-new-fact but can be a positive. Meaningful
        # approval is a ledger transition and is never an automatic ack negative.
        if not case.get("competition_panels"):
            local.append(_message(w, case, case["targets"][0], len(local), repeat=True))
        n_empty = max(2, round(len(local) / 3) - 1)
        # 25% no-new-fact exchanges, including the explicit restatement.
        for _ in range(n_empty):
            local.append(_message(w, case, None, len(local)))
        # Shuffle complete discussions rather than placing every gold first.
        rng = random.Random(f"render:{w['seed']}:{w['world']}:{ci}")
        rng.shuffle(local)
        blocks = block_ids(len(local), case["carrier"] if w.get("revision",2)>=3 or w.get("natural_render") else None)
        threshold = (len(set(blocks)) + 1) // 2
        cut = sum(b < threshold for b in blocks)
        case["session_split_round"] = cut
        if case["qtype"] == "multihop":
            # All equivalents of the route are in one session, the person fact
            # in the other. Neither session can provide both required hops.
            head = [m for m in local if case["targets"][0] in m["atoms"]]
            tail = [m for m in local if case["targets"][-1] in m["atoms"]]
            middle = [m for m in local if m not in head and m not in tail]
            first, second = middle[:cut - len(head)], middle[cut - len(head):]
            for message in head:
                first.insert(rng.randrange(len(first) + 1), message)
            for message in tail:
                second.insert(rng.randrange(len(second) + 1), message)
            local = first + second
        for j, m in enumerate(local):
            m["id"] = f"m{len(w['rounds']) + j}"
            m["local_block"] = blocks[j]
            m["thread"] = f"{case['id']}:thread{j % 2}"
        if w.get("revision",2)>=3 or w.get("natural_render"):
            configure_style(w,case,local)
        # Each message has an independent retry/quarantine boundary. Preserve
        # local_block only as a discussion/session annotation.
        for j,m in enumerate(local):
            family = (w["world"] + ci + m['local_block']) % 3
            jobs.append((case, j, family, [m]))
        w["rounds"].extend(local)
    def one(job):
        case, block, family, messages = job
        payload = {"task": "render", "carrier": case["carrier"], "project": w["project"], "messages": [render_input(m) for m in messages]}
        instructions = render_instructions(case['carrier'])
        fingerprint = digest([BUILD_CONTRACT, instructions, payload, family, models.mock, models.identity])
        path = Path(checkpoint_dir) / f"render_{messages[0]['id']}.json" if checkpoint_dir else None
        saved = read(path) if path and path.exists() else {}
        same_case = [f for f in w['facts'] if f['source']['case']==case['id']]
        forbidden = {m['id']:[signature_for(f) for f in same_case if f['id'] not in m['atoms'] and not leaks_anchors([signature_for(f)],m['core'])]
                     for m in messages} if w.get('revision',2)>=3 or w.get('natural_render') else {}
        def row_valid(row,m):
            reasons=message_failures(row,m,forbidden.get(m['id'],()))
            if reasons:row['_validation_failures']=reasons
            return not reasons
        def valid(result):
            rows = result.get('messages',[])
            by_id = {r['id']:r for r in rows if isinstance(r,dict) and isinstance(r.get('id'),str)}
            return len(by_id)==len(messages) and all(m['id'] in by_id and row_valid(by_id[m['id']],m) for m in messages)
        if saved.get('fingerprint') != fingerprint:
            try:
                failures={m['id']:[] for m in messages}
                result = models.ask(family,instructions,payload,row_validator=row_valid,failure_collector=failures)
                saved = {'fingerprint':fingerprint,'result':result,'status':'done' if valid(result) else 'partial','message_failures':failures}
            except ValueError as exc:
                saved = {'fingerprint':fingerprint,'status':'quarantined','reason':str(exc),'message_failures':failures}
            if path:write(path,saved)
        rows = saved.get('result',{}).get('messages',[])
        out = {r['id']:r for r in rows if isinstance(r,dict) and isinstance(r.get('id'),str)}
        for m in messages:
            m['failure_reasons']=saved.get('message_failures',{}).get(m['id'],[])
            row=out.get(m['id'])
            if row and row_valid(row,m):
                if case['carrier']=='group':augment_group(row,m)
                m.update(text=row['text'],family=family,ok=False)
                for key in ('turns','user_text','rendered_speaker_count'):
                    if key in row:m[key]=row[key]
                if 'assistant_reply' in row:m['assistant_reply']=row['assistant_reply']
            else:
                m.update(text='',family=family,ok=False,render_error=saved.get('reason','failed message validation'))
    with cf.ThreadPoolExecutor(max_workers=par) as executor:
        list(executor.map(one, jobs))
    match_lengths(w)
    if w.get('revision',2)>=3 or w.get('natural_render'):
        # Give duplicate empty rounds their own retry budget; successful fact
        # messages never get resent or replaced by this repair.
        duplicates=dedupe_fillers(w)
        for mid in duplicates:
            m=next(m for m in w['rounds'] if m['id']==mid)
            m.setdefault('failure_reasons',[]).append('duplicate no-new-fact filler')
            repair_failures={}
            case=next(c for c in w['cases'] if c['id']==m['case'])
            avoided={key for r in w['rounds'] if not r['core'] and not r.get('render_error')
                     for key in filler_keys(r,r)}
            forbidden_fillers=[signature_for(f) for f in w['facts'] if f['source']['case']==m['case']]
            payload={'task':'render','carrier':case['carrier'],'project':w['project'],
                     'messages':[render_input(m)], 'avoid_filler':[r['text'] for r in w['rounds'] if not r['core'] and not r.get('render_error')][-12:]}
            try:
                result=models.ask(m['family'],render_instructions(case['carrier'])+' Use a different social turn from avoid_filler.',payload,
                    row_validator=lambda row,source: valid_message(row,source,forbidden_fillers)
                        and not filler_keys(row,source) & avoided,
                    failure_collector=repair_failures)
                row=result['messages'][0]
                if case['carrier']=='group':augment_group(row,m)
                m['text']=row['text'];m.pop('render_error',None)
                for key in ('turns','user_text','rendered_speaker_count'):
                    if key in row:m[key]=row[key]
                if 'assistant_reply' in row:m['assistant_reply']=row['assistant_reply']
            except (ValueError,KeyError,IndexError):pass
            m['failure_reasons'].extend(repair_failures.get(mid,[]))
            if checkpoint_dir:
                write(Path(checkpoint_dir)/f'render_repair_{mid}.json',dict(
                    status='quarantined' if m.get('render_error') else 'done',
                    message_failures={mid:m['failure_reasons']}))
        for mid in dedupe_fillers(w):
            message=next(m for m in w['rounds'] if m['id']==mid)
            message.setdefault('failure_reasons',[]).append('duplicate no-new-fact filler')
    for f in w["facts"]:
        f["source"]["message_ids"] = [m["id"] for m in w["rounds"] if f["id"] in m["atoms"]]
    w['render_failure_counts']=dict(collections.Counter(reason for m in w['rounds'] for reason in m.get('failure_reasons',[])))
    def category(reason):
        for prefix in ('missing anchor','text length','assistant reply length','need user'):
            if reason.startswith(prefix):return prefix
        return reason
    by_carrier={}
    for carrier in CARRIERS:
        rows=[m for m in w['rounds'] if next(c['carrier'] for c in w['cases'] if c['id']==m['case'])==carrier]
        counts=collections.Counter(category(r) for m in rows for r in m.get('failure_reasons',[]))
        by_carrier[carrier]=dict(messages=len(rows),done=sum(not m.get('render_error') for m in rows),
            quarantined=sum(bool(m.get('render_error')) for m in rows),failures=dict(counts))
    def speaker_bin(n):return str(n) if n<4 else '4+'
    groups=[m for m in w['rounds'] if m.get('rendered_speaker_count') is not None]
    w['render_summary']=dict(messages=len(w['rounds']),by_carrier=by_carrier,
        failures=dict(collections.Counter(category(r) for m in w['rounds'] for r in m.get('failure_reasons',[]))),
        group_speakers=dict(rendered=dict(collections.Counter(speaker_bin(m['rendered_speaker_count']) for m in groups)),
            assembled=dict(collections.Counter(speaker_bin(len({t['speaker'] for t in m['turns']})) for m in groups))))
    print('RENDER_SUMMARY '+json.dumps(w['render_summary'],sort_keys=True),flush=True)
    w["rendered"] = True
    w['mock'] = models.mock
    w["renderer_identity"] = models.identity
    w.pop("_render_facts",None)
    return w


def verify(w, models, checkpoint_dir=None):
    """Judge messages independently; all required carriers must pass both families."""
    w=copy.deepcopy(w)
    instructions=(
        'Verify this one message against its source core. Return JSON {ok:bool,no_extra_facts:bool,quotes:object}. '
        'Return a supporting span per atom, keyed by message_id:atom_id (for example {"m2:f9":"supporting text"}). '
        'For messages without anchor_sets, use the message ID. '
        'Each span must occur in the message text, allowing whitespace normalization. '
        'Judge the entire message for entailment of every atom, including attribution, scope, condition, state, '
        'event date, version identity, replacement relation and links. Qualifiers may appear outside the supporting span; '
        'do not require all qualifiers in one quote. Set ok=false if any qualifier or fact is missing or wrong. '
        'A parenthesized date after an entity identifies that version by its first entry date. Paraphrases of the core are allowed. '
        'Check that events can be recorded on record_time; that date is supplied as message metadata. '
        'Empty-core turns may ask questions or make social comments but must assert no ledger facts or commitments. '
        'Reject additional, contradictory or cross-message facts and invented reasons for decisions. '
        'A reported fact belongs to its holder, not its reporter. Only the annotated speaker may assert facts in structured turns. '
        'Check annotated approvals and commitments. Check assistant_reply too; it may explain the notes but must add no facts '
        'or commitments and must not copy the assertions.')
    derived={'ok','verification','verify_failure_reasons','verify_required','evidence_quotes','evidence_contexts'}
    byid={f['id']:f for f in w['facts']}
    for ci,case in enumerate(w['cases']):
        local=[m for m in w['rounds'] if m['case']==case['id']]
        required=set(solve(w,case['query'])['atom_ids'])
        for m in local:
            source={k:v for k,v in m.items() if k not in derived}
            payload={'task':'verify','messages':[source]}
            renderer=m.get('family',(w['world']+ci)%3)
            fingerprint=digest([BUILD_CONTRACT,VERIFY_CONTRACT,instructions,payload,renderer,models.mock,models.identity])
            path=Path(checkpoint_dir)/f"verify_{m['id']}.json" if checkpoint_dir else None
            saved=read(path) if path and path.exists() else {}
            votes=[];raw_votes={};failures={}
            for shift in (1,2):
                family=(renderer+shift)%3
                vote=safe_vote(models,family,instructions,payload,
                    saved.get('votes',{}).get(str(shift)) if saved.get('fingerprint')==fingerprint else None,
                    blocked=bool(m.get('render_error')))
                raw_votes[str(shift)]=vote
                reasons=[]
                if m.get('render_error'):reasons.append('render failure: '+str(m['render_error']))
                if vote.get('ok') is not True:reasons.append('semantic rejection: '+str(vote.get('reason','ok=false')))
                if vote.get('no_extra_facts') is not True:reasons.append('no_extra_facts=false or missing')
                if vote.get('ok') is True and vote.get('no_extra_facts') is True:
                    spans=m.get('anchor_sets')
                    if spans is None:
                        spans={fid:anchors_for(byid[fid]) for fid in m.get('atoms',[])} if m['core'] else {}
                    for fid,anchors in spans.items():
                        quote=verification_quote(vote,m,fid)
                        if not isinstance(quote,str) or not collapse_whitespace(quote):
                            reasons.append(f'{fid}: missing supporting span')
                        elif collapse_whitespace(quote) not in collapse_whitespace(m['text']):
                            reasons.append(f'{fid}: supporting span not in message')
                        if not anchors_present(anchors,m['text']):
                            reasons.append(f'{fid}: missing message anchors')
                failures[str(family)]=reasons
                votes.append(not reasons)
            if path:write(path,{'fingerprint':fingerprint,'votes':raw_votes})
            m.update(ok=all(votes),verification=votes,verify_failure_reasons=failures,
                verify_required=bool(required.intersection(m['atoms'])))
            m['evidence_quotes']={fid:[verification_quote(raw_votes[str(shift)],m,fid) for shift in (1,2)]
                for fid in m.get('anchor_sets',{})} if m['ok'] else {}
            # A short locator quote is not a version identity. Bind downstream
            # projection to the full message whose qualifiers both judges checked.
            m['evidence_contexts']={fid:[m['text']] for fid in m.get('anchor_sets',{})} if m['ok'] else {}
        carriers=[m for m in local if m['verify_required']]
        covered={fid for m in carriers for fid in m['atoms']}
        case['verify_missing_atoms']=sorted(required-covered)
        case['semantic_ok']=not case['verify_missing_atoms'] and all(m['ok'] for m in carriers)
    def counts(rows):
        return dict(messages=len(rows),passed=sum(m['ok'] for m in rows),failed=sum(not m['ok'] for m in rows),
            required=sum(m['verify_required'] for m in rows),
            required_failed=sum(m['verify_required'] and not m['ok'] for m in rows),
            dropped_non_required=sum(not m['verify_required'] and not m['ok'] for m in rows),
            failures=dict(collections.Counter(reason for m in rows for reasons in m['verify_failure_reasons'].values() for reason in reasons)))
    carriers={c['id']:c['carrier'] for c in w['cases']}
    rows=[m for m in w['rounds'] if m['case'] in carriers]
    w['verify_summary']=dict(counts(rows),cases=len(w['cases']),
        semantic_ok=sum(c['semantic_ok'] for c in w['cases']),
        by_carrier={carrier:counts([m for m in rows if carriers[m['case']]==carrier]) for carrier in CARRIERS})
    print('VERIFY_SUMMARY '+json.dumps(w['verify_summary'],sort_keys=True),flush=True)
    w.update(verified=True,verify_contract=VERIFY_CONTRACT,verifier_identity=models.identity,mock=models.mock)
    return w


def serialize(w, include_unverified=False):
    """One message = one round, except true user/assistant rounds.

    Assistant-only facts live on the assistant side of a genuine round; human
    group/dyad/record messages are user-role turns with visible source prefixes.
    """
    names = w["team"]
    hs, dates = [], []
    for case in w["cases"]:
        # A rejected case must not poison other questions' histories either.
        if not include_unverified and not case.get("semantic_ok"):
            continue
        parts = [[], []]
        messages = [r for r in w["rounds"] if r["case"] == case["id"]]
        cut = case["session_split_round"]
        for mi, m in enumerate(messages):
            if w.get('verified') and not m.get('ok'):
                continue
            sess = parts[int(mi >= cut)]
            prefix = f"[{m['channel']}/{m['thread']}] {m['speaker']}: "
            # Attribution belongs to the message's existing carrier side.
            attribution = f"Reported by {m['reported_by']}. " if m.get("reported_by") else ""
            meta = dict(id=m["id"], stage=m["atoms"][0] if m["atoms"] else m["id"], kind=m["kind"],
                        carrier=case["carrier"], speaker=m["speaker"], recipients=m["recipients"],
                        channel=m["channel"], thread=m["thread"], atoms=m["atoms"],
                        expression=m["expression"], record_time=m["record_time"], new_fact=m["new_fact"])
            meta["meaningful_confirmation"] = m.get("meaningful_confirmation", False)
            meta["confirmation_kind"] = m.get("confirmation_kind")
            if m["speaker"] == "Assistant":
                prompt = m.get('user_text',ASSISTANT_PROMPT) if m.get("length_contract") else ("Please share the two recorded entries together, with their original dates and attribution. "
                          "I would like to read both complete statements so I can follow this part of the discussion and its context."
                          if m.get("max_chars") else "Could you share the recorded details?")
                prompt_author = "" if m.get("max_chars") else case['asker'] + ": "
                sess.append(dict(role="user", content=f"[{m['channel']}/{m['thread']}] {prompt_author}{prompt}", **meta))
                sess.append(dict(role="assistant", content=prefix + m["text"] + attribution, **meta))
            else:
                sess.append(dict(role="user", content=prefix + m["text"], **meta))
                if case["carrier"] == "assistant":
                    # No invented evidence in the reply.
                    sess.append(dict(role="assistant", content=m.get("assistant_reply","Thanks for the context."), **meta))
        hs.extend(parts)
        dates.extend([dt.date.fromisoformat(case["record_time"]).strftime("%B-%d-%Y")] * 2)
    return hs, dates


def visible_documents(q, max_chars=3000, token_limit=None, tokenizer=None):
    """Real fmt context and truncation, optionally the actual downstream tokenizer.

    No guessed char/token ratio is accepted for a token visibility guarantee.
    """
    prev = round_prev(q)
    docs = [doc_str(r[1], *split_round(r[2]), prev=prev[i], max_chars=max_chars)
            for i, r in enumerate(rounds_of(q))]
    if token_limit is not None:
        if tokenizer is None:
            raise ValueError("token_limit requires the actual training tokenizer")
        from fmt import SFX
        # train.py tokenizes query and document separately, strips the suffix,
        # limits document-body tokens to max_d, then appends suffix tokens again.
        docs = [tokenizer.decode(tokenizer.encode(d[:-len(SFX)] if d.endswith(SFX) else d,
                                                  add_special_tokens=False)[:token_limit]) + SFX for d in docs]
    return docs


def evidence_bindings(w):
    bindings=collections.defaultdict(list)
    for message in w.get('rounds',[]):
        if message.get('ok'):
            for fid,quotes in message.get('evidence_contexts',message.get('evidence_quotes',{})).items():
                bindings[fid].extend(quote for quote in quotes if quote)
    return bindings


def bound_atom(w, fact, bindings=None):
    """Attach verifier-approved spans, never infer version identity from 3 anchors."""
    atom=dict(fact,anchors=anchors_for(fact))
    if w.get('rounds'):
        bindings=evidence_bindings(w) if bindings is None else bindings
        atom['evidence_spans']=list(dict.fromkeys(bindings.get(fact['id'],[])))
    return atom


def evidence_map(w, q, required, docs=None):
    docs = docs if docs is not None else visible_documents(q)
    byid = {f["id"]: f for f in w["facts"]}
    bindings=evidence_bindings(w)
    return {fid: [i for i, doc in enumerate(docs) if atom_present(bound_atom(w,byid[fid],bindings) if byid[fid].get("natural_surface") else byid[fid], doc)] for fid in required}


def check_visibility(w, q, **kwargs):
    docs = visible_documents(q, **kwargs)
    emap = evidence_map(w, q, [a["id"] for a in q["answer_atoms"]], docs)
    errors = []
    for atom in q["answer_atoms"]:
        missing = set(atom["equivalent_rounds"]) - set(emap[atom["id"]])
        if not emap[atom["id"]] or missing:
            errors.append({"atom": atom["id"], "invisible_rounds": sorted(missing)})
    return {"ok": not errors, "errors": errors, "visible": emap}


def load_tokenizer(path):
    """Local files only: a missing training tokenizer is never a character proxy."""
    if not path:
        raise ValueError("actual training tokenizer path is required")
    path = Path(path)
    if not path.exists():
        raise ValueError("tokenizer path does not exist")
    if path.is_dir():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        tokenizer.world3_fingerprint = tokenizer_fingerprint(path)
        return tokenizer
    from tokenizers import Tokenizer
    backend = Tokenizer.from_file(str(path))
    class LocalTokenizer:
        world3_fingerprint = tokenizer_fingerprint(path)
        def encode(self, text, add_special_tokens=False):
            return backend.encode(text, add_special_tokens=add_special_tokens).ids
        def decode(self, ids):
            return backend.decode(ids, skip_special_tokens=False)
    return LocalTokenizer()


def visible_fact_rows(w, documents):
    by_statement=collections.defaultdict(list)
    for f in w['facts']:by_statement[f['statement']].append(f)
    bindings=evidence_bindings(w)
    bound={f['id']:bound_atom(w,f,bindings) for f in w['facts'] if f.get('natural_surface')}
    natural=collections.defaultdict(list)
    other=[]
    for f in w['facts']:
        if f.get('natural_surface'):natural[normalize_filler(f['entity'])].append(f)
        elif not f['statement'].startswith('On '):other.append(f)
    entity_pattern=re.compile(r"\b(?:"+'|'.join(map(re.escape,natural))+r")\b") if natural else None
    paraphrased=any(m['core'] and m['core'] not in m.get('text','') for m in w.get('rounds',[]))
    result=[]
    for d in documents:
        clauses=CORE_CLAUSE.findall(d)
        found=[f for c in clauses for f in by_statement.get(c,[]) if
               not f.get('natural_surface') or atom_present(bound[f['id']],d)]
        # Canonical fixtures take the exact parser path. A real paraphrase is
        # projected only if all anchors of that fact survive in the document.
        if entity_pattern and (not clauses or paraphrased):
            names=set(entity_pattern.findall(normalize_filler(d)))
            existing={f['id'] for f in found}
            found.extend(f for name in names for f in natural[name] if f['id'] not in existing and atom_present(bound[f['id']],d))
        found.extend(f for f in other if f['statement'] in d)
        result.append(found)
    return result


def solve_visible(w, query, documents, observed=None):
    """Solve only the ledger rows whose complete assertions survive rendering.

    No unobserved predecessor, approved state or rollback is copied from the
    hidden ledger. Lifecycle rows carry their explicitly declared replacement
    relation; the ledger itself is never supplied as evidence to the reader.
    """
    if observed is None:
        observed = [f for rows in visible_fact_rows(w, documents) for f in rows]
    observed = list({f["id"]: f for f in observed}.values())
    try:
        return {"available": True, **solve({"facts": observed}, query)}
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}


def necessity_check(w, q, docs=None):
    docs = visible_documents(q) if docs is None else docs
    rows = visible_fact_rows(w, docs)
    full = solve_visible(w, q["query_spec"], docs, [f for row in rows for f in row])
    meta = [m for sess in q["haystack_sessions"] for m in sess if m["role"] == "user"]
    checks = {}
    for atom in q["answer_atoms"]:
        # Include truncated contextual echoes as well as complete carriers.
        removed = set(atom["equivalent_rounds"]) | {i for i in q["support_rounds"]
            if i > 0 and atom["id"] in meta[i - 1].get("atoms", []) and atom["statement"][:80] in docs[i]}
        remaining = [f for i, row in enumerate(rows) if i not in removed for f in row]
        after = solve_visible(w, q["query_spec"], [], remaining)
        checks[atom["id"]] = dict(removed_rounds=sorted(removed), after=after,
            necessary=not after["available"] or after.get("answer") != q["answer"])
    return dict(full=full, blind=solve_visible(w, q["query_spec"], []), atoms=checks,
                ok=full.get("answer") == q["answer"] and all(c["necessary"] for c in checks.values()))


def classify_negatives(w, q, docs=None):
    """Query-conditioned, disjoint structural categories; partial support is blocked."""
    docs = docs if docs is not None else visible_documents(q)
    rounds = rounds_of(q)
    meta = [sess[i] for sess in q["haystack_sessions"] for i in range(len(sess))
            if sess[i]["role"] == "user"]
    # All serialized rounds start with user, so meta aligns with k1 exactly.
    byid = {f["id"]: f for f in w["facts"]}
    required = [byid[a["id"]] for a in q["answer_atoms"]]
    requested = q["query_spec"]["requests"] + [
        {k: f[k] for k in ("holder", "scope", "entity", "attr")} for f in required]
    entities = {f["entity"] for f in required}
    speakers = {meta[i]["speaker"] for a in q["answer_atoms"] for i in a["equivalent_rounds"]}
    protected = {i for a in q["answer_atoms"] for i in a["equivalent_rounds"]} | set(q.get("support_rounds", []))
    out = {kind: [] for kind in NEG_QUOTAS}
    reasons = {}
    previous = round_prev(q)
    for i, m in enumerate(meta):
        if i in protected:
            continue
        fs = [byid[a] for a in m.get("atoms", [])]
        # All possible partial facts visible through a 200-char follow-up context
        # are support, never negatives, even if the full clause was truncated.
        # A matching value/date prefix in a DIFFERENT vetoed assertion is not
        # partial gold. Protect only a real echo from the preceding user turn.
        context = clean(previous[i])[:200]
        if i > 0 and any(f["id"] in meta[i - 1].get("atoms", [])
                         and f["statement"].split(" was ")[0] in context for f in required):
            protected.add(i)
            continue
        if not fs:
            category = "contentless_ack"
            reason = "No state transition, commitment, reference necessary to resolve the query, or new atom."
        elif all(f["entity"] not in entities for f in fs):
            category = "same_speaker_other_topic" if m["speaker"] in speakers else "random"
            reason = "Different task/entity from every necessary answer atom."
        elif any(f["entity"] in entities and any(f["attr"] == r["attr"] and (f["holder"] != r["holder"] or f["scope"] != r["scope"]) for r in requested) for f in fs):
            category = "same_topic_wrong_scope"
            reason = "Same topic but explicitly wrong holder or personal/team scope."
        else:
            # Admit only candidates for which resolve explicitly says inapplicable.
            inappropriate = True
            for f in fs:
                req = {k: f[k] for k in ("holder", "scope", "entity", "attr")}
                answer = resolve(w["facts"], req, q["query_spec"]["as_of"], q["query_spec"]["known_at"], q["query_spec"].get("context"))
                if answer and answer["id"] == f["id"]:
                    inappropriate = False
            if not inappropriate:
                # Unrequested applicable facts remain ordinary negatives.
                category, reason = "random", "Applicable fact for an unrequested attribute."
            else:
                category, reason = "minimal_state_diff", "Proposal, condition mismatch, future event or inactive lifecycle."
        out[category].append(i)
        reasons[str(i)] = reason
    return out, reasons, sorted(protected)


def sample_negatives(candidates, rng, allowed=None, requirements=None):
    picked, labels, shortages = [], [], {}
    for kind, quota in NEG_QUOTAS.items():
        src = [i for i in candidates[kind] if i not in picked and (allowed is None or i in allowed)]
        take = []
        required = (requirements or {}).get(kind, {})
        for options in required.values():
            eligible = sorted(set(src) & set(options) - set(take))
            if eligible:
                take.append(rng.choice(eligible))
        # A missing required subtype consumes its reserved slot. Do not fill it
        # with another state and pretend the semantic quota was satisfied.
        free = max(0, quota - len(required))
        remaining = [i for i in src if i not in take]
        take.extend(rng.sample(remaining, min(free, len(remaining))))
        picked.extend(take)
        labels.extend([kind] * len(take))
        shortages[kind] = quota - len(take)
    # Never silently replace a missing semantic category with random negatives.
    return picked, labels, shortages


def build_questions(w, allow_unverified=False, only_case=None, tokenizer=None, token_limit=800, build_report=None):
    validate_ledger(w["facts"])
    if w.get("revision",2)>=3: validate_types(w)
    if not w.get("verified") and not allow_unverified:
        raise ValueError("run verify before build")
    hs, dates = serialize(w, include_unverified=allow_unverified)
    qdate = (dt.date.fromisoformat(w["cases"][-1]["record_time"]) + dt.timedelta(days=5)).strftime("%B-%d-%Y")
    result = []
    bindings=evidence_bindings(w)
    build_report = {} if build_report is None else build_report
    build_report["quarantine"] = []
    for case in w["cases"]:
        if only_case is not None and case["id"] != only_case:
            continue
        if not allow_unverified and not case.get("semantic_ok"):
            build_report["quarantine"].append(dict(case=case["id"], reason="semantic_or_render_failure"))
            continue
        solved = solve(w, case["query"])
        required = solved["atom_ids"]
        q = dict(question_id=f"w{w['world']}_v3_{case['qtype']}_{case['id']}", question_type=case["qtype"],
                 version=3, revision=w.get("revision", 2), split=w["split"], ledger_id=w["ledger_id"], case_id=case["id"],
                 question=case["question"], answer=solved["answer"], query_spec=copy.deepcopy(case["query"]),
                 rubric=["LLM response should contain: " + next(f["statement"] for f in w["facts"] if f["id"] == fid) for fid in required],
                 question_date=qdate, haystack_sessions=copy.deepcopy(hs), haystack_dates=dates,
                 hist_tok=sum(ntok(t["content"]) for s in hs for t in s), tilt="uniform", scope="date" if case["date_window"] else "semantic",
                 carrier=case["carrier"], carriers=[case["carrier"]], perspective=case["perspective"], asker=case["asker"],
                 assistant_only=case["assistant_only"], date_window=case["date_window"], mock=w.get("mock", False))
        q.update(domain=w.get("domain"), value_vocabulary=w.get("value_vocabulary", []),
                 evidence_speaker_is_asker=case.get("evidence_speaker_is_asker"),
                 evidence_speaker_is_holder=case.get("evidence_speaker_is_holder"),
                 meaningful_confirmations=[m["id"] for m in w["rounds"]
                     if m["case"] == case["id"] and m.get("meaningful_confirmation")
                     and any(fid in required for fid in m["atoms"])])
        q["domain_structure"] = observed_structure(w, case) if case["query"].get("domain_program") else None
        q["structure_hash"] = q["domain_structure"]["hash"] if q["domain_structure"] else None
        docs = visible_documents(q)
        emap = evidence_map(w, q, required, docs)
        byid = {f["id"]: f for f in w["facts"]}
        q["answer_atoms"] = [dict(id=fid, value=byid[fid]["value"], holder=byid[fid]["holder"],
                                  attr=byid[fid]["attr"], entity=byid[fid]["entity"], statement=byid[fid]["statement"],
                                  equivalent_rounds=emap[fid], **({key:val for key,val in bound_atom(w,byid[fid],bindings).items() if key in ('anchors','evidence_spans')} if byid[fid].get('natural_surface') else {})) for fid in required]
        if any(m.get('anchor_sets') is not None and m['core'] and m['core'] not in m.get('text','') for m in w['rounds']):
            q['evidence_catalog']=[bound_atom(w,f,bindings) for f in w['facts']
                if f['source'].get('message_ids')]
        q["round_atoms"] = {str(i): [fid for fid in required if i in emap[fid]] for i in range(len(docs))}
        if any(not emap[fid] for fid in required):
            build_report["quarantine"].append(dict(case=case["id"], reason="necessary_atom_not_visible"))
            continue
        # State reversals are necessary support, protected even under --no-set.
        support = set()
        for fid in case.get("state_support", []):
            support.update(i for i, doc in enumerate(docs) if atom_present(bound_atom(w,byid[fid],bindings) if byid[fid].get("natural_surface") else byid[fid], doc))
        q["support_rounds"] = sorted(support)
        negs, reasons, protected = classify_negatives(w, q, docs)
        q["support_rounds"] = sorted(set(protected) - {i for a in q["answer_atoms"] for i in a["equivalent_rounds"]})
        q["negative_candidates"], q["negative_reasons"] = negs, reasons
        meta = [m for sess in q["haystack_sessions"] for m in sess if m["role"] == "user"]
        q["atom_negative_candidates"] = {a["id"]: {
            category: [i for i in indices if category not in ("same_topic_wrong_scope", "minimal_state_diff") or
                       any(byid[fid]["entity"] == a["entity"] and byid[fid]["attr"] == a["attr"] for fid in meta[i].get("atoms", []))]
            for category, indices in negs.items()} for a in q["answer_atoms"]}
        q["atom_negative_requirements"] = {}
        for a in q["answer_atoms"]:
            rows = q["atom_negative_candidates"][a["id"]]["minimal_state_diff"]
            fs = {i: [byid[fid] for fid in meta[i].get("atoms", [])] for i in rows}
            context = q["query_spec"].get("context", {})
            applicable = lambda f: all(context.get(k) == val for k, val in f["condition"].items())
            q["atom_negative_requirements"][a["id"]] = {"minimal_state_diff": {
                "vetoed": [i for i in rows if any(f["status"] == "vetoed" for f in fs[i])],
                "other_time": [i for i in rows if any(f["status"] in ("approved", "executed") and applicable(f)
                    and f["event_time"] <= q["query_spec"]["as_of"] for f in fs[i])],
                "condition_or_future": [i for i in rows if any(not applicable(f)
                    or f["event_time"] > q["query_spec"]["as_of"] for f in fs[i])]}}
        _, _, q["negative_shortfall"] = sample_negatives(negs, random.Random(q["question_id"]))
        if case.get("competition_panels"):
            q['competition_panels'] = {fid: [next((i for i,m in enumerate(meta) if doc[0] in m.get('atoms',[])),-1)
                for doc in documents] for fid,documents in case['competition_documents'].items()}
            q['panel_subtypes'] = case['panel_subtypes']
            q['negative_recipe'] = 'competition_random_typed_clusters'
            q["negative_shortfall"] = {fid: max(0, 16 - len(set(indices) - {-1}))
                for fid, indices in q["competition_panels"].items()}
        if w.get('revision',2)>=3:
            from world3_semantics import parse_question, check_alignment
            q['question_alignment']=check_alignment(q['question'],case['query'],w['facts'])
            if not q['question_alignment']['ok']:
                build_report['quarantine'].append(dict(case=case['id'],reason='question_spec_mismatch',details=q['question_alignment']))
                continue
        q["gold_stages"] = required
        q["support_stages"] = []
        index = 0
        for sess in q["haystack_sessions"]:
            j = 0
            while j < len(sess):
                atoms = q["round_atoms"][str(index)]
                stage = atoms[0] if len(atoms) == 1 else f"context{index}" if atoms or index in protected else sess[j]["stage"]
                if len(atoms) > 1 or index in q["support_rounds"]:
                    q["support_stages"].append(stage)
                size = 2 if j + 1 < len(sess) and sess[j + 1]["role"] == "assistant" else 1
                for t in sess[j:j + size]:
                    t["stage"] = stage
                    t["has_answer"] = bool(atoms) and t["role"] == "user"
                index += 1
                j += size
        q["visibility"] = check_visibility(w, q)
        q["token_visibility"] = (dict(checked=True, token_limit=token_limit,
            tokenizer_fingerprint=getattr(tokenizer, "world3_fingerprint", None),
            **check_visibility(w, q, token_limit=token_limit, tokenizer=tokenizer))
            if tokenizer is not None else dict(checked=False, ok=False, reason="actual training tokenizer required"))
        if tokenizer is not None and not q["token_visibility"]["ok"]:
            build_report["quarantine"].append(dict(case=case["id"], reason="token_visibility", details=q["token_visibility"]))
            continue
        q["necessity"] = necessity_check(w, q, visible_documents(q,
            token_limit=token_limit, tokenizer=tokenizer) if tokenizer is not None else None)
        if not q["necessity"]["ok"]:
            build_report["quarantine"].append(dict(case=case["id"], reason="necessity", details=q["necessity"]))
            continue
        if case["date_window"]:
            in_window = [i for kind in ("same_topic_wrong_scope", "minimal_state_diff") for i in negs[kind]
                         if case["date_window"][0] <= dt.datetime.strptime(rounds_of(q)[i][1], "%B-%d-%Y").date().isoformat() <= case["date_window"][1]]
            if not in_window:
                build_report['quarantine'].append(dict(case=case['id'],
                    reason='dated question lacks in-window hard negatives',
                    negative_shortfall=q['negative_shortfall']))
                continue
            q["window_hard_negatives"] = in_window
        if re.search(r"\blist\s+\d+\b", q["question"], re.I):
            raise ValueError("List N answer-cardinality leak")
        result.append(q)
    build_report["summary"] = dict(questions=len(result),
        token_visibility_complete=bool(result) and bool(getattr(tokenizer, "world3_fingerprint", None))
        and not build_report["quarantine"])
    return result


def gates(w, questions, models, checkpoint_dir=None, reader=None):
    """Semantic gates use exactly doc_str-visible evidence, not raw long messages.

    Per-atom deletion removes *all* equivalent and context carriers. Mock results
    are conspicuously marked and may only be used for plumbing tests.
    """
    from world3_reader import LLMReader, measure_reader
    reader = reader or LLMReader(models)
    out = []
    for qi, q in enumerate(questions):
        path = Path(checkpoint_dir) / f"gate_{q['question_id']}.json" if checkpoint_dir else None
        fingerprint = digest([BUILD_CONTRACT, q, models.mock, models.identity, reader.identity, reader.mock, "gold-evidence-reader-v2"])
        saved = read(path) if path and path.exists() else {}
        if saved.get("fingerprint") == fingerprint:
            out.append(saved["result"])
            continue
        docs = visible_documents(q)
        atom_defs = [{"id": a["id"], "statement": a["statement"],
            **({key:a[key] for key in ('anchors','evidence_spans') if key in a})} for a in q["answer_atoms"]]
        candidates = {str(i): d for i, d in enumerate(docs) if any(atom_present(a,d) for a in q["answer_atoms"])
                      or i in q["support_rounds"] or any(i in xs for xs in q["negative_candidates"].values()) and q["case_id"] == q["haystack_sessions"][rounds_of(q)[i][0]][0]["thread"].split(":")[0]}
        checks = []
        for shift in (1, 2):
            vote = safe_vote(models, w["world"] + qi + shift,
                "Audit the question and computed answer against ONLY these visible documents. Return JSON "
                "{question_matches:bool,answer_correct:bool,supported:{atom_id:[{round:int,quote:str}]}}. "
                "For every atom enumerate ALL equivalent supporting rounds and quote an exact sufficient clause. "
                "Partial support for any required hop matters; a round need not answer the whole question. "
                "Do not count proposals, wrong holder/scope/conditions as current evidence. Verify temporal order.",
                {"task": "gate", "question": q["question"], "answer": q["answer"], "atoms": atom_defs, "documents": candidates})
            ok = vote.get("question_matches") is True and vote.get("answer_correct") is True
            for atom in q["answer_atoms"]:
                quotes = vote.get("supported", {}).get(atom["id"], [])
                got = set()
                for item in quotes:
                    idx = int(item["round"])
                    quote=item.get('quote')
                    # The round must retain its verified version binding. A new
                    # semantic audit may quote a shorter sufficient clause.
                    quote_ok=anchors_present(atom['anchors'],quote) if atom.get('anchors') and isinstance(quote,str) else atom_present(atom,quote or '')
                    if quote and quote in docs[idx] and atom_present(atom,docs[idx]) and quote_ok:
                        got.add(idx)
                ok = ok and got == set(atom["equivalent_rounds"])
            checks.append(bool(ok))
        necessity = necessity_check(w, q, docs)
        deletion = {fid: row["necessary"] for fid, row in necessity["atoms"].items()}
        computed = solve(w, q["query_spec"])
        program_ok = computed["answer"] == q["answer"] and set(computed["atom_ids"]) == {a["id"] for a in q["answer_atoms"]}
        if q.get("revision",2)>=3:
            from world3_semantics import check_alignment
            program_ok = program_ok and check_alignment(q["question"],q["query_spec"],w["facts"])["ok"]
        token_info = q.get("token_visibility", {})
        token_ok = bool(token_info.get("checked") and token_info.get("ok") and token_info.get("tokenizer_fingerprint"))
        measured = measure_reader(w, q, docs, necessity, reader)
        passed = measured["passed"] and program_ok and all(checks) and check_visibility(w, q)["ok"] and necessity["ok"] and (models.mock or token_ok)
        # Keep raw reader scores separate from the combined release decision.
        out.append(dict(qid=q["question_id"], type=q["question_type"], k=len(atom_defs),
                        blind=measured["blind"], suff=measured["suff"],
                        **{"del": measured["deletion"]}, reader=measured,
                        order=passed, v3_gate="visible_solver_necessity_two_family_support_and_measured_reader", semantic_votes=checks,
                        atom_deletion=deletion, necessity=necessity,
                        token_visibility_checked=q.get("token_visibility", {}).get("checked", False),
                        completion="mock_only" if models.mock or reader.mock else "complete" if passed else "incomplete",
                        mock=models.mock or reader.mock))
        if path:
            write(path, {"fingerprint": fingerprint, "result": out[-1]})
    return out


def quota_audit(w, questions):
    n = len(questions)
    bins = collections.Counter("one" if len(q["answer_atoms"]) == 1 else "two" if len(q["answer_atoms"]) == 2
                               else "three_five" if len(q["answer_atoms"]) <= 5 else "six_eight" for q in questions)
    return dict(planned_questions=len(w["cases"]), exported_questions=n, ledger_id=w["ledger_id"], split=w["split"],
                carrier=dict(collections.Counter(q["carrier"] for q in questions)),
                question_type=dict(collections.Counter(q["question_type"] for q in questions)),
                evidence=dict(bins), perspective={carrier: dict(collections.Counter(q["perspective"] for q in questions if q["carrier"] == carrier))
                                                for carrier in ("group", "dyad")},
                target_carrier=CARRIERS, target_type=QTYPES, target_evidence=EVIDENCE_V3 if w.get("revision",2)>=3 else EVIDENCE,
                no_new_fact_share=sum(not m["new_fact"] for m in w["rounds"]) / max(1, len(w["rounds"])),
                dated_questions=sum(bool(q["date_window"]) for q in questions),
                assistant_only_questions=sum(q["assistant_only"] for q in questions),
                quarantined_cases=[c["id"] for c in w["cases"] if not c.get("semantic_ok")],
                build_quarantine=w.get("build_quarantine", []),
                build_summary=w.get("build_summary", {}), domain=w.get("domain"),
                domain_splits=w.get("domain_splits"), domain_structure=distribution(questions),
                negative_shortfall={q["question_id"]: q["negative_shortfall"] for q in questions if any(q["negative_shortfall"].values())},
                mock=w.get("mock", False))


def validate_domain_splits(mapping):
    if set(mapping) != {"train", "dev", "audit"}:
        raise ValueError("domain splits must specify train/dev/audit")
    seen = set()
    for domains in mapping.values():
        if not domains or len(domains) != len(set(domains)) or any(d not in DOMAINS for d in domains):
            raise ValueError("empty, duplicate or unknown domains")
        if seen & set(domains):
            raise ValueError("domain holdout overlaps")
        seen.update(domains)
    return mapping


def tokenizer_fingerprint(path):
    if not path:
        return None
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.iterdir()
        if f.is_file() and f.suffix in (".json", ".txt", ".model"))
    return digest([(f.name, hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])


def sample_dump(worlds, path):
    """Ten actual rendered messages per available carrier, with ledger labels."""
    samples={c:[] for c in CARRIERS}
    for w in worlds:
        facts={f['id']:f for f in w['facts']}
        for case in w['cases']:
            required=set(solve(w,case['query'])['atom_ids'])
            messages=[m for m in w['rounds'] if m['case']==case['id'] and m.get('text')]
            random.Random(f"sample:{w['seed']}:{w['world']}:{case['id']}").shuffle(messages)
            for m in messages:
                samples[case['carrier']].append(dict(ledger_id=w['ledger_id'],message_id=m['id'],
                    carrier=case['carrier'],speaker=m['speaker'],text=m['text'],
                    assistant_reply=m.get('assistant_reply'),verified=m.get('ok',False),
                    question=case['question'],atoms=[dict(facts[fid],label=fid in required) for fid in m['atoms']],
                    label=bool(required & set(m['atoms'])),new_fact=m['new_fact']))
    for carrier,rows in samples.items():
        random.Random('sample-v6:'+carrier).shuffle(rows)
        samples[carrier]=rows[:10]
    write(path,dict(contract=BUILD_CONTRACT,mock=all(w.get('mock') for w in worlds),
        requested_per_carrier=10,counts={c:len(s) for c,s in samples.items()},messages=samples))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=("plan", "render", "verify", "build", "gates", "all"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--questions", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("train", "dev", "audit"), default="train")
    ap.add_argument("--project", default=None)
    ap.add_argument("--domain", choices=list(DOMAINS))
    ap.add_argument("--domain-splits", help="JSON mapping train/dev/audit to disjoint domain lists")
    ap.add_argument("--tokenizer", help="local training tokenizer directory or tokenizer.json; no downloads")
    ap.add_argument("--max-d", type=int, default=800)
    ap.add_argument("--par", type=int, default=4, help="parallel local rendering blocks")
    ap.add_argument("--timeout", type=float, default=120, help="seconds per request; at most three total attempts")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument('--sample-dump', help='write 10 rendered messages per carrier with atoms and ledger/query labels')
    ap.add_argument("--reader", help="optional MODEL@PORT for real answerer; default family 0 of the configured models")
    for flag in ("q27", "gem", "oss"):
        ap.add_argument("--" + flag, default="")
    a = ap.parse_args(argv)
    domain_splits = validate_domain_splits(read(a.domain_splits)) if a.domain_splits else None
    tokenizer = load_tokenizer(a.tokenizer) if a.tokenizer else None
    models = None if a.step in ("plan", "build") else ModelFamilies([a.q27, a.gem, a.oss], mock=a.mock, timeout=a.timeout)
    sample_worlds=[]
    for i in range(a.start, a.start + a.n):
        wp = Path(a.out) / f"w{i:05d}.json"
        if a.step in ("plan", "all"):
            plan = make_plan(i, a.questions, a.seed, a.split, a.project, a.domain, domain_splits)
            w = read(wp) if wp.exists() else plan
            if w.get("plan_fingerprint") != plan["plan_fingerprint"]:
                raise ValueError("existing plan differs; choose a new output/world range")
        else:
            w = read(wp)
        checkpoint_dir = Path(a.out) / "checkpoints" / f"w{i:05d}"
        if a.step in ("plan", "all"):
            write(wp, w)
        if a.step in ("render", "all"):
            if not w.get("rendered"):
                w = render(w, models, a.par, checkpoint_dir)
            elif w.get("renderer_identity") != models.identity:
                raise ValueError("renderer family changed on resume")
            write(wp, w)
        if a.step in ("verify", "all"):
            if w.get('verified') and w.get("verifier_identity") != models.identity:
                raise ValueError("verifier family changed on resume")
            if not w.get("verified") or w.get('verify_contract') != VERIFY_CONTRACT:
                w = verify(w, models, checkpoint_dir)
            write(wp, w)
        if a.step in ("build", "all"):
            qp = Path(a.out) / f"q{i:05d}.json"
            config = dict(contract=BUILD_CONTRACT, tokenizer=tokenizer_fingerprint(a.tokenizer), max_d=a.max_d,
                input=digest([w["facts"], w["cases"], w["rounds"], w["verified"], w["split"], w["ledger_id"]]))
            if w.get("build_config") == config and qp.exists():
                questions = read(qp)
            else:
                build_report = {}
                questions = build_questions(w, tokenizer=tokenizer, token_limit=a.max_d, build_report=build_report)
                w["build_quarantine"], w["build_summary"] = build_report["quarantine"], build_report["summary"]
                write(qp, questions)
                w["build_config"] = config
            write(Path(a.out) / f"audit{i:05d}.json", quota_audit(w, questions))
        if a.step in ("gates", "all"):
            questions = read(Path(a.out) / f"q{i:05d}.json")
            write(Path(a.out) / f"g{i:05d}.json", gates(w, questions, models, checkpoint_dir,
                __import__("world3_reader").LLMReader(ModelFamilies([a.reader] * 3, timeout=a.timeout)) if a.reader else None))
        w["llm_calls_last_step"] = dict(models.calls) if models else {}
        if a.sample_dump:
            sample_worlds.append(w)
            sample_dump(sample_worlds,a.sample_dump)
        write(wp, w)
        print(json.dumps({"world": i, "step": a.step, "facts": len(w["facts"]), "rounds": len(w["rounds"]),
                          "calls": w["llm_calls_last_step"], "mock": a.mock,
                          "token_visibility_complete": w.get("build_summary", {}).get("token_visibility_complete", False),
                          "completion": "mock_only" if a.mock else "tokenizer_required" if tokenizer is None else "check_gates"}))


if __name__ == "__main__":
    main()
