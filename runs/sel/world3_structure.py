"""Executable domain relations; no domain name is used by the structural hash."""
import collections
import hashlib
import json

# Transitions and constraints have operational meaning in solve_relations.
STRUCTURES = {
    'hospital': dict(machine=[('draft', 'rostered'), ('rostered', 'off_shift')], live='rostered',
        relation='worker_shift_interval', values=['06:00-10:00', '08:00-14:00', '14:00-22:00', '22:00-23:00'],
        constraint='pairwise_interval_intersection', shape='independent'),
    'travel': dict(machine=[('held', 'ticketed'), ('ticketed', 'cancelled'), ('held', 'expired')], live='ticketed',
        relation='reservation_requires', values=['confirmed', 'waitlisted', 'cancelled', 'expired'],
        constraint='ordered_dependency_conjunction', shape='chain'),
    'laboratory': dict(machine=[('received', 'screened'), ('screened', 'released'), ('screened', 'quarantined')], live='released',
        relation='sample_instrument_protocol_range', values=['2..8', '4..12', '6..10', '1..5'],
        constraint='range_intersection_contains_sample', shape='star'),
    'restaurant': dict(machine=[('ordered', 'received'), ('received', 'stocked'), ('stocked', 'consumed'), ('stocked', 'recalled')], live='stocked',
        relation='menu_inventory_supplier_units', values=['2', '4', '6', '8'],
        constraint='supplier_sum_covers_menu_demand', shape='join'),
    'school': dict(machine=[('nominated', 'elected'), ('elected', 'active'), ('active', 'resigned')], live='active',
        relation='member_role_event_vote', values=['yes', 'no', 'abstain', 'absent'],
        constraint='role_authorized_quorum', shape='cycle'),
    'software': dict(machine=[('built', 'tested'), ('tested', 'published'), ('published', 'yanked'), ('built', 'failed')], live='published',
        relation='component_version_dependency', values=['1.0', '1.5', '2.0', '3.0'],
        constraint='dependency_tree_version_bounds', shape='tree'),
}

RELATION_TYPES = {
    'hospital': ('doctor_department_interval', 'department_schedule_interval', 'schedule_shift_interval'),
    'travel': ('itinerary_booking_status', 'booking_document_status', 'document_requirement_status'),
    'laboratory': ('sample_instrument_range', 'instrument_protocol_range', 'protocol_measurement_range'),
    'restaurant': ('dish_ingredient_units', 'ingredient_supplier_units', 'supplier_delivery_units'),
    'school': ('event_organizer_vote', 'organizer_availability_vote', 'availability_committee_vote'),
    'software': ('component_dependency_version', 'dependency_constraint_version', 'constraint_provider_version'),
}


def parents(domain, n):
    """Rooted directed trees: every edge gates a distinct reachable operand."""
    if n<3: raise ValueError('native programs require at least two hops')
    edges=[[] for _ in range(n)]
    if domain=='travel':
        order=[0]+list(range(n-1,0,-1))
        for a,b in zip(order,order[1:]): edges[a].append(b)
    elif domain in ('laboratory','school'):
        edges[0]=[1];edges[1]=list(range(2,n))
    elif domain in ('restaurant','software'):
        edges[0]=list(range(1,n-1));edges[n-2]=[n-1]
    else:
        for i in range(n-1): edges[i]=[i+1]
    return edges


def traverse(rows):
    """Follow visible links from the first scoped entity, without consulting topology metadata."""
    byentity={f['entity']:f for f in rows}
    reachable=[];seen=set()
    def visit(entity):
        if entity in seen: return
        if entity not in byentity: raise ValueError('missing relation edge endpoint: '+entity)
        seen.add(entity);row=byentity[entity];reachable.append(row)
        for link in row.get('links',[]): visit(link)
    if rows: visit(rows[0]['entity'])
    return reachable


def normalized_structure(domain, n=7):
    """Names/lexical labels removed; directed transition topology and operators remain."""
    s = STRUCTURES[domain]
    nodes = list(dict.fromkeys(x for edge in s['machine'] for x in edge))
    return dict(transitions=sorted((nodes.index(a), nodes.index(b)) for a, b in s['machine']),
        live=nodes.index(s['live']), edges=parents(domain, n), operator=s['constraint'])


def structure_hash(domain):
    # Deliberately omit operator names too: a domain string cannot make hashes different.
    s = normalized_structure(domain)
    s.pop('operator')
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()


def solve_relations(domain, rows, program):
    """Each operator consumes every visible operand, and reports the intermediate result."""
    rows=traverse(rows)
    vals=[f['value'] for f in rows]
    entities=[f['entity'] for f in rows]
    edges=[[entities.index(link) for link in f.get('links',[])] for f in rows]
    if domain == 'hospital':
        def minutes(t):
            h, m = map(int, t.split(':')); return 60 * h + m
        times = [tuple(map(minutes, v.split('-'))) for v in vals]
        result = [(i, j, max(0, min(b, d) - max(a, c))) for i, (a, b) in enumerate(times)
                  for j, (c, d) in enumerate(times) if j > i]
        conclusion = f'overlap minutes={result}'
    elif domain == 'travel':
        def bookable(i):
            return vals[i]=='confirmed' and all(bookable(j) for j in edges[i])
        ready=[bookable(i) for i in range(len(rows))]
        conclusion = f'bookable dependency prefix={ready}'
    elif domain == 'laboratory':
        bounds = [tuple(map(int, v.split('..'))) for v in vals]
        lo, hi = max(a for a, b in bounds), min(b for a, b in bounds)
        conclusion = f'common protocol range={lo}..{hi}; sample={program["sample"]}; compatible={lo <= program["sample"] <= hi}'
    elif domain == 'restaurant':
        demand, supply = int(vals[0]), sum(map(int, vals[1:]))
        conclusion = f'menu demand={demand}; supplier inventory={supply}; portions={supply // demand}'
    elif domain == 'school':
        yes = sum(v == 'yes' for v in vals); attending = sum(v not in ('absent', 'abstain') for v in vals)
        conclusion = f'authorized delegation nodes={len(edges)}; votes={yes}/{attending}; quorum={program["quorum"]}; authorized={attending >= program["quorum"] and 2 * yes > attending}'
    else:
        versions = [tuple(map(int, v.split('.'))) for v in vals]
        compatible = [all(versions[i] >= versions[j] for j in edge) for i, edge in enumerate(edges)]
        conclusion = f'dependency version bounds satisfied={compatible}'
    return 'Reachable path: ' + ' -> '.join(entities) + '; ' + '; '.join(f'{f["entity"]} / {f["attr"]}={f["value"]}' for f in rows) + '; ' + conclusion


def distribution(questions):
    report = {}
    for domain in STRUCTURES:
        qs = [q for q in questions if q.get('domain') == domain]
        report[domain] = dict(normalized_hash=structure_hash(domain),
            realized_hashes=sorted({q.get('structure_hash') for q in qs if q.get('structure_hash')}),
            types=dict(collections.Counter(q['question_type'] for q in qs)),
            relations=dict(collections.Counter(a['attr'] for q in qs for a in q.get('answer_atoms', []))),
            relation_programs=dict(collections.Counter(q.get('query_spec', {}).get('domain_program', {}).get('relation', 'scalar_state') for q in qs)),
            realized_normalized=[q['domain_structure']['normalized'] for q in qs if q.get('domain_structure')],
            constraints=dict(collections.Counter(q.get('query_spec', {}).get('domain_program', {}).get('constraint', 'bitemporal_slot') for q in qs)))
    return report


def observed_structure(world, case):
    """Hash realized relation edges and lifecycle transitions after erasing every name/value.

    Unlike structure_hash(), this changes when a generated ledger edge is changed.
    Scalar bookkeeping facts are shared infrastructure and excluded explicitly.
    """
    rows=[f for f in world['facts'] if f['source']['case']==case['id'] and f.get('machine')]
    byid={f['id']:f for f in rows}
    states=list(dict.fromkeys(f['status'] for f in rows))
    transitions=sorted({(states.index(byid[f['prev']]['status']),states.index(f['status']))
        for f in rows if f.get('prev') in byid})
    targets={f['id']:f for f in rows if f['id'] in case['targets']}
    entities=[targets[fid]['entity'] for fid in case['targets'] if fid in targets]
    edges=sorted((i,entities.index(link)) for i,e in enumerate(entities)
        for f in targets.values() if f['entity']==e for link in f.get('links',[]) if link in entities)
    normalized=dict(state_count=len(states),transitions=transitions,nodes=len(entities),edges=edges)
    return dict(hash=hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest(),normalized=normalized)

INSTRUCTIONS = {
    'hospital': 'Start at the first scoped entity and follow Links through doctor, department and schedule. Use only reachable records. Report every scheduled interval and compute the overlap in minutes for every pair; touching endpoints have zero overlap.',
    'travel': 'Start at the first scoped entity and follow Links through itinerary, booking and document requirements. Use only reachable records. A reservation is bookable only when its recorded value is confirmed and every linked prerequisite is bookable. Evaluate prerequisites before dependents and report each result.',
    'laboratory': 'Start at the first scoped entity and follow Links through sample, instrument and protocol. Use only reachable records. Intersect every recorded sample, protocol and instrument range; report the common inclusive range and whether the stated sample measurement lies inside it.',
    'restaurant': 'Start at the first scoped entity and follow Links through dish, ingredient and supplier. Use only reachable records. The root dish gives menu demand per portion. All other reachable entities give on-hand ingredient or incoming supplier inventory units. Sum inventory and divide by demand, rounding down, to compute available portions; report all quantities.',
    'school': 'Start at the first scoped entity and follow Links through event, organizer and availability. Use only reachable records. Count yes and no as attending votes, exclude absent and abstain, and authorize only when attending votes meet the stated quorum and yes votes form a strict majority. Report every vote and both counts.',
    'software': 'Start at the first scoped entity and follow Links through component, dependency and version constraint. Use only reachable records. Follow each component version dependency link. A dependent version must be at least the linked provider version, comparing numeric major and minor versions. Report every version and whether each component satisfies all its dependency bounds.',
}
