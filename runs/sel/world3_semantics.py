"""Closed grammar for the generated natural language questions.

Input is question text only. Unknown/changed instructions fail closed. No query_spec,
answer, gold atoms, hidden task IDs or domain metadata are accepted by the parser.
"""
import re
from world3_domains import DOMAINS
from world3_structure import STRUCTURES, INSTRUCTIONS, parents


def parse_question(text):
    text=re.sub(r'^I \(\w+\) (?:reported on this|took part in the discussion)\. ', '', text)
    time=re.search(r' Evaluate the state effective on (\d{4}-\d\d-\d\d), using records available by (\d{4}-\d\d-\d\d)\.',text)
    if not time: raise ValueError('question lacks explicit snapshot constraints')
    context=re.search(r' Conditions: ([^.]+)\.',text)
    if not context: raise ValueError('question lacks conditions')
    conditions=dict(x.split('=',1) for x in context[1].split(', '))
    native=re.match(r"For (\w+), evaluate (\w+) using these scoped relations: (.+?)\. ",text)
    if native:
        domain=next((d for d,s in STRUCTURES.items() if s['relation']==native[2]),None)
        if domain is None or INSTRUCTIONS[domain] not in text: raise ValueError('unrecognized native question instructions')
        from world3 import NATIVE_TASKS
        scoped=native[3]
        operation=next((op for op in ('aggregation','temporal','multihop') if NATIVE_TASKS[op] in text),None)
        if operation is None:raise ValueError('missing native task instruction')
    else:
        patterns={'direct':r"What are ", 'preference':r'Prepare ', 'rule':r'Which of ',
            'update':r'After the decisions and reversals, ', 'aggregation':r'Summarize ',
            'temporal':r'In event order, ', 'recommendation':r'Should '}
        operation=next((k for k,v in patterns.items() if re.match(v,text)),None)
        if not operation: raise ValueError('unrecognized natural language question')
        match=re.search(r' Evaluate only these scoped attributes: (.+?)\.',text)
        if not match: raise ValueError('question lacks explicit scoped attributes')
        scoped=match[1]
        domain=next((d for d,o in DOMAINS.items() if o['context_key'] in conditions),None)
    requests=[]
    for clause in scoped.split('; '):
        m=re.fullmatch(r"(\w+)'s (personal|team) (.+?) for (.+)",clause)
        if not m: raise ValueError('unparseable scoped attribute')
        requests.append(dict(zip(('holder','scope','attr','entity'),m.groups())))
    if not requests or domain is None: raise ValueError('missing scoped constraint')
    ont=DOMAINS[domain]
    query=dict(operation=operation,entity=requests[0]['entity'],requests=requests,
        as_of=time[1],known_at=time[2],context=conditions,location_attr=ont['location'],
        available_value=ont['available'],clearance_value=ont['cleared'])
    if native:
        spec=STRUCTURES[domain]
        program=dict(domain=domain,relation=spec['relation'],constraint=spec['constraint'])
        for field,pattern in [('sample',r' Sample measurement: (\d+)\.'),('quorum',r' Quorum: (\d+)\.')]:
            m=re.search(pattern,text)
            if m:program[field]=int(m[1])
        query['domain_program']=program
    if operation=='update': query['include_reversals']=True
    # The complete prose must agree with the parsed constraints. This rejects a
    # contradictory headline even when the scoped list happens to remain intact.
    from world3 import question_text
    window=re.search(r' Use records between (\d{4}-\d\d-\d\d) and (\d{4}-\d\d-\d\d)\.',text)
    if window:
        from datetime import date
        bounds=list(window.groups())
        if date.fromisoformat(bounds[0]) > date.fromisoformat(bounds[1]):
            raise ValueError('reversed record date window')
        query['date_window']=bounds
    canonical=question_text(dict(query=query,perspective='external',asker='',explicit_as_of=True,
        date_window=list(window.groups()) if window else None))
    if canonical != text: raise ValueError('contradictory or unsupported question wording')
    return query


def check_alignment(text, specification, facts):
    from world3 import solve
    try:
        parsed=parse_question(text)
        expected=solve({'facts':facts},specification)
        actual=solve({'facts':facts},parsed)
        # Compare constraints, not just accidentally equal values at this snapshot.
        fields=('operation','requests','context','as_of','known_at','domain_program','include_reversals','date_window')
        same=all(parsed.get(k)==specification.get(k) for k in fields if not (k=='operation' and specification.get('domain_program') and specification.get('operation')=='direct'))
        return dict(ok=same and set(actual['atom_ids'])==set(expected['atom_ids']) and actual['answer']==expected['answer'],
            method='natural_language_parser_and_ledger_rules', parsed=parsed,
            natural_atom_ids=actual['atom_ids'],program_atom_ids=expected['atom_ids'])
    except (ValueError,KeyError,TypeError) as exc:
        return dict(ok=False,method='natural_language_parser_and_ledger_rules',error=str(exc))
