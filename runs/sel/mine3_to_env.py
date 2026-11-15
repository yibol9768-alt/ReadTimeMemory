"""Native mine3 -> the env_ws4 selector interface. No synthetic text fallback.

Run while the existing generation pools are still running. Failed writer/judge
batches are dropped and journalled. Both mine.jsonl and checkpoint layouts work.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# H20 keeps its audited fmt/k1/pool beside this adapter.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmt


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == '.jsonl':
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else [value]


def norm(text):
    return ' '.join(re.findall(r'\w+', str(text).casefold()))


def loose(phrase, text):
    phrase, text = norm(phrase), norm(text)
    return bool(phrase) and (' ' + phrase + ' ') in (' ' + text + ' ')


def load_pilots(roots):
    materials, mined, accepted = {}, {}, {}
    for root in map(Path, roots):
        for m in rows(root / 'materials.jsonl'):
            materials[m['material_id']] = m
        candidates = rows(root / 'mine.jsonl')
        for p in sorted((root / 'checkpoints' / 'mine').glob('*.json')):
            candidates += rows(p)
        for x in candidates:
            x = x.get('result', x)
            if not x.get('ok', True) or 'material' not in x:
                continue
            mid = x['material']['material_id']
            materials[mid], mined[mid] = x['material'], x
        # Exports have already passed mine3 acceptance; verification rows retain
        # explicit gold dependencies, so prefer those when both are present.
        for q in rows(root / 'export_questions.json'):
            accepted[q['question_id']] = q
        for q in rows(root / 'verify.jsonl'):
            if q.get('accepted') is True:
                accepted[q['question_id']] = q
    if not materials:
        raise ValueError('no material records found in materials.jsonl / successful mine records')
    return materials, mined, list(accepted.values())


def layout(material):
    """Return sessions, dates, and segment -> paired selector stage mapping."""
    source = material['source'].lower()
    sessions, dates, stage_of = [], [], {}
    for i, s in enumerate(material['segments']):
        sid, date = str(s['id']), str(s.get('time') or 'unknown')
        wild = source == 'wildchat'
        role = s.get('role', 'user') if wild else 'user'
        if role not in ('user', 'assistant', 'system'):
            role = 'user'
        new = not sessions or source in ('blog', 'blogs')
        if not new and not wild:
            new = len(sessions[-1]) >= 20 or dates[-1][:10] != date[:10]
        if new:
            sessions.append([])
            dates.append(date)
        stage = 'nat:' + material['material_id'] + ':' + sid
        if wild and role == 'assistant' and sessions[-1] and sessions[-1][-1]['role'] == 'user':
            stage = sessions[-1][-1]['stage']
        stage_of[sid] = stage
        prefix = ('Author: ' if source in ('blog', 'blogs') else '') + str(s.get('speaker', 'Unknown')) + ': '
        content = s['text'] if wild else prefix + s['text']
        sessions[-1].append(dict(role=role, content=content, id=sid, stage=stage,
                                  kind='ev', has_answer=False))
    return sessions, dates, stage_of


def make_question(material, qid, question, answer, kind, gold_ids, values=()):
    sessions, dates, stage_of = layout(material)
    gold_ids = {str(s) for s in gold_ids}
    if not gold_ids or not gold_ids <= set(stage_of):
        raise ValueError(f'{qid}: missing gold segments: {gold_ids - set(stage_of)}')
    gold = {stage_of[s] for s in gold_ids}
    support = {stage_of[str(s['id'])] for s in material['segments']
               if any(loose(v, s['text']) for v in values)} - gold
    for sess in sessions:
        for r in sess:
            r['has_answer'] = r['id'] in gold_ids
    # hist_tok is informational in this interface; the training planner uses the
    # actual Qwen tokenizer. Avoid loading a model merely to write the adapter.
    hist_tok = sum(len(re.findall(r'\w+|[^\w\s]', r['content'])) for sess in sessions for r in sess)
    return dict(question_id=qid, question_type=kind, split='', question=question,
                answer=answer, gold_stages=sorted(gold), support_stages=sorted(support),
                answer_values=list(values),
                question_date=str(material.get('cutoff') or dates[-1]),
                haystack_sessions=sessions, haystack_dates=dates, hist_tok=hist_tok,
                hist_tok_method='regex_estimate', material_id=material['material_id'],
                split_unit=material.get('split_unit', material['material_id']),
                source=material['source'], provenance='native')


def accepted_question(q, material, mined):
    segments = material['segments']
    if q.get('material_sha256') and q['material_sha256'] != material.get('immutable_sha256'):
        raise ValueError('accepted question references a different material snapshot')
    gold = q.get('positive_control_segment_ids') or q.get('gold_segments')
    if not gold:
        indices = q.get('gold_indices', q.get('support_rounds', []))
        gold = []
        for i in indices:
            deps = q.get('dependencies', {})
            dep = deps.get(str(i), []) if isinstance(deps, dict) else (deps[i] if deps else [])
            provenance = q.get('candidate_provenance', [])
            if not dep and provenance:
                dep = provenance[i].get('dependencies', [])
            gold.extend(dep or [segments[i]['id']])
    if not gold:
        facts = {f['id']: f for f in mined.get('facts', [])}
        gold = [facts[f]['segment_id'] for f in q.get('required', []) if f in facts]
    values = [c.get('value', '') for c in q.get('components', q.get('answer_atoms', []))]
    result = make_question(material, 'nat_' + q['question_id'], q['question'], q['answer'],
                           q.get('qtype', q.get('question_type', 'current')), gold, values)
    facts = {f['id']: f for f in mined.get('facts', [])}
    result['answer_evidence'] = [dict(a, **({'span': facts[a['id']]['span']}
                                          if a.get('id') in facts and facts[a['id']].get('span') else {}))
                                 for a in q.get('components', q.get('answer_atoms', []))]
    return result


def serialized_candidates(q):
    """The exact default mkdata document path, including pairing and follow-ups."""
    from k1 import rounds_of
    from pool import round_stages
    rounds, previous = rounds_of(q), fmt.round_prev(q)
    stages = round_stages(q)
    if len(rounds) != len(previous) or len(rounds) != len(stages):
        raise ValueError(f'{q["question_id"]}: candidate alignment changed')
    return [(stage, fmt.doc_str(r[1], *fmt.split_round(r[2]), prev=prev))
            for stage, r, prev in zip(stages, rounds, previous)]


def training_visible(doc, tokenizer=None, max_tokens=1100):
    # train.py removes SFX, tokenizes the whole document prefix, then caps it.
    body = doc[:-len(fmt.SFX)] if doc.endswith(fmt.SFX) else doc
    if tokenizer is None:
        return body[:max_tokens * 4000 // 1100]
    ids = tokenizer(body, add_special_tokens=False)['input_ids'][:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False)


def refilter(questions, tokenizer=None, recovered=None):
    """LLM-free relabel/drop only: ids, text, pairing and pool indices stay fixed."""
    kept, drops = [], []
    counts = Counter()
    for original in questions:
        q = dict(original)
        metadata = (recovered or {}).get(q['question_id'], {})
        if metadata and not q.get('answer_values'):
            counts['metadata_recovered'] += 1
        atoms = q.get('answer_evidence') or metadata.get('answer_evidence') or []
        values = q.get('answer_values') or metadata.get('answer_values') or []
        values = list(dict.fromkeys(str(v) for v in values if norm(v)))
        if not values:
            values = list(dict.fromkeys(str(a['value']) for a in atoms if norm(a.get('value', ''))))
        if not values:
            values = [str(q.get('answer', ''))] if norm(q.get('answer', '')) else []
            counts['literal_answer_fallback'] += 1
        q['answer_values'] = values
        if atoms:
            q['answer_evidence'] = atoms
        candidates = serialized_candidates(q)
        gold = set(q['gold_stages'])
        support = {stage for stage, doc in candidates
                   if any(loose(v, doc) for v in values)} - gold
        old = set(q.get('support_stages', []))
        counts['support_stages_added'] += len(support - old)
        counts['support_stages_removed'] += len(old - support)
        counts['support_questions_changed'] += support != old
        q['support_stages'] = sorted(support)
        full_gold = [(s, d[:-len(fmt.SFX)] if d.endswith(fmt.SFX) else d)
                     for s, d in candidates if s in gold]
        visible_gold = [(s, training_visible(d, tokenizer)) for s, d in candidates if s in gold]
        # Each value (or its literal evidence span) must survive somewhere in
        # gold, and every gold stage must retain evidence: mkdata may pick any.
        def evidence_visible(docs):
            alternatives = []
            for value in values:
                spans = [span for a in atoms if norm(a.get('value', '')) == norm(value)
                         for span in ([a.get('span', '')] + a.get('evidence_spans', [])) if norm(span)]
                alternatives.append([value] + spans)
            matches = lambda text, phrases: any(loose(p, text) for p in phrases)
            return (bool(alternatives) and gold <= {s for s, _ in docs}
                    and all(any(matches(d, ps) for _, d in docs) for ps in alternatives)
                    and all(any(matches(d, ps) for ps in alternatives) for _, d in docs))
        if not evidence_visible(full_gold):
            reason = 'fmt_gold_not_visible'
        elif not evidence_visible(visible_gold):
            reason = 'token_cap_gold_not_visible'
        else:
            kept.append(q)
            continue
        counts[reason] += 1
        drops.append(dict(question_id=q['question_id'], reason=reason, source=q.get('source')))
    report = dict(input_questions=len(questions), questions=len(kept), dropped_questions=len(drops),
                  single_fact_questions=sum(q['question_type'] == 'single_fact' for q in kept),
                  split=dict(Counter(q['split'] for q in kept)), counts=dict(counts), drops=drops,
                  protected_support_stages=sum(len(q['support_stages']) for q in kept),
                  dropped_by_source=dict(Counter(d['source'] for d in drops)),
                  visibility_method='reranker_tokenizer' if tokenizer is not None else 'char_proxy_1100_tokens_4000_chars',
                  max_document_tokens=1100)
    return kept, report


def recover_metadata(roots):
    """Old envs predate answer_values; recover accepted atoms without model calls."""
    recovered = {}
    if roots:
        materials, mined, accepted = load_pilots(roots)
        for q in accepted:
            if q['material_id'] in materials:
                native = accepted_question(q, materials[q['material_id']], mined.get(q['material_id'], {}))
                recovered[native['question_id']] = {k: native[k] for k in ('answer_values', 'answer_evidence')}
        for mid, result in mined.items():
            for fact in result.get('facts', []):
                qid = 'nat_sf_' + hashlib.sha256((mid + ':' + fact['id']).encode()).hexdigest()[:24]
                recovered[qid] = dict(answer_values=[fact.get('value', '')], answer_evidence=[
                    dict(value=fact.get('value', ''), span=fact.get('span', ''))])
    return recovered


def fact_candidates(material, mined, per_segment=2, per_material=40):
    byid = {str(s['id']): s for s in material['segments']}
    counts, result = Counter(), []
    for f in sorted(mined.get('facts', []), key=lambda f: (f.get('order', 0), f.get('span_start', 0), f['id'])):
        sid = str(f['segment_id'])
        c = f.get('confirmation', {})
        if (f.get('status') != 'asserted' or f.get('about_self') is not True
                or not all(c.get(k) is True for k in ('fits', 'holder_correct', 'status_correct'))
                or sid not in byid or not f.get('span') or f['span'] not in byid[sid]['text']
                or counts[sid] >= per_segment):
            continue
        s = byid[sid]
        result.append(dict(id=material['material_id'] + ':' + f['id'], fact=f,
                           context=s['text'], speaker=s['speaker'],
                           perspective='self' if material['source'] == 'wildchat' and s.get('role') == 'user' else 'observer'))
        counts[sid] += 1
        if len(result) >= per_material:
            break
    return result


WRITER = '''Write natural single-fact retrieval questions from confirmed native facts.
Return {questions:[{id,question}]}, exactly the supplied ids. Ask about the holder,
topic and attribute, without leaking the answer/value. Use ordinary language,
not ledger slot names. Observer: "What did Pia say about her new job?"; name
the holder, never I/my. Self (WildChat user only): "What did I say ...?".
Anchor the question to the supplied date/event when needed to distinguish later
updates. Ask only the supplied asserted fact, preserving scope/condition/time;
never turn a historical statement into a current-state claim.''' 
JUDGE = '''Independently judge each question against its confirmed fact and source.
Return {judgments:[{id,faithful:bool,answerable:bool,perspective_ok:bool}]}, all ids.
Require correct holder, attribute/topic, time/scope/condition, no answer leakage,
natural single-fact wording, literal source support, and the requested perspective.
Reject generic/ambiguous questions that could retrieve a different fact or update.
Observer must name the holder; self is only the WildChat user's own fact.'''


def generate_batch(batch, models, cache_dir):
    digest = hashlib.sha256(json.dumps([WRITER, JUDGE, batch, models.identity], sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / (digest + '.json')
    if path.exists():
        return json.loads(path.read_text())
    ids = {x['id'] for x in batch}
    def schema(key, fields):
        def valid(r):
            v = r.get(key)
            return (isinstance(v, list) and len(v) == len(ids)
                    and all(isinstance(x, dict) for x in v) and {x.get('id') for x in v} == ids
                    and all(all(isinstance(x.get(k), t) for k, t in fields.items()) for x in v))
        return valid
    try:
        written = models.ask(0, WRITER, dict(task='nat_questions', facts=batch),
                             validator=schema('questions', {'question': str}))
        votes = models.ask(2, JUDGE, dict(task='nat_intent', facts=batch, questions=written['questions']),
                           validator=schema('judgments', dict(faithful=bool, answerable=bool, perspective_ok=bool)))
        good = {v['id'] for v in votes['judgments'] if all(v[k] is True for k in ('faithful', 'answerable', 'perspective_ok'))}
        result = dict(questions=[q for q in written['questions'] if q['id'] in good and q['question'].strip()],
                      judgments=votes['judgments'], error=None)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        result = dict(questions=[], error=str(exc))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False))
    tmp.replace(path)
    return result


def convert(roots, models=None, cache_dir='nat_question_cache', batch_size=8, par=8, seed=0):
    materials, mined, accepted = load_pilots(roots)
    result, dropped = [], []
    for q in accepted:
        mid = q['material_id']
        if mid not in materials:
            raise ValueError(f'accepted {q["question_id"]}: material {mid} unavailable')
        result.append(accepted_question(q, materials[mid], mined.get(mid, {})))
    candidates = [x for mid, m in sorted(materials.items()) for x in fact_candidates(m, mined.get(mid, {}))]
    lookup = {c['id']: c for c in candidates}
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    if models is not None:
        with ThreadPoolExecutor(par) as ex:
            for i, output in enumerate(ex.map(lambda b: generate_batch(b, models, cache_dir), batches)):
                for q in output['questions']:
                    c = lookup[q['id']]
                    mid = q['id'].rsplit(':', 1)[0]
                    f = c['fact']
                    qid = 'nat_sf_' + hashlib.sha256(q['id'].encode()).hexdigest()[:24]
                    native = make_question(materials[mid], qid, q['question'], f['value'],
                                           'single_fact', [f['segment_id']], [f['value']])
                    native['answer_evidence'] = [dict(value=f['value'], span=f.get('span', ''))]
                    result.append(native)
                if output.get('error'):
                    dropped.append(dict(batch=i, error=output['error']))
                print(f'native questions batch {i + 1}/{len(batches)} kept={len(output["questions"])}', flush=True)
    # Also co-locate overlapping material windows from the same source identity;
    # split_unit is mine3's native identity and prevents cross-pilot leakage.
    units = sorted({q['split_unit'] for q in result})
    random.Random(seed).shuffle(units)
    ndev = max(1, round(len(units) * .1)) if len(units) > 1 else 0
    dev = set(units[:ndev])
    seen, unique = set(), []
    for q in result:
        if q['question_id'] in seen:
            continue
        seen.add(q['question_id'])
        q['split'] = 'dev' if q['split_unit'] in dev else 'train'
        unique.append(q)
    report = dict(materials=len(materials), mined=len(mined), accepted_questions=len(accepted),
                  eligible_single_facts=len(candidates), single_fact_questions=sum(q['question_type'] == 'single_fact' for q in unique),
                  questions=len(unique), split=dict(Counter(q['split'] for q in unique)), dropped_batches=dropped)
    return unique, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pilots', nargs='*')
    ap.add_argument('--refilter', help='offline env input; never generates questions or calls an LLM')
    ap.add_argument('--tokenizer', help='local reranker tokenizer path/name (local files only; otherwise char proxy)')
    ap.add_argument('--out', default='env_nat.json')
    ap.add_argument('--ids-prefix', default=None)
    ap.add_argument('--q27'); ap.add_argument('--gem'); ap.add_argument('--oss')
    ap.add_argument('--accepted-only', action='store_true', help='offline inspection only; training launcher requires single-fact questions')
    ap.add_argument('--cache-dir', default='nat_question_cache')
    ap.add_argument('--batch-size', type=int, default=8); ap.add_argument('--par', type=int, default=8)
    a = ap.parse_args()
    if a.batch_size < 1 or a.par < 1:
        ap.error('positive batch size and parallelism required')
    if a.refilter:
        if Path(a.refilter).resolve() == Path(a.out).resolve():
            ap.error('--out must differ from --refilter (preserve the pool parent)')
        tokenizer, tokenizer_error = None, None
        if a.tokenizer:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
            except (ImportError, OSError, ValueError) as exc:
                tokenizer_error = str(exc)
        # Fail closed (re-audit 11-09): the real training tokenizer is mandatory unless
        # NAT_ALLOW_CHAR_PROXY=1 is set explicitly (tests only).
        import os as _os
        if tokenizer is None and _os.environ.get('NAT_ALLOW_CHAR_PROXY') != '1':
            raise SystemExit('FAIL: training tokenizer unavailable for refilter: ' + str(tokenizer_error or 'no --tokenizer'))
        roots = a.pilots or sorted(p for p in Path(a.refilter).resolve().parent.glob('mine3_pilot*')
                                   if (p / 'materials.jsonl').exists() or (p / 'mine.jsonl').exists()
                                   or (p / 'checkpoints' / 'mine').is_dir())
        qs, report = refilter(json.loads(Path(a.refilter).read_text()), tokenizer, recover_metadata(roots))
        report.update(parent_env=str(Path(a.refilter).resolve()),
                      parent_md5=hashlib.md5(Path(a.refilter).read_bytes()).hexdigest(),
                      fmt_md5=hashlib.md5(Path(fmt.__file__).read_bytes()).hexdigest(),
                      tokenizer=a.tokenizer, tokenizer_error=tokenizer_error)
        output = json.dumps(qs, ensure_ascii=False)
        report['env_md5'] = hashlib.md5(output.encode()).hexdigest()
        Path(a.out).write_text(output)
        # The normal command writes nat_tr2.txt / nat_dv2.txt, retaining ids.
        prefix = a.ids_prefix or str(Path(a.out).with_name('nat'))
        for split, suffix in (('train', 'tr2'), ('dev', 'dv2')):
            Path(prefix + '_' + suffix + '.txt').write_text(''.join(q['question_id'] + '\n' for q in qs if q['split'] == split))
        Path(a.out + '.report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'drops'}))
        return
    if not a.pilots:
        ap.error('pilot directories or --refilter required')
    models = None
    if not a.accepted_only:
        from seed3_models import SeedModels
        models = SeedModels(specs=[a.q27, a.gem, a.oss], timeout=45)
    qs, report = convert(a.pilots, models, a.cache_dir, a.batch_size, a.par)
    if not qs:
        raise ValueError('no native questions survived')
    Path(a.out).write_text(json.dumps(qs, ensure_ascii=False))
    for split, suffix in (('train', 'tr'), ('dev', 'dv')):
        Path((a.ids_prefix or 'nat') + '_' + suffix + '.txt').write_text(''.join(q['question_id'] + '\n' for q in qs if q['split'] == split))
    Path(a.out + '.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
