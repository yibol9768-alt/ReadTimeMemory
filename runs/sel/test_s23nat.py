"""Fast CPU-only native experiment checks. No model, network, Docker or GPU."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

import budget_sample as bs
import mine3_to_env as conv
import nat_runtime as runtime
import s23
import s23nat as nat
from test_budget import environment, row
from test_lobo import args, sample


@pytest.fixture
def native_environment(environment, monkeypatch):
    for key in ('source', '_original_s1_flops'):
        monkeypatch.setattr(nat, key, s23.source if key == 'source' else s23.s1_flops)
    monkeypatch.setattr(nat, '_QUESTION_IDS_CACHE', {})
    def flops(path, world, accum, seed, epochs=1, w_base=0):
        assert accum == 4
        # Native rows are cheaper, forcing cycles for FLOPs matching.
        rows = nat.lines(path)
        consumed = (int(len(rows) // world * epochs) // accum) * accum * world
        weight = .6 if all(json.loads(r)['qid'].startswith('nat_') for r in rows) else 1
        return consumed * weight * (1 + bool(w_base) / 3)
    monkeypatch.setattr(nat, 'flops', flops)
    Path('tr_nat.jsonl').write_text(''.join(row(f'nat_sf_{i}', i) for i in range(37)))
    return environment


@pytest.mark.parametrize('fold', ('loc', 'ev'))
def test_nat_sources_budget_and_repetitions(native_environment, fold):
    a = args('nat', fold)
    p = nat.plan_budget(a, sample(a))
    assert .95 <= p['ratio'] <= 1.05
    assert p['fill_teacher_per_bench'] == 0
    assert p['start'] == '/w/ckpt/s23_s1_nat_s0'
    assert p['heldout_groups'] == 0
    assert p['native']['source'] == 'tr_nat.jsonl'
    assert p['native']['stages23_repetition_factor'] > 1
    recipe = nat.nat_stage1(0)
    assert .99 <= recipe['ratio'] <= 1.01
    assert [(s['world'], s['epochs']) for s in recipe['stages']] == [(4, 2), (2, 1)]
    source = set(nat.lines('tr_nat.jsonl'))
    for st in recipe['stages']:
        assert set(nat.lines(st['data'])) <= source
        assert st['repetition_factor'] > 0
    assert recipe['total_repetition_factor'] > 1
    for stage in (2, 3):
        rows = nat.lines(p[f's{stage}'])
        native = [r for r in rows if json.loads(r)['qid'].startswith('nat_')]
        assert native and set(native) <= source
        assert not any(json.loads(r)['qid'].startswith('syn') for r in rows)
        assert p['per_bench_groups'][fold] == dict(teacher2=0, teacher3=0, gold3=0, fill2=0)
    assert nat.plan_budget(a, sample(a)) == p
    # The non-heldout real blocks and selected yi checkpoint policy survive.
    yi = nat.plan_budget(args('yi', fold), sample(args('yi', fold)))
    for key in ('real_teacher_s2_md5', 'real_gold_s3_md5'):
        assert p['base'][key] == yi['base'][key]
    assert p['stage2_checkpoint'] == yi['stage2_checkpoint']


def test_stage1_commands_use_native_views_only(native_environment, monkeypatch):
    import preflight_s23
    recipe = nat.nat_stage1(0)
    calls = []
    monkeypatch.setattr(nat, 'preflight', lambda *a, **k: {})
    monkeypatch.setattr(preflight_s23, 'checkpoint_recipe', lambda *a, **k: None)
    monkeypatch.setattr(nat, 'budget_stage1_actual', lambda a: 1)
    monkeypatch.setattr(nat, 'torchrun', lambda g, out, argv, log, **kw: calls.append((g, out, argv)))
    nat.nat_s1(args('nat'))
    assert [g for g, _, _ in calls] == ['0,1,2,3', '0,1']
    assert [c[2][c[2].index('--data') + 1] for c in calls] == ['/w/' + s['data'] for s in recipe['stages']]
    assert calls[0][2][calls[0][2].index('--model') + 1] == nat.A0
    assert calls[1][2][calls[1][2].index('--model') + 1] == '/w/ckpt/s23_s1_nat_s0_native'
    assert '--w-base' in calls[1][2]


@pytest.mark.parametrize('arm', ('yi', 'jia', 'jiasmall'))
def test_other_arm_plans_byte_identical(native_environment, monkeypatch, arm):
    # Give both implementations precisely the same FLOPs oracle.
    monkeypatch.setattr(s23, 'flops', nat.flops)
    a = args(arm, 'loc')
    p = s23.plan_budget(a, sample(a))
    frozen_bytes = Path(s23.budget_dir(a), 'plan.json').read_bytes()
    assert nat.plan_budget(a, sample(a)) == p
    assert Path(s23.budget_dir(a), 'plan.json').read_bytes() == frozen_bytes
    assert nat.all_start(a) == s23.all_start(a)


def test_original_file_and_unrelated_functions_unchanged():
    original = Path(s23.__file__).read_bytes()
    assert hashlib.md5(original).hexdigest() == '31359d7eef61432441c21b66894af99b'
    old = {n.name: ast.get_source_segment(original.decode(), n) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.get_source_segment(Path(nat.__file__).read_text(), n) for n in ast.parse(Path(nat.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    touched = {'build', 'build_budget', 'flops', 'main', 'prepare_budget', 'plan_budget', 'budget', 's1_flops', 'all_start', 'budget_stage1_actual', 'torchrun', 'inner_score'}
    assert all(new[k] == value for k, value in old.items() if k not in touched)


def material(mid='m0', source='blogs', n=24):
    return dict(material_id=mid, source=source, cutoff='2026-01-03', split_unit=mid,
                segments=[dict(id=str(i), text=('I got a new job in New-York.' if i in (0, 2) else f'Unrelated message {i}.'),
                               speaker='Pia', time='2026-01-01', role='user' if i % 2 == 0 else 'assistant') for i in range(n)])


def fact(i=0, **kw):
    return dict(id=f'f{i}', holder='Pia', attribute='job', value='New York', segment_id=str(i),
                span='new job', status='asserted', about_self=True, time='2026-01-01',
                confirmation=dict(fits=True, holder_correct=True, status_correct=True), **kw)


def test_env_format_sessions_pairs_gold_support():
    from k1 import rounds_of
    from h20copy_pool import round_stages
    m = material()
    q = conv.make_question(m, 'nat_test', 'What did Pia say about her job?', 'New York', 'single_fact', ['0'], ['New York'])
    assert len(q['haystack_sessions']) == 24
    assert q['haystack_sessions'][0][0]['content'].startswith('Author: Pia: ')
    assert q['gold_stages'] == ['nat:m0:0']
    assert q['support_stages'] == ['nat:m0:2']
    assert rounds_of(q)[0][3] == 1
    assert len(round_stages(q)) == len(rounds_of(q))
    export = dict(question_id='export', question='What did Pia say?', answer='New York',
                  question_type='history', support_rounds=[0],
                  candidate_provenance=[dict(dependencies=['0', '1'])])
    assert conv.accepted_question(export, m, {})['gold_stages'] == ['nat:m0:0', 'nat:m0:1']
    m['source'] = 'wildchat'
    q = conv.make_question(m, 'nat_test', 'What did I say?', 'answer', 'single_fact', ['1'])
    assert len(q['haystack_sessions']) == 1
    assert len(rounds_of(q)) == 12
    assert round_stages(q)[0] in q['gold_stages']
    assert rounds_of(q)[0][3] == 1
    assert conv.fact_candidates(m, dict(facts=[fact()]))[0]['perspective'] == 'self'
    m['source'] = 'irc_ubuntu'
    q = conv.make_question(m, 'nat_test', 'What did Pia say?', 'answer', 'single_fact', ['1'])
    assert list(map(len, q['haystack_sessions'])) == [20, 4]
    assert len(rounds_of(q)) == 24
    assert q['haystack_sessions'][0][0]['content'].startswith('Pia: ')


class MockModels:
    identity = ['mock0', 'mock1', 'mock2']
    def __init__(self, fail=False): self.families, self.fail = [], fail
    def ask(self, family, instructions, payload, validator=None):
        self.families.append(family)
        if self.fail: raise ValueError('mock failed')
        if family == 0:
            result = {'questions': [dict(id=f['id'], question='What did Pia say about her new job?') for f in payload['facts']]}
        else:
            result = {'judgments': [dict(id=f['id'], faithful=True, answerable=True, perspective_ok=True) for f in payload['facts']]}
        assert validator(result)
        return result


def test_converter_checkpoint_loading_split_and_llm_drop(tmp_path):
    pilot = tmp_path / 'pilot'
    checkpoint = pilot / 'checkpoints' / 'mine'
    checkpoint.mkdir(parents=True)
    mats = [material(f'm{i}') for i in range(10)]
    (pilot / 'materials.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in mats))
    for m in mats:
        (checkpoint / (m['material_id'] + '.json')).write_text(json.dumps(dict(result=dict(material=m, facts=[fact()], updates=[], ok=True))))
    accepted = dict(question_id='accepted', material_id='m0', accepted=True, qtype='history',
                    question='What did Pia say?', answer='New York', gold_indices=[0], dependencies=[['0']])
    (pilot / 'verify.jsonl').write_text(json.dumps(accepted) + '\n' + json.dumps({**accepted, 'question_id': 'bad', 'accepted': False}) + '\n')
    models = MockModels()
    qs, report = conv.convert([pilot], models, tmp_path / 'cache', par=1)
    assert report['single_fact_questions'] == 10
    assert set(models.families) == {0, 2}
    assert len(qs) == 11
    owners = {}
    for q in qs: owners.setdefault(q['material_id'], set()).add(q['split'])
    assert all(len(s) == 1 for s in owners.values())
    assert sum(v == {'dev'} for v in owners.values()) == 1
    qs, report = conv.convert([pilot], MockModels(True), tmp_path / 'failcache', par=1)
    assert len(qs) == 1 and report['single_fact_questions'] == 0
    assert report['dropped_batches']


def test_fact_caps_and_confirmation():
    m = material(n=50)
    for s in m['segments']: s['text'] = 'I got a new job in New York.'
    facts = [{**fact(), 'id': f'f{i}', 'segment_id': str(i // 3)} for i in range(150)]
    assert len(conv.fact_candidates(m, dict(facts=facts))) == 40
    counts = {}
    for f in conv.fact_candidates(m, dict(facts=facts)):
        sid = f['fact']['segment_id']; counts[sid] = counts.get(sid, 0) + 1
    assert max(counts.values()) == 2
    assert conv.fact_candidates(m, dict(facts=[{**fact(), 'confirmation': {}}])) == []


def test_summary_uses_original_paired_bootstrap(tmp_path):
    import nat_summary
    for fold in ('loc', 'ev'):
        for arm in ('nat', 'yi', 'jia', 'jiasmall'):
            area = 52 if arm == 'nat' else 50
            report = dict(version=bs.LOBO_VERSION, arm=arm, fold=fold, seed=0,
                          plan=dict(sample={'same': True}, data_freeze_md5='same', ratio=1, native={}),
                          inference={}, inner_per_question={b: {'q1': area, 'q2': area + 1} for b in bs.BENCH},
                          heldout_area=area + .5, planned_flops=100, actual_flops=100)
            path = tmp_path / f'{arm}_{fold}_s0'
            path.mkdir()
            (path / 'results.json').write_text(json.dumps(report))
    result = nat_summary.compare(tmp_path)
    expected = bs.paired_bootstrap({'loc': {'q1': 52, 'q2': 53}}, {'loc': {'q1': 50, 'q2': 51}}, strata=('loc',))
    c = result['contrasts']['nat-yi']['folds']['loc']
    assert all(c[k] == v for k, v in expected.items())
    path = tmp_path / 'yi_loc_s0/results.json'
    broken = json.loads(path.read_text())
    broken['inference'] = {'different': True}
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='incompatible'):
        nat_summary.compare(tmp_path)


def test_runtime_deadline_log_and_no_next_day(monkeypatch):
    text = '[23:50:00 SGT] START yi loc\n[00:20:00 SGT] S23LOBO_DONE s23/yi_loc_s0\n[00:25:00 SGT] START yi ev\n[00:55:00 SGT] DONE yi ev'
    assert runtime.log_durations(text) == [1800, 1800]
    now = dt.datetime(2026, 11, 8, 7, tzinfo=runtime.SGT)
    assert runtime.deadline(now) < now.timestamp()
    monkeypatch.setattr(runtime.time, 'time', lambda: now.timestamp())
    with pytest.raises(ValueError, match='REFUSE'):
        runtime.require_time(300, 'training', dict(deadline_epoch=runtime.deadline(now)))


@pytest.mark.parametrize('fold', ('loc', 'ev'))
def test_natj_replaces_only_jia_fill(native_environment, monkeypatch, fold):
    from collections import Counter
    a = args('natj', fold)
    monkeypatch.setattr(nat, 'nat_stage1', lambda *a: pytest.fail('natj planned stage 1'))
    jia = nat.plan_budget(args('jia', fold), sample(a))
    original = {p: Path(p).read_bytes() for p in (jia['s2'], jia['s3'], nat.budget_dir(args('jia', fold)) + '/plan.json')}
    p = nat.plan_budget(a, sample(a))
    assert p['start'] == jia['start'] == nat.A0
    assert nat.budget_stage1_actual(a) == 0
    assert p['planned_stage_flops']['stage1'] == 0
    assert p['native']['stage1'] is None
    assert .95 <= p['ratio'] <= 1.05 and p['C_yi'] == jia['C_yi']
    assert 0 <= p['fill_teacher_per_bench'] < jia['fill_teacher_per_bench']
    removed = jia['fill_teacher_per_bench'] - p['fill_teacher_per_bench']
    assert p['teacher_fill_difference_vs_jia'] == dict(per_bench=-removed, total_rows=-4 * removed)
    assert p['jia_plan']['fill_teacher_per_bench'] == jia['fill_teacher_per_bench']
    assert p['stage2_checkpoint'] == jia['stage2_checkpoint']
    for key in ('real_teacher_s2_md5', 'real_gold_s3_md5', 'base_counts'):
        assert p['base'][key] == jia['base'][key]
    native_source = set(nat.lines('tr_nat.jsonl'))
    for stage, expected in ((2, nat.PER_STAGE), (3, nat.PER_STAGE - 4 * 800)):
        rows = nat.lines(p[f's{stage}'])
        native = [r for r in rows if json.loads(r)['qid'].startswith('nat_')]
        assert len(native) == expected
        assert Counter(native) == Counter(nat.take('tr_nat.jsonl', expected, 0 if stage == 2 else nat.PER_STAGE, a.seed))
        assert set(native) <= native_source
        assert p['native'][f'stage{stage}_rows'] == expected
        assert p['native'][f'stage{stage}_repetition_factor'] == expected / 37
        gold = lambda path: Counter(r for r in nat.lines(path) if not json.loads(r).get('nolabel') and not json.loads(r)['qid'].startswith('nat_'))
        assert gold(p[f's{stage}']) == gold(jia[f's{stage}'])
    # The exact retained teacher multiset is jia's base + a shorter cyclic fill.
    data = nat.prepare_budget(args('jia', fold), sample(a))
    expected = Counter()
    for block in data['teacher2'].values():
        if block:
            expected.update(block + [block[i % len(block)] for i in range(p['fill_teacher_per_bench'])])
    observed = Counter(r for r in nat.lines(p['s2']) if json.loads(r).get('nolabel'))
    assert observed == expected
    assert p['heldout_groups'] == 0
    assert not any(p['per_bench_groups'][fold].values())
    assert nat.plan_budget(a, sample(a)) == p
    assert all(Path(path).read_bytes() == content for path, content in original.items())


def test_natj_runtime_never_reads_stage1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for fold in ('loc', 'ev'):
        folder = tmp_path / f'natj_{fold}_s0'
        folder.mkdir()
        (folder / 'plan.json').write_text(json.dumps(dict(planned_stage_flops=dict(stage1=0, stage2=100, stage3=200))))
    config = dict(conservative_flops_per_second=1)
    assert runtime.phase_seconds(config, 'all', tmp_path, 'natj') == 1500
    assert runtime.phase_seconds(config, 'full', tmp_path, 'natj') == 600


def test_summary_natj(tmp_path):
    import nat_summary
    for fold in ('loc', 'ev'):
        for arm in ('natj', 'yi', 'jia', 'jiasmall'):
            area = 52 if arm == 'natj' else 50
            report = dict(version=bs.LOBO_VERSION, arm=arm, fold=fold, seed=0,
                          plan=dict(sample={'same': True}, data_freeze_md5='same', ratio=1, native={}),
                          inference={}, inner_per_question={b: {'q1': area} for b in bs.BENCH},
                          heldout_area=area, planned_flops=100, actual_flops=100)
            path = tmp_path / f'{arm}_{fold}_s0'
            path.mkdir()
            (path / 'results.json').write_text(json.dumps(report))
    result = nat_summary.compare(tmp_path, arm='natj')
    assert set(result['contrasts']) == {'natj-jia', 'natj-yi', 'natj-jiasmall'}
    assert result['contrasts']['natj-jia']['folds']['loc']['difference'] == 2


def test_natj_full_uses_jia_choice_and_writes_results(native_environment, monkeypatch):
    from test_lobo import fake_execution
    choice_path = Path(nat.choice_file(args('jia')))
    choice = json.loads(choice_path.read_text())
    choice['stage2_checkpoint'] = 'end'  # yi remains mid; catch accidental yi routing.
    choice_path.write_text(json.dumps(choice))
    calls = fake_execution(monkeypatch, {'natj': 100})
    monkeypatch.setattr(nat, 'train_all_stage', s23.train_all_stage)
    monkeypatch.setattr(nat, 'inner_score', s23.inner_score)
    monkeypatch.setattr(nat, 'preflight', lambda *a: None)
    monkeypatch.setattr(nat, 'nat_stage1', lambda *a: pytest.fail('natj ran stage 1'))
    a = args('natj', 'loc')
    result = nat.lobo(a)
    assert result['stage2_checkpoint'] == result['plan']['stage2_checkpoint'] == 'end'
    assert calls[0] == ('natj', 2, nat.A0)
    assert calls[1][2] == '/w/' + result['stage2_candidates'][-1]['path']
    assert result['actual_stage_flops']['stage1'] == 0
    assert result['actual_flops'] == pytest.approx(result['planned_flops'])
    assert Path(nat.budget_dir(a), 'results.json').is_file()
    assert result['inference'] == nat.budget_choices(a)['yi']['inference']['options']
    assert set(result['inner_per_question']) == set(bs.BENCH)
    assert nat.lobo(a) == result and len(calls) == 2



def test_legacy_build_natj_and_other_arms_unchanged(native_environment):
    from collections import Counter
    Path('trx.jsonl').write_bytes(Path('tr_w20.jsonl').read_bytes())
    # Existing legacy arm files/exposure audits retain exactly their old bytes.
    for arm in ('yi', 'jia', 'wu', 'bing', 'jiap'):
        a = nat.clone(args(arm, 'loc'), all_data=False, r=.25)
        paths = s23.build(a, 123)
        before = [Path(path).read_bytes() for path in (*paths, nat.wd(a) + '/exposures.json')]
        actual = nat.build(a, 123)
        assert [Path(path).read_bytes() for path in (*actual, nat.wd(a) + '/exposures.json')] == before
    a = nat.clone(args('natj', 'loc'), all_data=False, r=.25)
    jia2, jia3 = nat.build(nat.clone(a, arm='jia'), 1000)
    p2, p3 = nat.build(a, 100)
    for stage, native_count, path, jia_path in ((2, 4000, p2, jia2), (3, 3000, p3, jia3)):
        rows = nat.lines(path)
        native = [r for r in rows if json.loads(r)['qid'].startswith('nat_')]
        assert len(native) == native_count
        assert set(native) <= set(nat.lines('tr_nat.jsonl'))
        gold = lambda p: Counter(r for r in nat.lines(p) if not json.loads(r).get('nolabel') and not json.loads(r)['qid'].startswith('nat_'))
        assert gold(path) == gold(jia_path)
    assert Counter(r for r in nat.lines(p3) if json.loads(r).get('nolabel')) == Counter(r for r in nat.lines(jia3) if json.loads(r).get('nolabel'))
