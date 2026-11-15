"""Offline post-step regressions; no LLM, tokenizer download, GPU or Docker."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

import fmt
import mine3_to_env as conv
import nat_artifacts as artifacts
from test_s23nat import material

HERE = Path(__file__).resolve().parent


def question(text='My editor is Emacs.', source='irc_ubuntu', qid='nat_test'):
    m = material(source=source, n=24)
    for i, seg in enumerate(m['segments']):
        seg['text'] = text if i == 0 else f'Unrelated message {i}.'
    q = conv.make_question(m, qid, 'Which editor?', 'Emacs', 'single_fact', ['0'], ['Emacs'])
    q['split'] = 'train'
    return q


def test_followup_exact_serialization_and_mkdata_negatives(tmp_path):
    from k1 import rounds_of
    from pool import round_stages
    import mkdata
    q = question()
    q['haystack_sessions'][0][2]['content'] = 'Still working on that.'
    # The direct successor carries the answer only in fmt's follow-up context.
    assert q['support_stages'] == []
    docs = conv.serialized_candidates(q)
    r = rounds_of(q)
    assert [d for _, d in docs] == [fmt.doc_str(x[1], *fmt.split_round(x[2]), prev=p)
                                  for x, p in zip(r, fmt.round_prev(q))]
    assert 'Emacs' in docs[1][1] and 'Emacs' not in r[1][2]
    kept, report = conv.refilter([q])
    assert kept[0]['support_stages'] == [docs[1][0]]
    assert report['counts']['support_stages_added'] == 1
    assert q['support_stages'] == []
    pool = dict(fused=list(range(len(r))), stage=round_stages(q),
                gold=[bool(x[3]) for x in r], rr8={str(i): .9 - i / 100 for i in range(len(r))})
    (tmp_path / 'pool').mkdir()
    (tmp_path / 'pool' / 'nat_test.json').write_text(json.dumps(pool))
    env = tmp_path / 'env.json'
    env.write_text(json.dumps(kept))
    out = tmp_path / 'train.jsonl'
    mkdata.main([str(env), str(tmp_path / 'pool'), str(out), '--no-set', '--ignore-kinds', 'repeat,fu',
                 '--hard-skip', '0', '--boost', '3', '--steps', '8'])
    groups = [json.loads(s) for s in out.read_text().splitlines()]
    assert len(groups) == 8
    assert all('Emacs' not in d for g in groups for d in g['docs'][1:])


def test_wildchat_pair_assistant_context_and_session_reset():
    q = question(source='wildchat')
    q['haystack_sessions'][0][1]['content'] = 'It is called Emacs.'
    kept, _ = conv.refilter([q])
    # User+assistant are one stage; the next paired round inherits previous user.
    assert kept[0]['support_stages'] == ['nat:m0:2']
    q = question(source='blogs')  # blog entries each start a fresh session
    assert conv.refilter([q])[0][0]['support_stages'] == []


def test_followup_context_cap_and_marker_cleaning():
    q = question('x' * 210 + ' Emacs')
    q['haystack_sessions'][0][2]['content'] = 'Again?'
    # Previous text exists, but its answer falls beyond fmt's 200-char context.
    assert conv.refilter([q])[0][0]['support_stages'] == []
    q = question('Editor ->-> 2,10 Emacs')
    kept, _ = conv.refilter([q])
    assert kept and kept[0]['support_stages'] == ['nat:m0:1']


def test_fmt_truncation_drops_gold_and_removes_invisible_support():
    q = question('padding ' * 500 + 'My editor is Emacs.')
    assert 'Emacs' in q['haystack_sessions'][0][0]['content']
    kept, report = conv.refilter([q])
    assert not kept
    assert report['counts']['fmt_gold_not_visible'] == 1
    q = question()
    q['haystack_sessions'][0][4]['content'] = 'padding ' * 500 + 'Emacs'
    q['support_stages'] = ['nat:m0:4']
    kept, _ = conv.refilter([q])
    assert 'nat:m0:4' not in kept[0]['support_stages']


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        assert not text.endswith(fmt.SFX)
        return {'input_ids': list(map(ord, text))}

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is False
        return ''.join(map(chr, ids))


def test_token_cap_includes_doc_header_and_fmt_suffix_is_removed():
    q = question('x' * 1300 + ' Emacs')
    assert conv.refilter([q])[0]  # survives 3000-char fmt and 4000-char proxy
    kept, report = conv.refilter([q], CharacterTokenizer())
    assert not kept
    assert report['counts']['token_cap_gold_not_visible'] == 1
    q = question('x' * 1000 + ' Emacs')
    assert conv.refilter([q], CharacterTokenizer())[0]
    assert len(conv.training_visible('a' * 4100 + fmt.SFX)) == 4000


def test_span_alternative_multi_value_and_every_gold_candidate():
    q = question('A peculiar editor is installed.')
    q['answer_evidence'] = [dict(value='Emacs', span='peculiar editor')]
    assert conv.refilter([q])[0]
    q['answer_values'].append('Vim')
    assert not conv.refilter([q])[0]  # every answer atom must survive
    q = question()
    q['gold_stages'].append('nat:m0:4')
    assert not conv.refilter([q])[0]  # mkdata could choose this answerless gold


def run_refilter(tmp_path, qs):
    env = tmp_path / 'env_nat.json'
    env.write_text(json.dumps(qs))
    subprocess.run([sys.executable, '-B', str(HERE / 'mine3_to_env.py'),
                    '--refilter', str(env), '--out', str(tmp_path / 'env_nat2.json')],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, 'NAT_ALLOW_CHAR_PROXY': '1'})
    return json.loads((tmp_path / 'env_nat2.json.report.json').read_text())


def test_cli_no_llm_parent_hash_ids_counts_and_pool_reuse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    good = question()
    del good['answer_values']  # existing prepare uses the old converter
    bad = question('padding ' * 500 + 'Emacs', qid='nat_drop')
    report = run_refilter(tmp_path, [good, bad])
    assert report['parent_md5'] == artifacts.md5('env_nat.json')
    assert report['questions'] == report['dropped_questions'] == 1
    assert report['counts']['literal_answer_fallback'] == 1
    assert Path('nat_tr2.txt').read_text() == 'nat_test\n'
    assert Path('nat_dv2.txt').read_text() == ''
    Path('pool_nat').mkdir()
    Path('pool_nat/env_md5.txt').write_text(report['parent_md5'])
    Path('pool_nat/nat_test.json').write_text('{}')
    before = Path('pool_nat/env_md5.txt').read_bytes()
    assert artifacts.validate() == report
    assert Path('pool_nat/env_md5.txt').read_bytes() == before
    Path('pool_nat/env_md5.txt').write_text('wrong')
    with pytest.raises(AssertionError, match='pool parent'):
        artifacts.validate()


def test_legacy_composite_metadata_recovered_from_local_pilot(tmp_path):
    q = question()
    q['question_id'] = 'nat_old'
    q['question_type'] = 'compare'
    q['answer'] = 'changed from Emacs to Vim'
    q['haystack_sessions'][0][0]['content'] = 'Emacs then Vim'
    del q['answer_values']
    pilot = tmp_path / 'mine3_pilot_old'
    pilot.mkdir()
    m = material(source='irc_ubuntu')
    (pilot / 'materials.jsonl').write_text(json.dumps(m) + '\n')
    accepted = dict(question_id='old', material_id='m0', question='Which editors?',
                    answer=q['answer'], qtype='compare', gold_segments=['0'],
                    components=[dict(value='Emacs'), dict(value='Vim')])
    (pilot / 'export_questions.json').write_text(json.dumps([accepted]))
    report = run_refilter(tmp_path, [q])
    assert report['questions'] == 1
    kept = json.loads((tmp_path / 'env_nat2.json').read_text())
    assert kept[0]['answer_values'] == ['Emacs', 'Vim']


def test_legacy_single_fact_recovers_span_without_repeating_llm(tmp_path):
    q = question('A peculiar editor is installed.')
    q['question_id'] = 'nat_sf_' + hashlib.sha256(b'm0:f0').hexdigest()[:24]
    del q['answer_values']
    pilot = tmp_path / 'mine3_pilot_old'
    pilot.mkdir()
    mined = dict(material=material(), facts=[dict(id='f0', value='Emacs', span='peculiar editor')])
    (pilot / 'mine.jsonl').write_text(json.dumps(mined) + '\n')
    report = run_refilter(tmp_path, [q])
    assert report['questions'] == 1
    assert report['counts']['metadata_recovered'] == 1


def test_cli_local_tokenizer_visibility_and_fail_closed(tmp_path):
    env = tmp_path / 'env_nat.json'
    env.write_text(json.dumps([question('x' * 1300 + ' Emacs')]))
    module = tmp_path / 'transformers.py'
    module.write_text('class AutoTokenizer:\n'
                      ' @classmethod\n'
                      ' def from_pretrained(cls, name, local_files_only):\n'
                      '  assert local_files_only is True\n'
                      '  if name == "missing": raise OSError("fixture unavailable")\n'
                      '  return cls()\n'
                      ' def __call__(self, text, add_special_tokens):\n'
                      '  return {"input_ids": list(map(ord, text))}\n'
                      ' def decode(self, ids, skip_special_tokens):\n'
                      '  return "".join(map(chr, ids))\n')
    proc_env = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    proc_env.pop('NAT_ALLOW_CHAR_PROXY', None)
    out = tmp_path / 'env_nat2.json'
    command = [sys.executable, '-B', str(HERE / 'mine3_to_env.py'), '--refilter', str(env), '--out', str(out), '--tokenizer']
    subprocess.run(command + ['fixture'], env=proc_env, check=True, capture_output=True)
    report = json.loads(Path(str(out) + '.report.json').read_text())
    assert report['visibility_method'] == 'reranker_tokenizer'
    assert report['dropped_questions'] == 1
    before = out.read_bytes()
    for argv in (command + ['missing'], command[:-1]):
        failed = subprocess.run(argv, env=proc_env, capture_output=True, text=True)
        assert failed.returncode != 0
        assert 'FAIL: training tokenizer unavailable' in failed.stderr
        assert out.read_bytes() == before
    # Explicit opt-in remains available for offline unit fixtures only.
    subprocess.run(command + ['missing'], env={**proc_env, 'NAT_ALLOW_CHAR_PROXY': '1'}, check=True, capture_output=True)
    report = json.loads(Path(str(out) + '.report.json').read_text())
    assert report['questions'] == 1
    assert report['tokenizer_error'] == 'fixture unavailable'


def test_publish_archives_only_stale_native_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = run_refilter(tmp_path, [question()])
    Path('pool_nat').mkdir()
    Path('pool_nat/env_md5.txt').write_text(report['parent_md5'])
    Path('pool_nat/nat_test.json').write_text('{}')
    root = 's23/lobo'
    oldpaths = ['s23/nat_s1_s0', 'ckpt/s23_s1_nat_s0', 'ckpt/s23_s1_nat_s0_native',
                root + '/nat_loc_s0', root + '/nat_ev_s0']
    for path in oldpaths + [root + '/yi_loc_s0']:
        Path(path).mkdir(parents=True)
        Path(path, 'sentinel').write_text('old')
    Path('tr_nat.jsonl').write_text('old groups')
    row = dict(qid='nat_test', query='q', docs=['d'] * 5, teacher=[.5] * 5, label=0)
    new = json.dumps(row) + '\n'
    Path('new.jsonl').write_text(new)
    artifacts.publish('new.jsonl', root)
    artifacts.check()
    assert Path('tr_nat.jsonl').read_text() == new
    assert Path(root, 'yi_loc_s0/sentinel').read_text() == 'old'
    assert not any(Path(p).exists() for p in oldpaths)
    archive = next(Path('s23/nat_archive').iterdir())
    assert all((archive / p / 'sentinel').read_text() == 'old' for p in oldpaths)
    # An identical rebuild retains valid native plans/checkpoints.
    Path(oldpaths[0]).mkdir()
    Path(oldpaths[0], 'sentinel').write_text('new')
    Path('new.jsonl').write_text(new)
    artifacts.publish('new.jsonl', root)
    assert Path(oldpaths[0], 'sentinel').read_text() == 'new'
