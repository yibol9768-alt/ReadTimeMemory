"""CPU regressions for all seven audit blockers, ontology/export and recovery."""
import collections
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import fmt
import pairs3
import world3 as v
from world3_domains import DOMAINS, vocabulary
from world3_adapter import (export_s23, final_audit, groups_for, mock_pool,
                            rendered_query, shortcut_probes, training_query)


def legacy_plan(*args, **kwargs):
    """Keep the v2 contracts as compatibility regressions; v3 has a separate suite."""
    return v.make_plan(*args, **kwargs, revision=2)


class AuditFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = v.ModelFamilies(mock=True)
        cls.world = v.verify(v.render(legacy_plan(n_questions=80), cls.models, par=4), cls.models)
        cls.questions = v.build_questions(cls.world)
        cls.groups, cls.reports = [], []
        for q in cls.questions:
            gs, rs = groups_for(q, mock_pool(q), no_set=True, p=100000)
            cls.groups.extend(gs)
            cls.reports.extend(rs)

    def test_selected_uses_only_real_fmt_snippets(self):
        q = next(q for q in self.questions if q['assistant_only'])
        carriers = sorted({i for a in q['answer_atoms'] for i in a['equivalent_rounds']})
        text, covered = rendered_query(q, carriers)
        self.assertEqual(covered, set())
        self.assertTrue(any(a['statement'] in v.visible_documents(q)[carriers[0]] for a in q['answer_atoms']))
        pool = mock_pool(q)
        pool['rr8'] = {str(i): int(i in carriers) for i in pool['fused']}
        gs, _ = groups_for(q, pool, steps=50, p=100000)
        self.assertTrue(gs)
        for g in gs:
            payload = g['query'].split('<Selected>:\n', 1)[1]
            actual = {a['id'] for a in q['answer_atoms'] if a['statement'] in payload}
            self.assertEqual(set(g['covered_atoms']), actual)
            self.assertEqual(g['ncov'], len(actual))
            self.assertTrue(set(g['positive_atoms']) - actual)

    def test_selected_omission_and_snippet_length(self):
        q = copy.deepcopy(self.questions[0])
        a = q['answer_atoms'][0]
        q['haystack_sessions'] = [[dict(role='user', content=a['statement'] + ' notes' * 50)] +
                                  [dict(role='user', content='Unrelated discussion ' * 40) for _ in range(40)]]
        q['haystack_dates'] = ['January-20-2025']
        self.assertNotIn(a['id'], rendered_query(q, list(range(41)))[1])
        q['haystack_sessions'][0][0]['content'] = 'padding ' * 50 + a['statement']
        self.assertNotIn(a['id'], rendered_query(q, [0])[1])

    def test_same_values_and_real_holder_scope_time_veto_negatives(self):
        w = self.world
        for f in w['facts']:
            self.assertIn(f['value'], vocabulary(w['ontology'], f['attr']))
            self.assertNotRegex(f['statement'], r'alternative|rejected choice|emergency choice|checkpoint \d|option \d')
        for c in w['cases']:
            target = next(f for f in w['facts'] if f['id'] == c['targets'][0])
            candidates = [f for f in w['facts'] if f['id'] in c['negative_atoms']
                          and f['entity'] == target['entity'] and f['attr'] == target['attr']]
            self.assertTrue(any(f['holder'] != target['holder'] for f in candidates))
            self.assertTrue(any(f['holder'] == target['holder'] and f['scope'] != target['scope'] for f in candidates))
            self.assertTrue(any(f['status'] == 'vetoed' for f in candidates))
            self.assertTrue(any(f['status'] == 'executed' and f['event_time'] < target['event_time'] for f in candidates))
            wrong_slots = [f for f in candidates if f['holder'] != target['holder'] or f['scope'] != target['scope']]
            self.assertTrue(all(f['event_time'] == target['event_time'] for f in wrong_slots))

    def test_speakers_are_balanced_including_reported_preferences(self):
        audit = final_audit(self.groups, self.questions, self.reports)
        self.assertTrue(audit['checks']['speakers'], audit['speakers'])
        prefs = [c for c in self.world['cases'] if c['qtype'] == 'preference']
        reporters = {c['reporter'] == next(f['holder'] for f in self.world['facts'] if f['id'] == c['targets'][0]) for c in prefs}
        self.assertEqual(reporters, {True, False})
        self.assertTrue(any('said they prefer' in m['text'] and m['atoms'] for m in self.world['rounds']))
        for c in self.world['cases']:
            target = next(f for f in self.world['facts'] if f['id'] == c['targets'][0])
            if c['asker'] != target['holder']:
                neg = [m for m in self.world['rounds'] if m['case'] == c['id'] and m['atoms']
                       and m['atoms'][0] in c['negative_atoms']]
                self.assertTrue(any(m['speaker'] == c['asker'] for m in neg), c['id'])

    def test_four_probes_final_groups_and_fail_closed_audit(self):
        audit = final_audit(self.groups, self.questions, self.reports)
        self.assertEqual(len(self.groups), 80)
        self.assertFalse(audit['cpu_contract_pass'])
        self.assertFalse(audit['shortcut_probes']['group_center']['pass'])
        self.assertFalse(audit['release_ready'])
        self.assertFalse(audit['token_visibility_complete'])
        self.assertEqual(len(audit['carrier_type_perspective']), 4 * 8 * 3)
        for name in ('entity_value_vocabulary', 'speaker_is_asker', 'speaker_is_not_asker', 'no_question'):
            row = audit['shortcut_probes'][name]
            self.assertLessEqual(row['unique_rate'], row['chance'] + .10, (name, row))
        leaky = copy.deepcopy(self.groups[:1])
        leaky[0]['docs'] = [leaky[0]['entity'] + ' ' + ' '.join(leaky[0]['value_vocabulary'])] + ['nothing'] * 15
        self.assertFalse(shortcut_probes(leaky)['entity_value_vocabulary']['pass'])
        blind_leak = []
        for i in range(6):
            g = copy.deepcopy(self.groups[0])
            g.update(qid=f'blind{i}', ledger_id=f'blind{i}', docs=['canary_positive_token'] + ['neutral words'] * 15)
            blind_leak.append(g)
        self.assertEqual(shortcut_probes(blind_leak)['no_question']['unique_rate'], 1.)
        # Deliberate consumption bias must fail an actual marginal/cross check.
        biased = [g for g in self.groups if g['type'] == 'direct']
        bad = final_audit(biased, self.questions, [])
        self.assertFalse(bad['cpu_contract_pass'])
        self.assertTrue(bad['lost_cells'])

    def test_identical_visible_rounds_cannot_fill_two_negative_slots(self):
        q = next(q for q in self.questions if len(q['answer_atoms']) == 1)
        a = q['answer_atoms'][0]
        documents = v.visible_documents(q)
        by_text = collections.defaultdict(list)
        candidates = q['atom_negative_candidates'][a['id']]
        for i in candidates['contentless_ack']:
            by_text[documents[i]].append(i)
        duplicate = next(indices[:2] for indices in by_text.values() if len(indices) >= 2)
        indices, kinds, _ = v.sample_negatives(candidates, random.Random(4))
        fused = [a['equivalent_rounds'][0]] + [i for i, k in zip(indices, kinds) if k != 'contentless_ack'] + duplicate
        gs, reports = groups_for(q, dict(fused=fused, rr8={}, mock=True), no_set=True, p=100000)
        self.assertEqual(gs, [])
        self.assertEqual(reports[0]['drop'], 'structural_quota_shortfall')
        self.assertEqual(reports[0]['shortfall']['contentless_ack'], 1)

    def test_group_hard_negatives_match_the_actual_positive_attribute(self):
        by_qid = {q['question_id']: q for q in self.questions}
        facts = {f['id']: f for f in self.world['facts']}
        for g in self.groups:
            q = by_qid[g['qid']]
            a = next(a for a in q['answer_atoms'] if a['id'] == g['negative_focus_atom'])
            meta = [m for s in q['haystack_sessions'] for m in s if m['role'] == 'user']
            for i, kind in zip(g['round_indices'][1:], g['neg_src']):
                if kind in ('same_topic_wrong_scope', 'minimal_state_diff'):
                    self.assertTrue(any(facts[fid]['entity'] == a['entity'] and facts[fid]['attr'] == a['attr']
                                        for fid in meta[i]['atoms']))
            self.assertEqual(g['visible_document_count'], 16)
            self.assertEqual(len(set(g['docs'])), 16)
            self.assertTrue(all(g['negative_subtypes'][k] for k in ('vetoed', 'other_time', 'condition_or_future')))
            self.assertTrue(all(a['speaker'] == facts[a['atom']]['source']['reporter'] for a in g['atom_authors']))

    def test_recommendation_names_candidate_and_reports_failed_constraints(self):
        for q in self.questions:
            if q['question_type'] != 'recommendation':
                continue
            self.assertIn(q['query_spec']['requests'][1]['holder'], q['question'])
            for atom in q['answer_atoms']:
                self.assertIn(atom['value'], q['answer'])
            self.assertTrue(q['necessity']['ok'])
        values = set()
        for seed in range(4):
            p = legacy_plan(n_questions=80, seed=seed)
            for c in p['cases']:
                if c['qtype'] == 'recommendation':
                    attr = p['ontology']['availability']
                    f = next(f for f in p['facts'] if f['id'] in c['targets'] and f['attr'] == attr)
                    values.add(f['value'] == p['ontology']['available'])
        self.assertEqual(values, {True, False})

    def test_same_value_veto_is_not_misclassified_as_partial_gold(self):
        w = dict(facts=[])
        case = dict(id='veto', offset=30, record_time=v.day(30))
        good = v.add_version(w, case, 'A', 'personal', 'project', 'format', 'PDF', when=0)[-1]
        bad = v.add_version(w, case, 'A', 'personal', 'project', 'format', 'PDF', when=1,
                            statuses=('proposed', 'vetoed'))[-1]
        byid = {f['id']: f for f in w['facts']}
        q = copy.deepcopy(self.questions[0])
        q.update(query_spec=dict(operation='direct', entity='project', as_of=v.day(20), known_at=v.day(30),
            requests=[dict(holder='A', scope='personal', entity='project', attr='format')]),
            answer_atoms=[dict(id=good, statement=byid[good]['statement'], equivalent_rounds=[0])],
            support_rounds=[], haystack_dates=['January-20-2025'])
        q['haystack_sessions'] = [[dict(role='user', content=byid[fid]['statement'] + ' context' * 40,
            stage=fid, speaker='A', atoms=[fid]) for fid in (good, bad)]]
        categories, _, protected = v.classify_negatives(w, q)
        self.assertIn(1, categories['minimal_state_diff'])
        self.assertNotIn(1, protected)

    def test_necessity_really_resolves_remaining_visible_evidence(self):
        for q in self.questions:
            proof = q['necessity']
            self.assertTrue(proof['ok'], q['question_id'])
            self.assertFalse(proof['blind']['available'])
            self.assertEqual(proof['full']['answer'], q['answer'])
            for atom in proof['atoms'].values():
                self.assertTrue(atom['removed_rounds'])
                self.assertTrue(not atom['after']['available'] or atom['after']['answer'] != q['answer'])
        update = next(q for q in self.questions if q['question_type'] == 'update')
        reversal = next(a for a in update['answer_atoms'] if 'rolled_back' in a['statement'])
        reversal_fact = next(f for f in self.world['facts'] if f['id'] == reversal['id'])
        witness = copy.deepcopy(reversal_fact)
        witness['replacement']['value'] = '__RESTORED_VALUE_MUST_REQUIRE_ITS_OWN_EVIDENCE__'
        self.assertNotIn(witness['replacement']['value'], v.core(witness))
        self.assertTrue(update['necessity']['atoms'][reversal['id']]['after']['available'])
        self.assertNotEqual(update['answer'], update['necessity']['atoms'][reversal['id']]['after']['answer'])
        case = next(c for c in self.world['cases'] if c['id'] == update['case_id'])
        for fid in case['lifecycle_atoms']:
            f = next(f for f in self.world['facts'] if f['id'] == fid)
            self.assertTrue(f['source']['message_ids'])
        # Duplicated same-value versions are NOT necessary merely because one
        # particular atom ID/statement disappeared.
        w = copy.deepcopy(self.world)
        q = next(q for q in self.questions if q['question_type'] == 'direct' and len(q['answer_atoms']) == 1)
        primary = next(f for f in w['facts'] if f['id'] == q['answer_atoms'][0]['id'])
        old = next(f for f in w['facts'] if f['id'] == primary['supersedes'])
        for f in w['facts']:
            if f['chain'] == old['chain']:
                f['value'] = primary['value']
                f['statement'] = v.core(f)
        docs = v.visible_documents(q)
        old_original = next(f for f in self.world['facts'] if f['id'] == old['id'])['statement']
        docs = [d.replace(old_original, old['statement']) for d in docs]
        self.assertFalse(v.necessity_check(w, q, docs)['ok'])

    def test_approved_commitment_is_positive_social_ack_is_negative(self):
        byid = {f['id']: f for f in self.world['facts']}
        positives = {a['id'] for q in self.questions for a in q['answer_atoms']}
        approval = [m for m in self.world['rounds'] if m['atoms'] and m['atoms'][0] in positives
                    and m['confirmation_kind'] == 'approval_commitment']
        self.assertTrue(approval)
        self.assertTrue(all(byid[m['atoms'][0]]['status'] == 'approved' for m in approval))
        self.assertTrue(all('I approve' in m['text'] for m in approval))
        for q in self.questions:
            meta = [m for s in q['haystack_sessions'] for m in s if m['role'] == 'user']
            self.assertTrue(q['negative_candidates']['contentless_ack'])
            self.assertTrue(all(not meta[i]['atoms'] and not meta[i]['meaningful_confirmation']
                                for i in q['negative_candidates']['contentless_ack']))

    def test_supersedes_drift_and_edges_dominate_timestamp(self):
        w = dict(facts=[])
        c = dict(id='state', offset=40, record_time=v.day(40))
        old = v.add_version(w, c, 'A', 'personal', 'repo', 'policy', 'A', when=0)[-1]
        new = v.add_version(w, c, 'A', 'personal', 'repo', 'policy', 'B', when=5, supersedes=old)
        bad = copy.deepcopy(w['facts'])
        bad[-1]['supersedes'] = None
        with self.assertRaisesRegex(ValueError, 'changed supersedes'):
            v.validate_ledger(bad)
        # An ancestor's delayed execution must not resurrect a superseded value.
        w['facts'][2]['event_time'] = v.day(20)
        w['facts'][2]['statement'] = v.core(w['facts'][2])
        # Point to the older approved row (same version), preserving all edges.
        for f in w['facts'][3:]:
            f['supersedes'] = w['facts'][1]['id']
            f['replacement'] = {k: w['facts'][1][k] for k in ('value', 'event_time', 'proposal')}
            f['statement'] = v.core(f)
        self.assertTrue(v.validate_ledger(w['facts']))
        req = dict(holder='A', scope='personal', entity='repo', attr='policy')
        self.assertEqual(v.resolve(w['facts'], req, v.day(30), v.day(40))['id'], new[-1])
        # Only the old execution and the new execution are visible; the exact
        # approved atom named by the edge is absent. The displayed proposal
        # identity still binds the replacement to the old execution's version.
        query = dict(operation='direct', entity='repo', requests=[req], as_of=v.day(30), known_at=v.day(40))
        observed = v.solve_visible(w, query, [w['facts'][2]['statement'], w['facts'][-1]['statement']])
        self.assertTrue(observed['available'])
        self.assertIn('policy: B', observed['answer'])
        rollback = copy.deepcopy(w['facts'][-1])
        rollback.update(id='late_rollback', seq=99, status='rolled_back', prev=new[-1], event_time=v.day(8))
        rollback['statement'] = v.core(rollback)
        w['facts'].append(rollback)
        self.assertTrue(v.validate_ledger(w['facts']))
        query['include_reversals'] = True
        # The rollback points at the old APPROVED row, whereas the restored
        # row is its later EXECUTED state. They are the same proposal.
        self.assertIn('late_rollback', v.solve(w, query)['atom_ids'])

    def test_six_ontologies_and_disjoint_domain_holdout(self):
        self.assertEqual(len(DOMAINS), 6)
        self.assertEqual(len({o['entity'] for o in DOMAINS.values()}), 6)
        self.assertEqual(len({o['skill'] for o in DOMAINS.values()}), 6)
        for domain in DOMAINS:
            p = legacy_plan(n_questions=20, domain=domain)
            m = v.ModelFamilies(mock=True)
            w = v.verify(v.render(p, m, par=4), m)
            qs = v.build_questions(w)
            self.assertEqual(len(qs), 20, domain)
            self.assertTrue(all(q['necessity']['ok'] for q in qs))
            for f in w['facts']:
                self.assertIn(f['value'], vocabulary(w['ontology'], f['attr']))
        splits = dict(train=['software', 'hospital', 'travel', 'school'], dev=['restaurant'], audit=['laboratory'])
        v.validate_domain_splits(splits)
        for split in splits:
            self.assertIn(legacy_plan(n_questions=20, split=split, domain_splits=splits)['domain'], splits[split])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            v.validate_domain_splits(dict(train=['software'], dev=['software'], audit=['travel']))
        with self.assertRaisesRegex(ValueError, 'excluded'):
            legacy_plan(n_questions=20, domain='laboratory', domain_splits=splits)
        self.assertEqual({legacy_plan(i, 20)['domain'] for i in range(40)}, set(DOMAINS))

    def test_retry_cap_block_resume_and_quarantine(self):
        class Broken(v.MockPool):
            def chat(self, prompt, **kwargs):
                self.calls += 1
                return '{bad json', self.family
        class LoseCore(v.MockPool):
            def chat(self, prompt, **kwargs):
                payload = json.loads(prompt.split('\nPAYLOAD\n')[1])
                self.calls += 1
                return json.dumps(dict(messages=[dict(id=m['id'], text='lost') for m in payload['messages']])), self.family
        plan = legacy_plan(n_questions=20)
        # One session bundle is enough to test independent block recovery.
        plan['cases'] = plan['cases'][:1]
        plan['facts'] = [f for f in plan['facts'] if f['source']['case'] == plan['cases'][0]['id']]
        for pool_type in (Broken, LoseCore):
            with tempfile.TemporaryDirectory() as d:
                m = v.ModelFamilies(mock=True, pools=[pool_type() for _ in range(3)])
                w = v.render(plan, m, par=4, checkpoint_dir=d)
                blocks = len(list(Path(d).glob('render_m*.json')))
                block_rows=collections.defaultdict(list)
                for row in w['rounds']:block_rows[row['local_block']].append(row)
                expected=sum(3 if pool_type is Broken or r['core'] else 1 for r in w['rounds'])
                self.assertEqual(blocks,len(w['rounds']))
                self.assertEqual(sum(m.calls.values()), expected)
                # Successful empty-core rows survive a partial block failure;
                # only failed IDs are re-asked and isolated.
                self.assertTrue(all(r.get('render_error') for r in w['rounds'] if pool_type is Broken or r['core']))
                if pool_type is LoseCore:
                    self.assertTrue(all(not r.get('render_error') for r in w['rounds'] if not r['core']))
                verified = v.verify(w, m, d)
                self.assertEqual(v.build_questions(verified), [])
                before = dict(m.calls)
                v.render(plan, m, par=2, checkpoint_dir=d)
                self.assertEqual(dict(m.calls), before)
        with tempfile.TemporaryDirectory() as d:
            m = v.ModelFamilies(mock=True)
            serial = v.render(plan, m, par=1, checkpoint_dir=d)
            cached = v.render(plan, m, par=4, checkpoint_dir=d)
            self.assertEqual(serial, cached)
            before = dict(m.calls)
            a = v.verify(serial, m, d)
            after = dict(m.calls)
            b = v.verify(serial, m, d)
            self.assertEqual(a, b)
            self.assertEqual(dict(m.calls), after)
            self.assertNotEqual(before, after)
        # A crash after the first published block leaves that block reusable.
        with tempfile.TemporaryDirectory() as d:
            original = v.write
            published = []
            def crash(path, value):
                if Path(path).name.startswith('render_m'):
                    if published:
                        raise RuntimeError('simulated crash')
                    published.append(str(path))
                original(path, value)
            with patch.object(v, 'write', side_effect=crash):
                with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
                    v.render(plan, v.ModelFamilies(mock=True), par=1, checkpoint_dir=d)
            done = len(list(Path(d).glob('render_m*.json')))
            m = v.ModelFamilies(mock=True)
            v.render(plan, m, par=4, checkpoint_dir=d)
            total = len(list(Path(d).glob('render_m*.json')))
            self.assertEqual(sum(m.calls.values()), total - done)

    def test_transport_has_only_one_retry_budget_and_bounded_timeout(self):
        with patch('urllib.request.urlopen', side_effect=TimeoutError('offline timeout')) as request:
            m = v.ModelFamilies(['offline@1'] * 3, timeout=.1)
            with self.assertRaisesRegex(ValueError, 'exhausted 3 attempts'):
                m.ask(0, 'CPU transport check', dict(task='render', messages=[]))
            self.assertEqual(request.call_count, 3)
            self.assertTrue(all(call.kwargs['timeout'] == .1 for call in request.call_args_list))

    def test_malformed_verifier_schema_and_gate_indices_retry_then_isolate(self):
        class WrongSchema(v.MockPool):
            def chat(self, prompt, **kwargs):
                payload = json.loads(prompt.split('\nPAYLOAD\n')[1])
                if payload['task'] == 'verify':
                    return json.dumps(dict(ok=True, no_extra_facts=True, quotes=[])), 'bad'
                if payload['task'] == 'gate':
                    return json.dumps(dict(question_matches=True, answer_correct=True,
                        supported={a['id']: [dict(round=999999, quote=a['statement'])] for a in payload['atoms']})), 'bad'
                return super().chat(prompt, **kwargs)
        m = v.ModelFamilies(mock=True, pools=[WrongSchema() for _ in range(3)])
        w = copy.deepcopy(self.world)
        w['cases'] = w['cases'][:1]
        rejected = v.verify(w, m)
        self.assertFalse(rejected['cases'][0]['semantic_ok'])
        verify_calls = 6 * sum(r['case'] == w['cases'][0]['id'] for r in w['rounds'])
        self.assertEqual(sum(m.calls.values()), verify_calls)
        result = v.gates(self.world, self.questions[:1], m)
        self.assertFalse(result[0]['order'])
        self.assertEqual(result[0]['suff'], 1)  # raw mock reader score; semantic gate still fails
        self.assertEqual(sum(m.calls.values()), verify_calls + 6)

    def test_par_actually_overlaps_render_blocks(self):
        class Concurrent(v.MockPool):
            def __init__(self):
                super().__init__()
                self.active = self.peak = 0
                self.lock = threading.Lock()
            def chat(self, prompt, **kwargs):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(.02)
                try:
                    return super().chat(prompt, **kwargs)
                finally:
                    with self.lock:
                        self.active -= 1
        p = legacy_plan(n_questions=20)
        p['cases'] = p['cases'][:1]
        pool = Concurrent()
        v.render(p, v.ModelFamilies(mock=True, pools=[pool] * 3), par=4)
        self.assertGreater(pool.peak, 1)

    def test_s23_export_schema_reservation_and_no_fake_8k(self):
        rows = []
        for i in range(3):
            for g in self.groups:
                item = copy.deepcopy(g)
                item.update(qid=f'ledger{i}_{g["qid"]}', ledger_id=f'ledger{i}')
                rows.append(item)
        with tempfile.TemporaryDirectory() as d:
            manifest = export_s23(rows, d, final_audit(rows, self.questions, []))
            self.assertFalse(manifest['release_ready'])
            self.assertFalse(manifest['s23_budget_count_ready'])
            self.assertEqual(manifest['stage_counts'], {'1': 80, '2': 80, '3': 80})
            sets = [set(s) for s in manifest['stage_ledgers'].values()]
            self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
            for path in manifest['paths']:
                for line in Path(path).read_text().splitlines():
                    r = json.loads(line)
                    self.assertEqual(len(r['docs']), len(r['teacher']))
                    self.assertEqual(len(r['docs']), 16)
                    self.assertEqual(r['label'], 0)
                    self.assertEqual(collections.Counter(r['neg_src']), v.NEG_QUOTAS)
            bad = copy.deepcopy(rows)
            bad[0]['split'] = 'audit'
            with self.assertRaisesRegex(ValueError, 'train-only'):
                export_s23(bad, d)
            with self.assertRaisesRegex(ValueError, 'three independent'):
                export_s23(self.groups, d)
            for r in rows:
                r['mock'] = False
            with self.assertRaisesRegex(ValueError, 'actual tokenizer'):
                export_s23(rows, d)

    def test_real_byte_bpe_tokenizer_file_and_train_truncation(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        except ImportError:
            self.skipTest('optional real tokenizers package unavailable; production completion stays refused')
        with tempfile.TemporaryDirectory() as d:
            # A real serialized byte BPE fixture, not a len/character estimator.
            alphabet = pre_tokenizers.ByteLevel.alphabet()
            backend = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(alphabet)}, merges=[]))
            backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
            backend.decoder = decoders.ByteLevel()
            path = Path(d) / 'tokenizer.json'
            backend.save(str(path))
            tokenizer = v.load_tokenizer(path)
            self.assertEqual(tokenizer.world3_fingerprint, v.tokenizer_fingerprint(path))
            q = self.questions[0]
            self.assertTrue(v.check_visibility(self.world, q, token_limit=1024, tokenizer=tokenizer)['ok'])
            self.assertFalse(v.check_visibility(self.world, q, token_limit=20, tokenizer=tokenizer)['ok'])
            g, _ = groups_for(q, mock_pool(q), no_set=True, p=100000, tokenizer=tokenizer)
            self.assertEqual(len(g), 1)
            self.assertTrue(g[0]['token_visibility']['checked'])
            self.assertTrue(g[0]['token_visibility']['ok'])
            self.assertEqual(len(v.build_questions(self.world, tokenizer=tokenizer)), 80)
            checked = v.build_questions(self.world, only_case=q['case_id'], tokenizer=tokenizer)[0]
            self.assertEqual(checked['token_visibility']['tokenizer_fingerprint'], tokenizer.world3_fingerprint)
            with self.assertRaisesRegex(ValueError, 'differs from question'):
                groups_for(checked, mock_pool(checked), tokenizer=tokenizer, max_d=20)
            raw = 'prefix ' * 50 + 'middle ' * 400 + 'tail ' * 50
            ids = tokenizer.encode(raw, add_special_tokens=False)
            self.assertEqual(training_query(raw, tokenizer, 400), tokenizer.decode(ids[:200] + ids[-200:]))
            _, reports = groups_for(q, mock_pool(q), no_set=True, p=100000, tokenizer=tokenizer, max_d=20)
            self.assertEqual(reports[0]['drop'], 'token_invisible_positive')
        with self.assertRaisesRegex(ValueError, 'actual training tokenizer'):
            v.load_tokenizer(None)
        # Mock reader responses are insufficient for a production completion.
        q = copy.deepcopy(self.questions[0])
        q['mock'] = False
        m = v.ModelFamilies(mock=False, pools=[v.MockPool() for _ in range(3)])
        self.assertEqual(v.gates(self.world, [q], m)[0]['completion'], 'incomplete')
        q['token_visibility'] = dict(checked=True, ok=True, tokenizer_fingerprint=None)
        self.assertEqual(v.gates(self.world, [q], m)[0]['completion'], 'incomplete')


class PairAuditFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = v.ModelFamilies(mock=True)
        cls.qs, cls.pairs, cls.bases = pairs3.generate_pairs(140, models=cls.models)

    def test_scope_and_complex_confirmations(self):
        self.assertIn('scope', {p['intervention'] for p in self.pairs})
        self.assertTrue({'multihop', 'aggregation', 'rule', 'meaningful_confirmation'} <= {p['structure'] for p in self.pairs})
        byid = {q['question_id']: q for q in self.qs}
        self.assertTrue(any(q['meaningful_confirmations'] for q in self.qs))
        for p in self.pairs:
            l, r = byid[p['left']], byid[p['right']]
            if p['intervention'] == 'scope':
                self.assertNotEqual(l['query_spec']['requests'][0]['scope'], r['query_spec']['requests'][0]['scope'])
                self.assertNotEqual(l['answer'], r['answer'])
            if p['structure'] in ('multihop', 'aggregation'):
                self.assertGreaterEqual(len(l['answer_atoms']), 2)

    def test_reporter_keeps_actual_user_assistant_side(self):
        # Cover several random assistant carriers, including assistant-only.
        seen_assistant = False
        for seed in range(12):
            b = v.verify(v.render(pairs3.diagnostic_plan(9000 + seed, seed), self.models), self.models)
            r = pairs3.intervene(b, 'reporter')
            self.assertEqual(b['facts'], r['facts'])
            for old, new in zip(b['rounds'], r['rounds']):
                self.assertEqual(old['speaker'] == 'Assistant', new['speaker'] == 'Assistant')
                seen_assistant |= old['speaker'] == 'Assistant'
            lq, rq = v.build_questions(b)[0], v.build_questions(v.verify(r, self.models))[0]
            for la, ra in zip(lq['answer_atoms'], rq['answer_atoms']):
                self.assertEqual(la['equivalent_rounds'], ra['equivalent_rounds'])
                for i in la['equivalent_rounds']:
                    lu, ls = fmt.split_round(v.rounds_of(lq)[i][2])
                    ru, rs = fmt.split_round(v.rounds_of(rq)[i][2])
                    self.assertEqual(la['statement'] in lu, ra['statement'] in ru)
                    self.assertEqual(la['statement'] in ls, ra['statement'] in rs)
        self.assertTrue(seen_assistant)

    def test_pair_resume_skips_all_completed_calls(self):
        with tempfile.TemporaryDirectory() as d:
            m = v.ModelFamilies(mock=True)
            first = pairs3.generate_pairs(120, models=m, checkpoint_dir=d)
            before = dict(m.calls)
            second = pairs3.generate_pairs(120, models=m, checkpoint_dir=d)
            self.assertEqual(first, second)
            self.assertEqual(dict(m.calls), before)
            self.assertEqual(len(list((Path(d) / 'pairs').glob('pair*.json'))), 120)


if __name__ == '__main__':
    unittest.main()
