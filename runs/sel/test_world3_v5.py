"""Fifth audit: record windows, carrier lengths, pure-length attacks and speakers."""
import copy
import json
from pathlib import Path
import re
import statistics
import tempfile
import unittest
from unittest.mock import patch

import world3 as v
from world3_semantics import parse_question,check_alignment
from world3_extra_probes import length_only_probes
import natural_stats as ns
import test_world3_v3 as fixtures


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Revision3Tests.setUpClass();cls.fixture=fixtures.Revision3Tests

    def test_every_generated_window_is_parsed_solved_and_compared(self):
        n=0
        for w in self.fixture.worlds:
            for c in w['cases']:
                if not c['date_window']:continue
                n+=1;q=parse_question(c['question'])
                self.assertEqual(q['date_window'],c['date_window'])
                self.assertEqual(v.solve(w,q),v.solve(w,c['query']))
                text=re.sub(r'Use records between [\d-]+ and [\d-]+\.',
                            'Use records between 2030-01-01 and 2030-01-02.',c['question'])
                changed=parse_question(text)
                self.assertNotEqual(changed,q)
                with self.assertRaises(ValueError):v.solve(w,changed)
                self.assertFalse(check_alignment(text,c['query'],w['facts'])['ok'])
                unwindowed=dict(c['query']);unwindowed.pop('date_window')
                # Equal answers do not excuse unequal constraints.
                self.assertFalse(check_alignment(c['question'],unwindowed,w['facts'])['ok'])
        self.assertEqual(n,36)

    def test_window_filters_by_record_date_inclusive_and_changes_gold(self):
        w={'facts':[]};c=dict(id='test',offset=40,record_time=v.day(10))
        old=v.add_version(w,c,'Ari','personal','Orion','format','PDF',when=0)[-1]
        c['record_time']=v.day(20)
        new=v.add_version(w,c,'Ari','personal','Orion','format','plain text',when=5,supersedes=old)[-1]
        q=dict(operation='direct',entity='Orion',requests=[dict(holder='Ari',scope='personal',entity='Orion',attr='format')],as_of=v.day(30),known_at=v.day(40))
        self.assertEqual(v.solve(w,q)['atom_ids'],[new])
        q['date_window']=[v.day(10)]*2
        self.assertEqual(v.solve(w,q)['atom_ids'],[old]) # events occur BEFORE this window
        q['date_window']=[v.day(20)]*2
        self.assertEqual(v.solve(w,q)['atom_ids'],[new])
        q['date_window']=[v.day(20),v.day(10)]
        with self.assertRaises(ValueError):v.solve(w,q)

    def test_native_endpoints_and_reversals_also_obey_record_window(self):
        w=copy.deepcopy(self.fixture.worlds[0])
        c=next(c for c in w['cases'] if c['query'].get('domain_program'))
        query=parse_question(c['question']);query['date_window']=[c['record_time']]*2
        atoms=v.solve(w,query)['atom_ids']
        endpoint=next(f for f in w['facts'] if f['id']==atoms[-1])
        # Remove all endpoint-slot records from the record window while keeping
        # event timestamps and the root's snapshot untouched.
        for f in w['facts']:
            if v.slot(f)==v.slot(endpoint):f['record_time']='2030-01-01'
        with self.assertRaisesRegex(ValueError,'endpoint'):v.solve(w,query)
        rows={'facts':[]};case=dict(id='rollback',offset=40,record_time=v.day(10))
        old=v.add_version(rows,case,'Ari','personal','Orion','format','PDF',when=0)[-1]
        case['record_time']=v.day(20)
        v.add_version(rows,case,'Ari','personal','Orion','format','plain text',when=5,
                      statuses=('proposed','approved','executed','rolled_back'),supersedes=old)
        q=dict(operation='update',include_reversals=True,entity='Orion',as_of=v.day(30),known_at=v.day(40),
               date_window=[v.day(10)]*2,requests=[dict(holder='Ari',scope='personal',entity='Orion',attr='format')])
        self.assertEqual(v.solve(rows,q)['atom_ids'],[old])

    def test_changed_empty_window_fails_entire_mock_gate(self):
        w=self.fixture.worlds[0]
        q=copy.deepcopy(next(q for q in self.fixture.questions if q['ledger_id']==w['ledger_id'] and q['date_window']))
        self.assertTrue(v.gates(w,[q],v.ModelFamilies(mock=True))[0]['order'])
        q['question']=re.sub(r'Use records between [\d-]+ and [\d-]+\.',
                            'Use records between 2030-01-01 and 2030-01-02.',q['question'])
        self.assertFalse(v.gates(w,[q],v.ModelFamilies(mock=True))[0]['order'])


class CarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Revision3Tests.setUpClass();cls.fixture=fixtures.Revision3Tests

    def test_lengths_after_actual_formatter_and_reply_placement(self):
        limits={'assistant':(1800,3100),'record':(2000,2800),'group':(350,800),'dyad':(300,750)}
        for carrier,(lo,hi) in limits.items():
            gs=[g for g in self.fixture.groups if g['carrier']==carrier]
            median=statistics.median(len(d) for g in gs for d in g['docs'])
            self.assertTrue(lo<=median<=hi,(carrier,median))
        replies=[]
        for w in self.fixture.worlds:
            for m in w['rounds']:
                self.assertIn(m['core'],m['text'])
                if 'assistant_reply' in m:
                    replies.append(len(m['assistant_reply']))
                    self.assertEqual(v.CORE_CLAUSE.findall(m['assistant_reply']),[])
                    self.assertGreaterEqual(len(m['text'].split()),30)
        self.assertTrue(1800<=statistics.median(replies)<=3000)

    def test_length_probe_stratifies_carriers_and_detects_middle_length_leak(self):
        report=length_only_probes(self.fixture.groups)
        self.assertEqual(set(report['by_carrier']),set(v.CARRIERS))
        self.assertTrue(report['pass'],report)
        # A middle-length signal makes both extreme rules fail, so test learning.
        gs=[dict(ledger_id=f'L{i//10}',carrier='group',label=0,
                 docs=['x'*200]+['x'*100]*7+['x'*300]*8) for i in range(40)]
        leak=length_only_probes(gs)
        self.assertFalse(leak['pass'])
        self.assertEqual(leak['by_carrier']['group']['rules']['heldout_chars_forest']['unique_rate'],1.)
        changed=[dict(g,label=7) if g['ledger_id']=='L0' else g for g in gs]
        # Independently test the model boundary with labels changed in held-out ledger.
        from world3_extra_probes import fit
        import numpy as np
        xs=[np.array([[len(d)] for d in g['docs']],float) for g in gs]
        self.assertEqual(fit(gs,xs)[:10],fit(changed,xs)[:10])

    def test_carrier_layout_preserves_frozen_v4_ledger(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('v4_ledger_test','bak_world3_v4/world3.py')
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        before=frozen.make_plan(0,20,seed=31,domain='software')
        after=v.make_plan(0,20,seed=31,domain='software')
        # Natural names/prose changed; compare every structural ledger field
        # after reversing the surface-only bijection.
        inverse={}
        for old,new in zip(before['facts'],after['facts']):
            for key in ('entity','proposal'):inverse[new[key]]=old[key]
            inverse.update(zip(new.get('links',[]),old.get('links',[])))
        restored=v.rename_values(copy.deepcopy(after['facts']),inverse)
        for fact in restored:
            fact.pop('natural_surface',None)
            fact['statement']=frozen.core(fact)
        self.assertEqual(before['facts'],restored)
        for old,new in zip(before['cases'],after['cases']):
            query=v.rename_values(new['query'],inverse)
            # v5 deliberately moved the existing record window into the
            # executable query; the frozen v4 query omits that field.
            self.assertEqual(old['query'],{k:val for k,val in query.items() if k!='date_window'})
            self.assertEqual(query.get('date_window'),old.get('date_window'))
            expected=frozen.solve(before,old['query'])
            # Temporal questions explicitly break equal-date ties by entity
            # name. Restore names BEFORE solving to compare answer order; the
            # renamed world must still require exactly the same ledger atoms.
            self.assertEqual(expected,v.solve({'facts':restored},query))
            self.assertEqual(set(expected['atom_ids']),set(v.solve(after,new['query'])['atom_ids']))

    def test_real_renderer_accepts_natural_short_reply(self):
        # MockPool simulates a remote LLM transport; models.mock=False still sends
        # budgets and validates replies, without ever making a network call.
        class Short(v.MockPool):
            def chat(self,prompt,**kwargs):
                result,family=super().chat(prompt,**kwargs)
                if '"task": "render"' in prompt:
                    rows=json.loads(result)
                    for row in rows['messages']:
                        if row.get('id') and '"render_side": "assistant"' in prompt:
                            source=json.loads(prompt.split('\nPAYLOAD\n')[1])['messages'][0]
                            if not source['core']:row['text']='Thanks.'
                    result=json.dumps(rows)
                return result,family
        w=copy.deepcopy(self.fixture.worlds[0]);w['cases']=[next(c for c in w['cases'] if c['carrier']=='assistant' and not c['assistant_only'])]
        m=v.ModelFamilies(mock=False,pools=[Short() for _ in range(3)])
        rendered=v.render(w,m)
        self.assertTrue(all(not x.get('render_error') for x in rendered['rounds'] if x['core']))
        # Repeated identical social replies still hit the independent filler
        # deduplication guard; short length alone must not reject factual rows.
        self.assertTrue(all(not x['core'] and x['render_error']=='duplicate no-new-fact filler'
                            for x in rendered['rounds'] if x.get('render_error')))
        self.assertTrue(any(x.get('assistant_reply')=='Thanks.' for x in rendered['rounds']))
        self.assertTrue(any(not x['core'] and not x.get('render_error') for x in rendered['rounds']))


class SpeakerStatsTests(unittest.TestCase):
    def test_metadata_and_known_benchmark_prefixes(self):
        # natural_stats now deliberately trusts metadata and parses prefixes
        # only for the known EverMem/LoCoMo forms. Keep world3's trusted speaker
        # metadata instead of teaching the baseline parser a generated prefix.
        for text,name,benchmark in [('[Group 1] Nadia: Hello.','Nadia','ev'),
                ('Caroline: Hi.','Caroline','loc'),('[Group 2] Mary Jane: Hello.','Mary Jane','ev')]:
            self.assertEqual(ns.turn_speaker(dict(role='user',content=text),benchmark),(name,'text_prefix'))
            self.assertEqual(ns.turn_speaker(dict(role='user',speaker='Ari',content=text),benchmark),('Ari','metadata'))
        self.assertEqual(ns.turn_speaker(dict(role='user',speaker='Ari',content='[planning/task3:thread1] Ari: Hello.')),('Ari','metadata'))
        self.assertEqual(ns.turn_speaker(dict(role='assistant',content='Hello.')),('Assistant','role_fallback'))
        docs=['<Document>: (January-20-2025) User: [Group 1] Nadia: Hi.\nAssistant: ',
              '<Document>: (January-20-2025) User: [Group 1] Alex: Hi.\nAssistant: ']
        self.assertEqual(ns.summarize([dict(docs=docs,benchmark='ev')])['same_speaker_pair_share'],0.)
        self.assertEqual(ns.summarize([dict(docs=docs,benchmark='ev',speakers=['User','User'])])['same_speaker_pair_share'],1.)

    def test_mixed_real_inputs_produce_benchmark_and_carrier_tables(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);files=[];dirs=[]
            for benchmark,names in [('ev',['Nadia','Alex']),('loc',['Caroline','Melanie'])]:
                pool=root/('pool_'+benchmark);pool.mkdir();dirs.append(pool)
                q=dict(question_id=benchmark,haystack_sessions=[[dict(role='user',content=('[Group 1] ' if benchmark=='ev' else '')+f'{names[i%2]}: message {i}') for i in range(20)]],haystack_dates=['January-20-2025'])
                file=root/f'q_train_{benchmark}.json';file.write_text(json.dumps([q]));files.append(file)
                (pool/f'{benchmark}.json').write_text(json.dumps(dict(fused=list(range(20)))))
            baseline,source=ns.pool_groups(files,dirs)
            self.assertTrue(all(s=='text_prefix' for g in baseline for s in g['speaker_sources']))
            report=ns.compare([dict(g,carrier='group' if g['benchmark']=='ev' else 'dyad') for g in baseline],baseline)
            self.assertEqual(set(report['by_benchmark']),{'ev','loc'})
            self.assertEqual(set(report['by_carrier']),{'group','dyad'})
            self.assertEqual(len(report['carrier_baseline_comparison']),2)
            self.assertLess(report['baseline']['same_speaker_pair_share'],1.)
            self.assertIn('基准分层',ns.markdown(report))

    def test_historical_reports_reuse_lengths_only(self):
        report=ns.reported_length_comparison([],sorted(Path('codex_tmp').glob('natural_real_*.md')))
        self.assertEqual(len(report['by_benchmark']),5)
        self.assertEqual(report['by_benchmark']['beam']['chars_mean'],3024.0127)
        self.assertTrue(all(s['same_speaker_pair_share'] is None for s in report['by_benchmark'].values()))
        self.assertFalse(report['source']['speaker_stats_remeasured'])
        self.assertFalse(report['real_baseline_measured'])
        self.assertFalse(report['naturalness_pass'])


if __name__=='__main__':unittest.main()
