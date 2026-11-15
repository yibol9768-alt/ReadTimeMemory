"""Offline regression for the second real-LLM rendering fix."""
import collections
import copy
import json
from pathlib import Path
import tempfile
import unittest

import world3 as v
from world3_style import (anchors_for, anchors_present, atom_present, assemble_turns,
                          message_failures, valid_message, match_lengths, dedupe_fillers)


class Fix2Tests(unittest.TestCase):
    def test_replay_captured_natural_dates_and_dependency_labels(self):
        captured=json.loads(Path(__file__).with_name('dbg_v5_1108.json').read_text())
        # These captured rows retain their facts. Replay old required dates too,
        # although date binding now belongs to semantic verification.
        dates=dependencies=0
        for sample in captured['samples']:
            for spans in sample['anchors'].values():
                if 'On January' in sample['text'] and anchors_present([spans[0],spans[2],spans[3]],sample['text']):
                    if spans[4][5:7]=='01' and (f'January {int(spans[4][-2:])}' in sample['text']):
                        with self.subTest(text=sample['text'][:80],date=spans[4]):
                            self.assertTrue(anchors_present([spans[0],spans[2],spans[3],spans[4]], sample['text']))
                            dates+=1
                if spans[1]=='component_dependency_version' and 'component dependency version' in sample['text']:
                    self.assertTrue(anchors_present([spans[0],spans[1],spans[2],spans[3],spans[4],'hotfix'], sample['text']))
                    dependencies+=1
        self.assertGreaterEqual(dates,2)
        self.assertGreaterEqual(dependencies,1)

    def test_normalizer_variants_and_near_misses(self):
        for date in ['January 6','Jan 6, 2025','6 January','January\u202f6th, 2025','2025/01/06']:
            self.assertTrue(anchors_present(['2025-01-06'],date),date)
        self.assertFalse(anchors_present(['2025-01-06'],'January 6, 2024'))
        self.assertFalse(anchors_present(['2025-01-06'],'January 16'))
        self.assertTrue(anchors_present(['component_dependency_version','publish','hotfix'],
                                      'Component-dependency versions published on hot‑fix.'))
        self.assertFalse(anchors_present(['2.0'],'12.0'))
        self.assertFalse(anchors_present(['billing-api'],'billing-apis-extra'))

    def message(self,channel='dyad',core='billing-api version 2.0 published'):
        return dict(id='m1',speaker='Ari',core=core,channel=channel,thread='thread1',
                    participants=['Ari','Bela'] if channel=='dyad' else ['Ari','Bela','Cleo'],
                    length_contract='carrier_v7',text_chars=160,
                    anchor_sets={'f1':['billing-api','2.0','published']} if core else {})

    def test_structured_turns_not_prose_speaker_regex(self):
        m=self.message()
        row=dict(turns=[dict(speaker='Ari',text=m['core']+'. I can explain the wording if that would help.'),
                        dict(speaker='Bela',text='Thanks.')])
        self.assertTrue(valid_message(row,m),message_failures(row,m))
        self.assertIn('Ari: ',row['text']);self.assertIn('Bela: ',row['text'])
        row['turns'][1]['speaker']='Mallory'
        self.assertIn('need exactly 2 participant speakers',message_failures(row,m))
        self.assertFalse(valid_message(dict(text='Ari: '+m['core']+'\nBela: Thanks.'),m))
        m=self.message('software')
        row=dict(turns=[dict(speaker=n,text=m['core']+'. These are the entries we are reading together with their dates and original qualifications.' if i==0 else 'Thanks.') for i,n in enumerate(m['participants'])])
        assemble_turns(row,m)
        self.assertIn('@Bela',row['text']);self.assertIn('[thread thread1]',row['text'])
        self.assertTrue(valid_message(row,m),message_failures(row,m))
        m.update(core='',text=row['text'],turns=row['turns'])
        other=copy.deepcopy(m);other.update(id='m2',thread='thread2')
        assemble_turns(other,other)
        self.assertNotEqual(m['text'],other['text'])
        self.assertEqual(dedupe_fillers(dict(rounds=[m,other])),['m2'])

    def test_assistant_both_directions_and_reply_budget_keyerror(self):
        m=self.message('assistant');m['reply_chars']=300
        row=dict(turns=[dict(speaker='Ari',text=m['core']+'. These are the notes I wanted to read together with their dates and original qualifications.'),
                        dict(speaker='Assistant',text='Let us read the supplied wording with care. '*4)])
        self.assertTrue(valid_message(row,m),message_failures(row,m))
        row['turns'][1]['text']=('The term billing-api identifies a component, 2.0 labels a version, and published is a status word. '
                                  'Those words must be read with the dates and qualifications in the source.')
        self.assertTrue(valid_message(row,m),message_failures(row,m))
        row['turns'][1]['text']=(m['core']+'. ')*5
        self.assertFalse(valid_message(row,m))
        m.pop('reply_chars');m.update(speaker='Assistant',user_speaker='Bela',text_chars=160)
        row=dict(turns=[dict(speaker='Bela',text='Please explain.'),dict(speaker='Assistant',text=m['core']+'. These are the notes we are reading together with their dates and original qualifications.')])
        self.assertTrue(valid_message(row,m),message_failures(row,m))
        m.update(row,assistant_reply='Unexpected optional reply.')
        m.pop('turns')
        match_lengths(dict(rounds=[m])) # formerly KeyError: reply_chars
        self.assertEqual(m['assistant_reply'],'Unexpected optional reply.')

    def test_empty_filler_and_length_tolerance(self):
        m=self.message('record',core='');m['text_chars']=2000
        self.assertTrue(valid_message(dict(text='Ari paused to read.'),m))
        m['core']='facts';m['anchor_sets']={};m['text_chars']=100
        for n in (30,200):self.assertTrue(valid_message(dict(text=('x '*15).ljust(n,'x')),m))
        self.assertFalse(valid_message(dict(text='x'*40),m)) # factual sanity: 15 words
        self.assertFalse(valid_message(dict(text='x '*126),m)) # 2.5x ceiling

    def test_three_anchors_require_verified_binding_for_projection(self):
        fact=dict(statement='canonical',entity='billing-api',value='2.0',status='published',holder='Ari')
        self.assertEqual(anchors_for(fact),['billing-api','2.0','published'])
        fact.update(anchors=anchors_for(fact),evidence_spans=['Ari published billing-api 2.0 on January 6.'])
        self.assertFalse(atom_present(fact,'Bela published billing-api 2.0 on January 8.'))
        self.assertTrue(atom_present(fact,'Ari published billing-api 2.0 on January 6.'))

    def test_failed_only_retries_feedback_and_checkpoints(self):
        class Repair(v.MockPool):
            def __init__(self):super().__init__();self.payloads=[]
            def chat(self,prompt,**kw):
                payload=json.loads(prompt.split('\nPAYLOAD\n')[1]);self.payloads.append(payload)
                messages=[]
                for m in payload['messages']:
                    row=dict(id=m['id'],text=m['core'])
                    if len(self.payloads)==1 and m['id']=='bad':row['text']='missing name'
                    messages.append(row)
                return json.dumps(messages),'offline'
        pool=Repair();models=v.ModelFamilies(pools=[pool]*3);failures={}
        messages=[dict(id=n,core='billing-api',anchor_sets={'f1':['billing-api']}) for n in ('good','bad')]
        models.ask(0,'Render',dict(task='render',messages=messages),row_validator=valid_message,failure_collector=failures)
        self.assertEqual([[m['id'] for m in p['messages']] for p in pool.payloads],[['good','bad'],['bad']])
        self.assertIn("missing anchor 'billing-api'",pool.payloads[1]['retry_feedback']['bad']['reasons'])
        self.assertEqual(failures['bad'],["missing anchor 'billing-api'"])
        # End-to-end checkpoint/world logging includes failures even when all
        # retries fail, without relying on a real transport.
        class Broken(v.MockPool):
            def chat(self,prompt,**kw):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                return json.dumps([dict(id=m['id'],text='broken') for m in p['messages']]),'offline'
        plan=v.make_plan(2,8,seed=73,domain='software')
        plan['cases']=plan['cases'][:1]
        plan['facts']=[f for f in plan['facts'] if f['source']['case']==plan['cases'][0]['id']]
        with tempfile.TemporaryDirectory() as root:
            world=v.render(plan,v.ModelFamilies(pools=[Broken()]*3),checkpoint_dir=root)
            self.assertTrue(world['render_failure_counts'])
            self.assertTrue(all(m['failure_reasons'] for m in world['rounds']))
            for path in Path(root).glob('*.json'):
                self.assertTrue(v.read(path)['message_failures'])

    def test_same_day_approval_and_veto_are_separate_proposals(self):
        from world3_probes import ideal_prediction
        query=dict(operation='direct',entity='trip',as_of=v.day(21),known_at=v.day(25),
            context={'weather':'rainy'},requests=[dict(holder='Ari',scope='personal',entity='trip',attr='booking permission / materials')])
        case=dict(id='collision',offset=40,record_time=v.day(25),qtype='direct',query=query,
                  perspective='external',explicit_as_of=True)
        world=dict(world=7,seed=7,domain='travel',facts=[],cases=[case])
        old=v.add_version(world,case,'Ari','personal','trip','booking permission / materials','confirmed',
                          condition={'weather':'rainy'},when=9)[-1]
        approved=v.add_version(world,case,'Ari','personal','trip','booking permission / materials','waitlisted',
                          condition={'weather':'rainy'},when=14,statuses=('proposed','approved'),supersedes=old)[-1]
        vetoed=v.add_version(world,case,'Ari','personal','trip','booking permission / materials','waitlisted',
                          condition={'weather':'rainy'},when=14,statuses=('proposed','vetoed'),supersedes=old)[-1]
        world=v.naturalize(world);case=world['cases'][0];byid={f['id']:f for f in world['facts']}
        self.assertIn(byid[approved]['proposal'],byid[approved]['statement'])
        self.assertNotEqual(byid[approved]['proposal'],byid[vetoed]['proposal'])
        q=dict(question=case['question'],haystack_sessions=[[dict(role='user',content='\n'.join(f['statement'] for f in world['facts']))]],haystack_dates=['January-26-2025'])
        pred,answer=ideal_prediction(q,[byid[approved]['statement'],byid[vetoed]['statement']],set())
        self.assertEqual(pred,[0]);self.assertEqual(answer,v.solve(world,case['query'])['answer'])
        self.assertIn('waitlisted',answer)

    def test_domain_names_and_qualifier_policy(self):
        plan=v.make_plan(8,20,seed=73,domain='software')
        ordinary=[f for f in plan['facts'] if not f['natural_surface']['update']]
        self.assertGreater(len(ordinary),len(plan['facts'])/2)
        self.assertTrue(all('This was the ' not in f['statement'] for f in ordinary))
        self.assertTrue(all(len(f['entity'].split())>=2 for f in plan['facts']))
        self.assertTrue(any('-api' in f['entity'] for f in plan['facts']))
        for f in plan['facts']:self.assertLessEqual(len(anchors_for(f)),4)
        self.assertTrue(v.validate_ledger(plan['facts']))


if __name__=='__main__':unittest.main()
