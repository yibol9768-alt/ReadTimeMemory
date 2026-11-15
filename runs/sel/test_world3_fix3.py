"""CPU-only replay and rendering contracts for the third real-model fix."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import world3 as v
from world3_style import (STATUS_SYNONYMS, anchor_present, anchors_present,
                          augment_group, filler_keys, message_failures, valid_message)


class Fix3Tests(unittest.TestCase):
    def test_sanity_limits_and_social_padding_do_not_hide_duplicate_filler(self):
        m=dict(id='m1',speaker='Ari',core='source',channel='record',anchor_sets={},
               text_chars=100,length_contract='carrier_v8')
        text='one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen'
        self.assertTrue(valid_message(dict(text=text),m))
        self.assertTrue(valid_message(dict(text=text.ljust(249)),m))
        self.assertFalse(valid_message(dict(text=text.ljust(250)),m))
        self.assertFalse(valid_message(dict(text='a short factual claim'),m))
        m.update(core='',text_chars=2200)
        self.assertTrue(valid_message(dict(text='Thanks.'),m))
        m.update(channel='group',participants=list(v.NAMES))
        row=dict(turns=[dict(speaker='Ari',text='Can we read this together?'),dict(speaker='Bela',text='Sure.')])
        before=filler_keys(row,m)
        augment_group(row,m)
        self.assertEqual(filler_keys(row,m),before)

    def test_status_synonyms_cover_all_machines_without_state_collisions(self):
        machines=[{s for edge in spec['machine'] for s in edge} for spec in v.STRUCTURES.values()]
        machines.append({s for states in v.TRANSITIONS.values() for s in states})
        for states in machines:
            self.assertLessEqual(states,STATUS_SYNONYMS.keys())
            for status in states:
                for phrase in STATUS_SYNONYMS[status]:
                    self.assertTrue(anchor_present(status,phrase))
                    for other in states-{status}:
                        self.assertFalse(anchor_present(other,phrase),(status,phrase,other))
        self.assertTrue(anchors_present(['executed','integration'], 'We ran the integration tests.'))
        self.assertFalse(anchor_present('executed','The runner is ready.'))
        self.assertTrue(anchor_present('rolled_back','We rolled it back.'))

    def test_replay_intact_partial_checkpoint_rows(self):
        root=Path(__file__).parent/'e2e6_ckpt_1108'
        samples=[('task2_block3.json','m92','dyad','Davi',
                  ['research payments frontend','package index','executed','Cleo']),
                 ('task3_block28.json','m268','group','Bela',
                  ['central storage indexer','3.0','published','Bela']),
                 ('task7_block6.json','m632','group','Farah',
                  ['advanced sessions proxy','archive','executed','Cleo'])]
        for filename,mid,carrier,speaker,anchors in samples:
            saved=v.read(root/filename);self.assertEqual(saved['status'],'partial')
            row=copy.deepcopy(next(r for r in saved['result']['messages'] if r['id']==mid))
            m=dict(id=mid,core=row['turns'][0]['text'],speaker=speaker,channel=carrier,
                   participants=['Davi','Cleo'] if carrier=='dyad' else list(v.NAMES),
                   length_contract='carrier_v8',text_chars=1100,anchor_sets={'f':anchors})
            self.assertTrue(valid_message(row,m),message_failures(row,m))
        # A real short record filler and assistant exchange also survive.
        for filename,mid,carrier in [('task4_block0.json','m272','record'),('task0_block1.json','m2','assistant')]:
            saved=v.read(root/filename);self.assertEqual(saved['status'],'partial')
            row=copy.deepcopy(next(r for r in saved['result']['messages'] if r['id']==mid))
            m=dict(id=mid,core='',speaker='Bela',channel=carrier,participants=list(v.NAMES),
                   length_contract='carrier_v8',text_chars=2200,reply_chars=2200,anchor_sets={})
            if carrier=='record':m.pop('reply_chars')
            self.assertTrue(valid_message(row,m),message_failures(row,m))

    def test_assistant_retries_only_missing_side_and_hides_ledger_from_reply(self):
        class OneSide(v.MockPool):
            def __init__(self):super().__init__();self.payloads=[]
            def chat(self,prompt,**kw):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);self.payloads.append(p)
                source=p['messages'][0]
                if source['render_side']=='assistant':
                    if sum(x['messages'][0]['render_side']=='assistant' for x in self.payloads)==1:
                        return '{"messages":[]}','offline'
                    return json.dumps(dict(id=source['id'],turns=[dict(speaker='Assistant',text='Thanks for sharing. We can read this carefully together.')])), 'offline'
                return json.dumps(dict(id=source['id'],text=source['draft'])), 'offline'
        pool=OneSide();models=v.ModelFamilies(pools=[pool]*3)
        text='Ari ran the integration tests for the billing service today and recorded the result in the team notes.'
        source=dict(id='m1',speaker='Ari',participants=['Ari','Bela'],channel='assistant',
                    core=text,draft=text,anchor_sets={'f':['billing service','integration','executed']},
                    length_contract='carrier_v8',text_chars=300,reply_chars=2200)
        failures={}
        result=models.ask(0,'Render',dict(task='render',carrier='assistant',messages=[source]),
                          row_validator=valid_message,failure_collector=failures)
        self.assertEqual([p['messages'][0]['render_side'] for p in pool.payloads],['user','assistant','assistant'])
        self.assertEqual([t['speaker'] for t in result['messages'][0]['turns']],['Ari','Assistant'])
        self.assertEqual(failures,{'m1':['missing JSON row']})
        for payload in pool.payloads[1:]:
            self.assertEqual(payload['user_turn'],text)
            self.assertEqual(payload['messages'][0]['core'],'')
            self.assertEqual(payload['messages'][0]['anchor_sets'],{})

    def test_assistant_caller_validation_retries_only_rejected_side(self):
        class Repair(v.MockPool):
            def __init__(self):super().__init__();self.sides=[]
            def chat(self,prompt,**kw):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);m=p['messages'][0]
                self.sides.append(m['render_side'])
                if m['render_side']=='assistant':
                    text='Repeated filler.' if self.sides.count('assistant')==1 else 'We can read this together.'
                    return json.dumps(dict(id=m['id'],text=text)),'offline'
                return super().chat(prompt,**kw)
        m=dict(id='m1',speaker='Ari',participants=['Ari','Bela'],channel='assistant',
               core='',draft='Could you help me read the notes?',anchor_sets={},
               length_contract='carrier_v8',text_chars=100,reply_chars=2000)
        def valid(row,source):
            if row.get('assistant_reply')=='Repeated filler.':
                row['_validation_failures']=['duplicate no-new-fact filler'];return False
            return valid_message(row,source)
        pool=Repair();models=v.ModelFamilies(pools=[pool]*3);failures={}
        result=models.ask(0,'Render',dict(task='render',carrier='assistant',messages=[m]),
                          row_validator=valid,failure_collector=failures)
        self.assertEqual(pool.sides,['user','assistant','assistant'])
        self.assertEqual(failures,{'m1':['duplicate no-new-fact filler']})
        self.assertEqual(result['messages'][0]['assistant_reply'],'We can read this together.')

    def test_reply_anchor_leak_retries_and_group_attribution(self):
        class Leaks(v.MockPool):
            def chat(self,prompt,**kw):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);source=p['messages'][0]
                if source.get('render_side')=='assistant' and not source['core']:
                    return json.dumps(dict(id=source['id'],text='The billing service is easy to discuss.')),'offline'
                return super().chat(prompt,**kw)
        source=dict(id='m1',speaker='Ari',participants=['Ari','Bela','Cleo'],channel='assistant',
                    core='Ari ran integration tests for the billing service and recorded the results carefully in the shared notes.',
                    draft='Ari ran integration tests for the billing service and recorded the results carefully in the shared notes.',
                    anchor_sets={'f':['billing service','integration','executed']},length_contract='carrier_v8',text_chars=300,reply_chars=2200)
        failures={};models=v.ModelFamilies(pools=[Leaks()]*3)
        with self.assertRaises(ValueError):
            models.ask(0,'Render',dict(task='render',carrier='assistant',messages=[source]),failure_collector=failures)
        self.assertEqual(failures['m1'],['assistant reply contains fact anchor']*3)
        source['channel']='group';source.pop('reply_chars')
        row=dict(turns=[dict(speaker='Ari',text='Thanks.'),dict(speaker='Bela',text=source['core'])])
        self.assertIn('annotated speaker must assert the fact',message_failures(row,source))
        row['turns'].reverse();source['speaker']='Bela'
        self.assertTrue(valid_message(row,source),message_failures(row,source))
        augment_group(row,source)
        self.assertTrue(valid_message(row,source),message_failures(row,source))

    def test_single_message_quarantine_summary_and_checkpoint_resume(self):
        class Missing(v.MockPool):
            def __init__(self):super().__init__();self.ids=[]
            def chat(self,prompt,**kw):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);self.ids.append([m['id'] for m in p['messages']])
                self.assert_single(p)
                if p['messages'][0]['id']=='m0':return '{"messages":[]}', 'offline'
                return super().chat(prompt,**kw)
            def assert_single(self,p):
                if len(p['messages'])!=1:raise AssertionError('batched rendering')
        plan=v.make_plan(2,8,seed=73,domain='software')
        case=next(c for c in plan['cases'] if c['carrier']=='record')
        plan['cases']=[case];plan['facts']=[f for f in plan['facts'] if f['source']['case']==case['id']]
        pool=Missing();models=v.ModelFamilies(pools=[pool]*3)
        with tempfile.TemporaryDirectory() as root:
            world=v.render(plan,models,par=8,checkpoint_dir=root)
            self.assertEqual([m['id'] for m in world['rounds'] if m.get('render_error')],['m0'])
            self.assertEqual(world['render_summary']['failures'],{'missing JSON row':3})
            stats=world['render_summary']['by_carrier']['record']
            self.assertEqual(stats['quarantined'],1)
            self.assertEqual(stats['done'],len(world['rounds'])-1)
            self.assertEqual(len(list(Path(root).glob('render_m*.json'))),len(world['rounds']))
            calls=len(pool.ids);cached=v.render(plan,models,par=4,checkpoint_dir=root)
            self.assertEqual(calls,len(pool.ids));self.assertEqual(world,cached)


if __name__=='__main__':unittest.main()
