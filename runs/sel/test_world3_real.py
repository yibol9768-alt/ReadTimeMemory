"""Offline regressions for captured real render outputs and paraphrased facts."""
import collections
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import world3 as v
from world3_style import anchors_for, anchors_present, valid_message, leaks_anchors
from world3_adapter import groups_for, mock_pool, length_distribution, match_length_groups


class ResponseTests(unittest.TestCase):
    def test_replay_real_raw_outputs(self):
        captured=json.loads(Path(__file__).with_name('dbg_render_raw_1108.json').read_text())
        parsed=rows=0;failed=[];partial={}
        for i,item in enumerate(captured):
            with self.subTest(index=i,family=item['family']):
                try:result=v.parse_model_response(item['raw'],expected_ids={m['id'] for m in item['payload']['messages']} if item.get('payload') else None)
                except ValueError:
                    failed.append(i);continue
                self.assertTrue(v.ModelFamilies.response_schema(result,{'task':'render'}))
                self.assertEqual(len({r['id'] for r in result['messages']}),len(result['messages']))
                expected=item.get('payload',{}).get('messages')
                if expected:
                    self.assertLessEqual({r['id'] for r in result['messages']},{m['id'] for m in expected})
                n=len(expected) if expected else item['n']
                if len(result['messages'])<n:partial[i]=len(result['messages'])
                parsed+=1;rows+=len(result['messages'])
        self.assertEqual((parsed,rows),(93,545))
        self.assertEqual(failed,[65,77,86]) # truncated inside the first row
        self.assertEqual(partial,{17:1,73:5,94:4,95:3})

    def test_fences_prose_duplicate_rows_and_partial_retry(self):
        quoted='An example: ```text\n<think>literal markup</think>\n```'
        self.assertEqual(v.parse_model_response(json.dumps(dict(messages=[dict(id='a',text=quoted)])))['messages'][0]['text'],quoted)
        source=dict(id='m2',core='Maple',anchor_sets={'f9':['Maple']})
        for quotes in ({'m2:f9':'Maple'},{'f9':'Maple'},{'m2':'Maple'},{'m2':{'f9':'Maple'}}):
            vote=dict(ok=True,no_extra_facts=True,quotes=quotes)
            self.assertTrue(v.ModelFamilies.response_schema(vote,dict(task='verify',messages=[source])))
            self.assertEqual(v.verification_quote(vote,source,'f9'),'Maple')
        for raw in ['```json\n[{"id":"a","text":"hello"}]\n```\nHere you go.',
                    'Here is the result: {"messages":[{"id":"a","text":"hello"}]}\nDone.']:
            self.assertEqual(v.parse_model_response(raw)['messages'][0]['text'],'hello')
        class Partial(v.MockPool):
            def __init__(self):super().__init__();self.ids=[]
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);self.ids.append([m['id'] for m in p['messages']])
                if len(self.ids)==1:return '[{"id":"a","text":"first answer"},{"id":"b","text":"bad"}]','offline'
                return '{"messages":[{"id":"b","text":"fixed"}]} trailing explanation','offline'
        pool=Partial();models=v.ModelFamilies(pools=[pool]*3)
        result=models.ask(0,'Render',dict(task='render',messages=[dict(id='a'),dict(id='b')]),
                          row_validator=lambda row,m:row['text']!='bad')
        self.assertEqual(pool.ids,[['a','b'],['b']])
        self.assertEqual(result['messages'],[dict(id='a',text='first answer'),dict(id='b',text='fixed')])
        class Duplicate(Partial):
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);self.ids.append([m['id'] for m in p['messages']])
                if len(self.ids)==1:return '[{"id":"a","text":"one"},{"id":"a","text":"two"},{"id":"b","text":"good"}]','offline'
                return '[{"id":"a","text":"fixed"}]','offline'
        pool=Duplicate();models=v.ModelFamilies(pools=[pool]*3)
        models.ask(0,'Render',dict(task='render',messages=[dict(id='a'),dict(id='b')]))
        self.assertEqual(pool.ids,[['a','b'],['a']])

    def test_long_carrier_blocks_and_length_band(self):
        self.assertTrue(anchors_present(['Maple Grove','rolled_back','2025-01-10'],
            'MAPLE\nGROVE — rolled back (2025/01/10)'))
        self.assertLessEqual(max(collections.Counter(v.block_ids(37,'record')).values()),2)
        self.assertLessEqual(max(collections.Counter(v.block_ids(37,'assistant')).values()),2)
        m=dict(id='a',core='',length_contract='carrier_v6',channel='record',speaker='Ari',text_chars=1500,anchor_sets={})
        self.assertTrue(valid_message(dict(text='A simple sentence. '*65),m))
        self.assertTrue(valid_message(dict(text='A simple sentence. '*40),m))
        self.assertTrue(valid_message(dict(text='A short note.'),m))
        self.assertFalse(valid_message(dict(text='A simple sentence. '*210),m))

    def test_statistical_length_match_drops_whole_groups(self):
        matched=[dict(qid=f'q{i}',carrier='group',label=0,docs=['x'*400]*16) for i in range(20)]
        bad=dict(qid='leak',carrier='group',label=0,docs=['x'*1900]+['x'*400]*15)
        # Many bad groups shift both the mean and distribution.
        kept,drops,report=match_length_groups(matched+[dict(bad,qid=f'bad{i}') for i in range(12)])
        self.assertTrue(report['pass']);self.assertTrue(drops)
        self.assertTrue(length_distribution(kept)['pass'])
        self.assertTrue(all(len(g['docs'])==16 for g in kept))
        self.assertTrue(all(g in matched or g['qid'].startswith('bad') for g in kept))


class NaturalRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=v.make_plan(102,20,seed=73,domain='software')
        ids={next(c['id'] for c in cls.plan['cases'] if c['carrier']==carrier) for carrier in v.CARRIERS}
        cls.plan['cases']=[c for c in cls.plan['cases'] if c['id'] in ids]
        cls.plan['facts']=[f for f in cls.plan['facts'] if f['source']['case'] in ids]
        class Paraphrase(v.MockPool):
            def chat(self,prompt,**kwargs):
                result,family=super().chat(prompt,**kwargs)
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='render':
                    for message in p['messages']:
                        assert not ({'kind','state_support','targets','label','query_spec','query','qtype'} & set(message))
                    rows=json.loads(result)
                    for row in rows['messages']:
                        row['text']=row['text'].replace('On 20','As of 20').replace('; its status was ','; the recorded status is ')
                        for turn in row.get('turns',[]):
                            turn['text']=turn['text'].replace('On 20','As of 20').replace('; its status was ','; the recorded status is ')
                    result=json.dumps(rows)
                return result,family
        cls.models=v.ModelFamilies(pools=[Paraphrase() for _ in range(3)],mock=False)
        cls.world=v.verify(v.render(cls.plan,cls.models,par=4),cls.models)

    def test_paraphrase_build_and_adapter(self):
        self.assertTrue(all(c['semantic_ok'] for c in self.world['cases']))
        report={};questions=v.build_questions(self.world,build_report=report)
        self.assertEqual(len(questions),4,report)
        for m in self.world['rounds']:
            self.assertTrue(all(anchors_present(a,m['text']) for a in m['anchor_sets'].values()))
            if m['core']:self.assertNotIn(m['core'],m['text'])
        for q in questions:
            self.assertTrue(q['visibility']['ok'])
            gs,rs=groups_for(q,mock_pool(q),no_set=True,p=100000)
            self.assertEqual(len(gs),1,rs)
            self.assertTrue(gs[0]['ideal_rule']['correct'],gs[0]['ideal_rule'])
            self.assertTrue(gs[0]['competition']['pass'])
        gate=v.gates(self.world,questions[:1],v.ModelFamilies(mock=True))[0]
        self.assertTrue(gate['order'],gate)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.json';v.sample_dump([self.world],path)
            dump=v.read(path)
            self.assertEqual(dump['counts'],dict.fromkeys(v.CARRIERS,10))
            self.assertFalse(dump['mock'])
            self.assertTrue(all('atoms' in m and 'label' in m for rows in dump['messages'].values() for m in rows))

    def test_names_carrier_forms_no_filler_duplicates_and_leaks(self):
        facts=self.world['facts'];names={f['entity'] for f in facts}
        self.assertGreater(len(names),16)
        for f in facts:
            self.assertNotRegex(f['statement'],r'\b[A-Z]{8}\b|\w+=|Links: none|Decision ')
        fillers=[v.normalize_filler(m['text']) for m in self.world['rounds'] if not m['atoms']]
        self.assertEqual(len(fillers),len(set(fillers)))
        for c in self.world['cases']:
            ms=[m for m in self.world['rounds'] if m['case']==c['id']]
            if c['carrier']=='group':self.assertTrue(all('@' in m['text'] and len({t['speaker'] for t in m['turns']})>=2 for m in ms))
            elif c['carrier']=='dyad':self.assertTrue(all(m['text'].count(':')>=2 for m in ms))
            elif c['carrier']=='record':self.assertTrue(all('notebook' in m['text'] for m in ms))
            else:self.assertTrue(any(m.get('assistant_reply') for m in ms))
        m=next(m for m in self.world['rounds'] if m['atoms'])
        f=next(f for f in facts if f['source']['case']==m['case'] and f['entity'] not in m['core'] and f['id'] not in m['atoms'])
        self.assertTrue(leaks_anchors([anchors_for(f)],m['text']+'\n'+f['statement']))
        row=dict(text=m['text']+'\n'+f['statement'])
        if m.get('assistant_reply'):row['assistant_reply']=m['assistant_reply']
        self.assertFalse(valid_message(row,m,[anchors_for(f)]))

    def test_duplicate_empty_assistant_rounds_retry_both_sides_only(self):
        class Duplicates(v.MockPool):
            def __init__(self):super().__init__();self.repairs=[]
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                result,family=super().chat(prompt,**kwargs)
                if p['task']=='render':
                    if p.get('avoid_filler'):
                        self.repairs.extend(p['messages'])
                    else:
                        rows=json.loads(result)
                        sources={m['id']:m for m in p['messages']}
                        for row in rows['messages']:
                            source=sources[row['id']]
                            if not source['core'] and source.get('render_side'):
                                row['text']=('Could we take a moment to read this together?' if source['render_side']=='user'
                                             else 'A little room for questions can make a discussion easier to follow.')
                        result=json.dumps(rows)
                return result,family
        pool=Duplicates();models=v.ModelFamilies(mock=True,pools=[pool]*3)
        plan=copy.deepcopy(self.plan)
        plan['cases']=[next(c for c in plan['cases'] if c['carrier']=='assistant')]
        plan['facts']=[f for f in plan['facts'] if f['source']['case']==plan['cases'][0]['id']]
        world=v.render(plan,models)
        self.assertTrue(pool.repairs)
        self.assertTrue(all(not m['core'] for m in pool.repairs))
        empty=[m for m in world['rounds'] if not m['core']]
        self.assertFalse(any(m.get('render_error') for m in empty))
        for side in ('text','assistant_reply'):
            values=[v.normalize_filler(m[side]) for m in empty if m.get(side)]
            self.assertEqual(len(values),len(set(values)),side)

    def test_sample_cli_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v.write(root/'w00102.json',self.plan)
            with patch.object(v,'render',return_value=copy.deepcopy(self.world)):
                v.main(['render','--mock','--out',d,'--start','102','--sample-dump',str(root/'samples.json')])
            self.assertEqual(v.read(root/'samples.json')['counts'],dict.fromkeys(v.CARRIERS,10))


if __name__=='__main__':unittest.main()
