"""Standard-library unittest coverage; all inference is mock and local."""
import collections
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import seed_extract as sx
import seed3 as s
from seed3_pilot_review import review
from world3 import ModelFamilies, visible_documents
from world3_style import anchors_present, normalize

ROOT=Path(__file__).resolve().parent
STATS={k:{'turn_chars':[280,350,440]} for k in ('group','dyad','assistant','record')}


def related_hosts(skeleton,facts):
    # Negative/relay protocol tests require related hosts under r2k. Previously
    # they depended on wholesale replacement of unrelated source messages.
    skeleton=copy.deepcopy(skeleton)
    for t in skeleton['turns']:
        t['text']+=' We are still discussing '+facts[0]['topic']+'.'
    return skeleton


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={src:sx.extract(src,sx.read_rows([str(ROOT/'seed_raw_samples'/(src+'.jsonl'))]),
                    version='fixture-sha',stats=STATS,limit=100) for src in sx.SOURCES}

    def test_all_six_actual_sources(self):
        for src,rows in self.rows.items():
            with self.subTest(source=src):
                self.assertGreaterEqual(len(rows),2)
                for r in rows:
                    self.assertEqual(r['source_version'],'fixture-sha')
                    self.assertTrue(r['license']); self.assertFalse(r['public_release_allowed'])
                    self.assertEqual(r['rename_audit']['leftover_count'],0)
                    self.assertEqual(r['length_sampling'],'natural_stats.turn_chars')
                    self.assertEqual(r['carrier'],sx.SOURCES[src][0])
                    for t in r['turns']:
                        self.assertLessEqual(len(t['text']),t['length_cap'])
                        self.assertFalse(sx.EMAIL.search(t['text']))
                        self.assertFalse(sx.URL.search(t['text']))

    def test_rename_aliases_mentions_quotes_and_replies(self):
        turns=[dict(tid='0',speaker='alice.smith@example.org',text='Alice told @bob: "Alice Smith agrees".',reply_to=['8']),
               dict(tid='1',speaker='bob',text='@Alice Smith, I agree with alice.smith@example.org.',reply_to=['0'])]
        aliases={'alice.smith@example.org':['Alice Smith','alice.smith@example.org','Alice'], 'bob':['bob']}
        renamed,audit=sx.rename_all(turns,aliases,42)
        text=' '.join(t['text'] for t in renamed)
        self.assertNotRegex(text.lower(),r'alice|bob|example.org')
        self.assertEqual(renamed[1]['reply_to'],['0'])
        self.assertIn('@'+renamed[0]['speaker'],renamed[1]['text'])
        self.assertIn('@'+renamed[1]['speaker'],renamed[0]['text'])
        self.assertEqual(audit['leftover_count'],0)

    def test_pii_and_language(self):
        text=sx.clean_text('Hi, email foo@example.org, https://example.org/a or call +1 212-555-1234.')
        self.assertIn('[EMAIL]',text); self.assertIn('[URL]',text); self.assertIn('[PHONE]',text)
        self.assertFalse(sx.is_english('这是一个非英文对话。'))
        self.assertFalse(sx.is_english('Hello there','fr'))
        self.assertTrue(sx.is_english('Thanks!'))

    def test_email_threads_signatures_and_split(self):
        rows=[dict(subject=sub,**{'from':sender,'to':['b@example.org'],'cc':[], 'date':i,
              'body':f'I have the report number {i}.\nBest,\nAlice\n> old quote\nFrom: someone'})
              for i,(sub,sender) in enumerate([('Re: Fwd: Weekly work','a@example.org'),('Weekly work','b@example.org'),('RE: weekly work','a@example.org')])]
        out=sx.extract('enron',rows,stats=STATS)
        self.assertEqual(len(out),1); self.assertEqual(len(out[0]['turns']),3)
        self.assertEqual(len({t['speaker'] for t in out[0]['turns']}),2)
        self.assertNotIn('old quote',' '.join(t['text'] for t in out[0]['turns']))
        again=sx.extract('enron',rows*2,version='other',stats=STATS)
        self.assertEqual(again[0]['source_conv_id'],out[0]['source_conv_id'])
        self.assertEqual(again[0]['split'],out[0]['split'])

    def test_split_leakage_and_author_windows(self):
        allrows=[r for rows in self.rows.values() for r in rows]
        s.validate_skeletons(allrows)
        groups=collections.defaultdict(set)
        for r in allrows: groups[(r['source'],r['source_conv_id'])].add(r['split'])
        self.assertTrue(all(len(v)==1 for v in groups.values()))
        bad=copy.deepcopy(allrows[0]); bad['skel_id']='different'; bad['split']='audit' if bad['split']!='audit' else 'train'
        with self.assertRaisesRegex(ValueError,'leakage'): s.validate_skeletons([allrows[0],bad])

    def test_irc_contiguous_windows_and_reply_ids(self):
        rows=[dict(id=i,raw=f'[10:00] <{"alice" if i%2 else "bob"}> I see @alice, thanks.',date='2004-12-25',connections=[i-1] if i else []) for i in range(12)]
        out=sx.extract('irc_ubuntu',rows,window_turns=6,stats=STATS)
        self.assertEqual(len(out),2)
        self.assertEqual(out[0]['source_conv_id'],out[1]['source_conv_id'])
        self.assertEqual(out[0]['split'],out[1]['split'])
        self.assertEqual(out[1]['turns'][0]['reply_to'],['5'])
        self.assertEqual(out[0]['turns'][0]['speaker'],out[1]['turns'][0]['speaker'])
        self.assertEqual([t['tid'] for t in out[0]['turns']],list(map(str,range(6))))

    def test_oasst_paths_are_root_to_leaf_not_flattened_branches(self):
        rows=[dict(message_id=str(i),parent_id=p,message_tree_id='t',role='prompter' if i%2==0 else 'assistant',text='I have a question, thanks.',lang='en') for i,p in enumerate([None,'0','1','2','1','4'])]
        out=sx.extract('oasst2_val_en',rows,min_turns=3,stats=STATS)
        self.assertEqual({tuple(t['tid'] for t in r['turns']) for r in out},{('0','1','2','3'),('0','1','4','5')})
        self.assertEqual(len({r['source_conv_id'] for r in out}),1)

    def test_real_stats_uses_benchmark_not_synthetic_carrier(self):
        vals,method=sx.length_samples({'by_carrier':{'group':{'chars_p10':9000,'chars_p50':9000,'chars_p90':9000}},
                                      'by_benchmark':{'ev':{'chars_p10':100,'chars_p50':200,'chars_p90':300}}},'group',[])
        self.assertEqual(vals,[100,200,300]); self.assertEqual(method,'natural_stats.quantiles')

    def test_zip_blog_ingestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'blogs.zip'
            with zipfile.ZipFile(path,'w') as z: z.writestr('blogs/123.female.xml','<Blog><date>1,May,2004</date><post>I wrote this today.</post></Blog>')
            rows=list(sx.read_rows([str(path)])); self.assertEqual(rows[0]['file'],'blogs/123.female.xml')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skeletons=[]
        for src in sx.SOURCES:
            skeletons+=sx.extract(src,sx.read_rows([str(ROOT/'seed_raw_samples'/(src+'.jsonl'))]),stats=STATS,limit=100)
        cls.skeletons=s.choose_pilot(skeletons)
        cls.models=s.seed_models(mock=True)
        cls.plans=s.plan(cls.skeletons,cls.models)
        cls.rendered=s.parallel_map(lambda c:s.render_case(c,cls.models),cls.plans,4)
        cls.verified=s.parallel_map(lambda c:s.verify_case(c,cls.models),cls.rendered,4)
        cls.questions,cls.pools,cls.groups=s.build(cls.verified)

    def test_pilot_8_skeletons_24_questions(self):
        self.assertIsInstance(self.models,ModelFamilies)
        self.assertEqual(collections.Counter(x['carrier'] for x in self.skeletons),{'group':2,'dyad':2,'assistant':2,'record':2})
        self.assertEqual(len(self.plans),24)
        self.assertEqual(len(self.questions),24,[(c['question_id'],c.get('drop_reason')) for c in self.verified if not c['accepted']])
        self.assertEqual(collections.Counter(x['perspective'] for x in self.questions),{p:8 for p in s.PERSPECTIVES})
        self.assertEqual(set(x['question_type'] for x in self.questions),set(s.QTYPES))
        self.assertEqual(len(self.pools),24)

    def test_program_relations_and_real_bindings(self):
        for c in self.plans:
            self.assertEqual(s.solve_intent(c['slots'],c['qtype'])['answer'],c['answer'])
            self.assertTrue(3<=len(c['slots'])<=6)
            self.assertEqual(len({f['id'] for f in c['facts']}),len(c['facts']))
            self.assertTrue(c['question_intent_checks'])
            self.assertFalse(s.question_failures(c['question'],c))
            for t in c['turns']:
                for fid in t['fact_ids']:
                    f=next(f for f in c['facts'] if f['id']==fid)
                    self.assertEqual(f['source']['speaker'],t['speaker'])
                    self.assertEqual(f['source']['reporter'],t['speaker'])
                    self.assertEqual(f['source']['holder'],f['holder'])
                    self.assertEqual(f['source']['tid'],t['tid'])
                    self.assertIn(f['holder'],{t['speaker'] for t in c['turns']})

    def test_matched_schedule_and_all_turn_provenance(self):
        for c in self.rendered:
            rate=sum(t['edit_kind']!='none' for t in c['turns'])/len(c['turns'])
            self.assertTrue(.4<=rate<=.6)
            self.assertEqual(c['rewrite_schedule']['target_edit_rate'],rate)
            self.assertTrue(c['rewrite_schedule']['r1_window_40_60_satisfied'])
            self.assertTrue(c['rewrite_schedule']['actual_matches_schedule'])
            for row in c['rewrite_schedule']['strata'].values():
                self.assertTrue(0<=row['target_edit_rate']<=1)
                # r2k permits new turns and measures them in the same strata.
                self.assertTrue(0<=row['target_insertion_rate']<=1)
            for t in c['turns']:
                self.assertIn(t['family'],range(3))
                self.assertEqual(t['original_text']==t['final_text'],t['edit_kind']=='none')
                self.assertEqual(t['text'],t['final_text'])
                self.assertGreaterEqual(t['edit_distance'],0)
                self.assertLessEqual(t['edit_distance'],1)
                if t['edit_kind']=='insert':
                    self.assertEqual(t['original_text'],'')
                    self.assertLessEqual(len(t['text']),400)
                elif t['fact_ids']:
                    # r2h permits the assigned statement's length plus 40 chars.
                    extra=max(len(f.get('natural_statement') or f['value']) for f in c['facts'] if f['id'] in t['fact_ids'])+40
                    self.assertTrue(.8*len(t['original_text'])<=len(t['text'])<=1.25*len(t['original_text'])+extra)
                else:self.assertTrue(.8<=len(t['text'])/len(t['original_text'])<=1.25)
                if t['edit_requested']:
                    self.assertTrue(t['register_checks']['passed'])

    def test_all_visible_labels_have_quotes_and_provenance(self):
        for c,q in zip(self.verified,self.questions):
            rows=c['judgment']['result']['candidates']
            self.assertEqual(len(rows),len(c['visible_documents']))
            self.assertEqual(set(c['gold_rounds']),{r['index'] for r in rows if r['support']!='none'})
            for r in rows:
                for cid in r['components']: self.assertIn(r['quotes'][cid],c['visible_documents'][r['index']])
            self.assertEqual(len(q['candidate_provenance']),len(q['visible_documents']))
            self.assertTrue(c['positive_control']['pass_control'])
            self.assertTrue(c['negative_control']['pass_control'])
            self.assertTrue(c['untouched_control']['judgment']['result']['abstain'])
            self.assertEqual(len(c['host_restoration_controls']),sum(t['edit_kind']!='none' for t in c['turns']))

    def test_adapter_serialization_and_groups(self):
        for q in self.questions:
            self.assertEqual(visible_documents(q),q['visible_documents'])
            for a in q['answer_atoms']:
                self.assertTrue(a['equivalent_rounds'])
                self.assertTrue(all(any(span in q['visible_documents'][i] for span in a['evidence_spans']) for i in a['equivalent_rounds']))
        for g in self.groups:
            self.assertEqual(g['kinds'][g['label']],'pos')
            self.assertNotEqual(g['provenance'][g['label']]['final_support']['support'],'none')
            self.assertTrue(all(p['final_support']['support']=='none' for i,p in enumerate(g['provenance']) if i!=g['label']))
            self.assertEqual(len(g['docs']),len(g['provenance']))

    def test_provenance_matching_measures_actual_gaps(self):
        report=s.provenance_table(self.questions)
        pos=report['table']['pos']
        self.assertEqual(pos['edit_amount_cdf_gap_to_positive'],0)
        self.assertEqual(report['matching_pass'],all(r['within_5pp'] for r in report['table'].values()))
        # Positive control for an edit shortcut: deliberately mismatched rows.
        p=copy.deepcopy(self.questions[0]); ps=p['candidate_provenance']
        for i,r in enumerate(ps):
            r['was_edited']=i==0; r['was_inserted']=i==0
            r['edit_amount']=float(i==0); r['position']=float(i==0)
            r['final_support']['support']='full' if i==0 else 'none'; r['planned_kind']='random'
        measured=s.provenance_table([p])
        self.assertEqual(measured['table']['random']['edited_rate_gap'],1)
        self.assertFalse(measured['matching_pass'])

    def test_was_edited_probe_actual_candidate_random_baseline(self):
        groups=[]
        for fold,n in enumerate((3,7,11,5)):
            groups.append(dict(qid=str(fold),ledger_id=str(fold),docs=['x']*n,label=0,
                               provenance=[dict(was_edited=i==0,was_inserted=False) for i in range(n)]))
        r=s.edited_probe(groups)
        self.assertAlmostEqual(r['random_baseline'],sum(1/n for n in (3,7,11,5))/4)
        self.assertEqual(r['accuracy'],1)
        self.assertFalse(r['pass_probe'])
        for g in groups:
            for p in g['provenance']: p['was_edited']=False
        tied=s.edited_probe(groups)
        self.assertAlmostEqual(tied['accuracy'],tied['random_baseline'])

    def test_final_truncation_drops_missing_fact(self):
        c=copy.deepcopy(self.rendered[0])
        checked=s.verify_case(c,self.models,max_chars=20)
        self.assertFalse(checked['accepted'])
        self.assertEqual(checked['drop_reason'],'final_answer_or_support')

    def test_untouched_skeleton_answerable_question_is_dropped(self):
        c=copy.deepcopy(self.rendered[0])
        for t in c['turns']: t['original_text']=t['final_text']
        checked=s.verify_case(c,self.models)
        self.assertFalse(checked['accepted'])
        self.assertEqual(checked['drop_reason'],'answerable_untouched_skeleton')

    def test_relay_is_equivalent_evidence_and_joint_support(self):
        c=next(copy.deepcopy(c) for c in self.rendered if c['qtype']=='history')
        c['turns'][0]['speaker']='DifferentReporter'
        for f in c['facts']:
            if f['id'] in c['required']:
                t=next(t for t in c['turns'] if f['id'] in t['fact_ids'])
                t['speaker']='DifferentReporter'
        checked=s.verify_case(c,self.models)
        self.assertTrue(checked['accepted'])
        self.assertTrue(any(r['support']=='joint' for r in checked['judgment']['result']['candidates']))
        self.assertTrue(all(r['quotes'] for r in checked['judgment']['result']['candidates'] if r['support']!='none'))

    def test_optional_quote_schema_and_third_family_vote(self):
        c=next(c for c in self.rendered if c['qtype']=='current'); docs,_=s.documents(c)
        cid=c['components'][0]['id']; payload=dict(components=c['components'],document='A “quoted”   value.')
        row=dict(support='full',components=[cid],quote='A "quoted" value.',reason='visible')
        self.assertTrue(s.valid_support(row,payload))
        self.assertTrue(s.valid_support({k:v for k,v in row.items() if k!='quote'},payload))
        with self.assertRaisesRegex(ValueError,'substring'):
            s.valid_support(dict(row,quote='invented quote'),payload)
        target=next(r['index'] for r in c['expected_relations'] if r['support']!='none')
        class Disagree(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                raw,fam=super().chat(prompt,**kwargs);p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_support' and p['index']==target:
                    return json.dumps(dict(support='none',components=[],reason='dissent')),fam
                return raw,fam
        models=s.seed_models(mock=True,pools=[s.SeedMockPool('q'),s.SeedMockPool('g'),Disagree('o')])
        checked=s.verify_case(c,models)
        self.assertTrue(checked['accepted'])
        judged=checked['judgment']['candidates'][target]
        self.assertFalse(judged['primary_agreement']);self.assertEqual(len(judged['votes']),3)
        self.assertAlmostEqual(judged['agreement_rate'],2/3)
        self.assertTrue(judged['consensus'])

    def test_render_retries_include_concrete_reason_and_one_message(self):
        class BrokenOnce(s.SeedMockPool):
            def __init__(self): super().__init__(); self.payloads=[]
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]); self.payloads.append(p)
                if len(self.payloads)==1: return json.dumps({'messages':[dict(id=p['messages'][0]['id'],text='broken')]}),'x'
                return super().chat(prompt,**kwargs)
        pool=BrokenOnce(); model=s.seed_models(mock=True,pools=[pool,pool,pool])
        rendered=s.render_case(self.plans[0],model)
        self.assertTrue(rendered['render_ok'])
        self.assertTrue(all(len(p['messages'])==1 for p in pool.payloads if p['task']=='render'))
        self.assertTrue(any('retry_feedback' in p for p in pool.payloads))
        self.assertTrue(any(t['render_failures'] for t in rendered['turns']))

    def test_report_and_complete_review_include_quarantines(self):
        cs=copy.deepcopy(self.verified); cs[0]['accepted']=False; cs[0]['drop_reason']='test quarantine'
        report=s.gates(cs,self.questions,self.groups,False)
        self.assertFalse(report['release_ready']); self.assertFalse(report['enforced'])
        self.assertEqual(report['attempted_questions'],24)
        with tempfile.TemporaryDirectory() as tmp:
            out=review(cs,Path(tmp)/'review.md'); text=out.read_text()
            self.assertEqual(text.count('\n## seed3_'),24)
            self.assertIn('test quarantine',text); self.assertIn('**Before:**',text); self.assertIn('**After:**',text)
            self.assertIn('host_restoration_controls',text); self.assertIn('pending',text)

    def test_build_rejects_changes_after_judgment(self):
        altered=copy.deepcopy(self.verified[0])
        altered['turns'][0]['final_text']+=' A changed answer.'
        with self.assertRaisesRegex(ValueError,'changed after'):
            s.build([altered])

    def test_build_rejects_changed_question_or_intent(self):
        altered=copy.deepcopy(self.verified[0])
        altered['question']='What was the previous value?'
        with self.assertRaisesRegex(ValueError,'semantics changed'):
            s.build([altered])

    def test_count_answer_needs_computed_operation_result(self):
        c=next(c for c in self.verified if c['qtype']=='count')
        self.assertTrue(s.answer_correct(c,dict(answer=c['answer'],abstain=False)))
        self.assertTrue(s.answer_correct(c,dict(answer='Two distinct values.',abstain=False)))
        for answer in ('99','17; 29','2 or 3','not 2'):
            self.assertFalse(s.answer_correct(c,dict(answer=answer,abstain=False)))

    def test_token_truncation_uses_actual_tokenizer(self):
        class CharacterTokenizer:
            world3_fingerprint='test-character-tokenizer'
            def encode(self,text,**kwargs): return list(map(ord,text))
            def decode(self,ids,**kwargs): return ''.join(map(chr,ids))
        checked=s.verify_case(self.rendered[0],self.models,CharacterTokenizer(),15)
        self.assertFalse(checked['accepted'])
        self.assertTrue(checked['visibility']['tokenizer_checked'])
        self.assertEqual(checked['visibility']['token_limit'],15)

    def test_unchanged_exporter_can_consume_native_groups(self):
        try:
            from world3_adapter import export_s23
        except ImportError:
            self.skipTest('optional adapter numpy dependency unavailable')
        train=[g for g in self.groups if g['split']=='train']
        self.assertGreaterEqual(len({g['ledger_id'] for g in train}),3)
        with tempfile.TemporaryDirectory() as tmp:
            manifest=export_s23(train,tmp)
            self.assertFalse(manifest['release_ready'])
            stages=[set(v) for v in manifest['stage_ledgers'].values()]
            self.assertFalse(stages[0]&stages[1] or stages[0]&stages[2] or stages[1]&stages[2])
            self.assertEqual(sum(manifest['stage_counts'].values()),len(train))

    def test_scaled_quotas(self):
        self.assertEqual(s.scaled_quotas(15),s.NEG_QUOTAS)
        for n in range(33):
            quotas=s.scaled_quotas(n); self.assertEqual(sum(quotas.values()),n)
            if n>=5: self.assertTrue(all(v>=1 for k,v in quotas.items() if s.NEG_QUOTAS[k]))

    def test_status_synonyms_are_imported(self):
        self.assertTrue(anchors_present(['approved'],'It was signed off.'))
        self.assertEqual(normalize('HOT-FIX'),'hotfix')


class Revision2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=sx.extract('irc_ubuntu',sx.read_rows([str(ROOT/'seed_raw_samples/irc_ubuntu.jsonl')]),stats=STATS)[0]
        cls.models=s.seed_models(mock=True)
        cls.facts,_=s.propose_facts(cls.skeleton,cls.models)

    def case(self,operation='current'):
        c=s.make_case(related_hosts(self.skeleton,self.facts),self.facts,operation,'observer',0,31)
        c.update(mock=True,models=self.models.identity,contract=s.CONTRACT)
        return s.write_question(c,self.models)

    def test_no_world_question_templates_or_value_pools_imported(self):
        import ast
        for filename in ('seed3.py','seed3_models.py'):
            tree=ast.parse((ROOT/filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module=='world3':
                    self.assertLessEqual({a.name for a in node.names},{'ModelFamilies','load_tokenizer'})
        self.assertFalse(hasattr(s,'VALUES'))

    def test_short_fact_values_dedup_and_program_ids(self):
        facts=copy.deepcopy(self.facts)
        facts.insert(0,copy.deepcopy(facts[0]))
        facts[0]['id']='model-invented-id'
        clean=s.validate_facts(facts,self.skeleton)
        self.assertEqual([f['id'] for f in clean],['f0','f1','f2'])
        for field,value in [('holder','Absent Speaker'),('value','maybe something'),
                            ('value',' '.join(['long']*9)),('grounding_quote','not in this window')]:
            broken=copy.deepcopy(self.facts);broken[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                s.validate_facts(broken,self.skeleton)

    def test_fact_fit_rejection_is_retained_as_planning_failure(self):
        class Ungrounded(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                payload=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if payload['task']=='seed_fact_check':
                    return json.dumps(dict(facts=[dict(id=f['id'],fits=False,reason='unrelated wedding venue') for f in payload['facts']])),'mock'
                return super().chat(prompt,**kwargs)
        pool=Ungrounded();model=s.seed_models(mock=True,pools=[pool]*3)
        cases=s.plan([self.skeleton],model)
        self.assertEqual(len(cases),3)
        for c in cases:
            self.assertFalse(c['plan_ok']);self.assertIn('unrelated wedding venue',c['plan_error'])
            self.assertEqual(len(c['fact_planning_attempts']),3)
            checked=s.verify_case(s.render_case(c,model),model)
            self.assertEqual(checked['drop_reason'],'plan_failure')
        with tempfile.TemporaryDirectory() as tmp:
            text=review(cases,Path(tmp)/'review.md').read_text()
            self.assertIn('unrelated wedding venue',text)

    def test_current_old_is_negative_but_history_count_compare_require_it(self):
        for operation in s.QTYPES:
            c=self.case(operation);old,new,_=s.find_update(c['facts'])
            old_round=next(i for i,ids in enumerate(c['layout']) if any(old['id'] in c['turns'][j]['fact_ids'] for j in ids))
            new_round=next(i for i,ids in enumerate(c['layout']) if any(new['id'] in c['turns'][j]['fact_ids'] for j in ids))
            self.assertLess(old_round,new_round)
            relation=c['expected_relations'][old_round]
            self.assertEqual(old['id'] in c['required'],operation!='current')
            self.assertEqual(relation['support']=='none',operation=='current')
            checked=s.verify_case(s.render_case(c,self.models),self.models)
            self.assertTrue(checked['accepted'],checked.get('drop_reason'))
            self.assertEqual(checked['judgment']['result']['candidates'][old_round]['support']=='none',operation=='current')
            if operation=='count': self.assertEqual(c['answer'],'2')
            if operation=='compare': self.assertEqual(c['answer'],'increased')

    def test_wrong_person_own_fact_and_relay_have_distinct_relations(self):
        c=self.case();target=next(f for f in c['facts'] if f['id'] in c['required'])
        self.assertIn('wrong_person',c['planned_kinds'])
        self.assertIn('relayed_holder',c['planned_kinds'])
        for kind in ('wrong_person','relayed_holder'):
            rid=c['planned_kinds'].index(kind)
            f=next(f for f in c['facts'] if any(f['id'] in c['turns'][i]['fact_ids'] for i in c['layout'][rid]))
            self.assertEqual(f['value'],target['value'])
            self.assertEqual(f['assertion'],'own' if kind=='wrong_person' else 'relay')
            self.assertEqual(f['holder']==target['holder'],kind=='relayed_holder')
            self.assertEqual(c['expected_relations'][rid]['support']=='none',kind=='wrong_person')

    def test_shared_value_leak_check_allows_wrong_person(self):
        c=self.case();rid=c['planned_kinds'].index('wrong_person')
        t=next(c['turns'][i] for i in c['layout'][rid] if c['turns'][i]['fact_ids'])
        assigned=[f for f in c['facts'] if f['id'] in t['fact_ids']]
        f=assigned[0]; text=f"For me, {f['value']} is the {f['attribute']} for {f['topic']}."
        # r2k: this tests shared values, using a minimally integrated negative.
        text=t['original_text'].rstrip(' .')+'; '+text
        if t['original_text'].islower():text=text.lower()
        row=dict(text=text,speaker=t['speaker'])
        self.assertFalse(s.render_failures(row,t,assigned,c['facts']))
        other=next(f for f in c['facts'] if f['value']!=assigned[0]['value'])
        row['text']+=' '+other['value']
        self.assertTrue(any('foreign value leak' in r for r in s.render_failures(row,t,assigned,c['facts'])))

    def test_loose_value_matching_and_bounded_render_checks(self):
        self.assertTrue(s.phrase_present('HOTFIX','my hot-fix is ready'))
        self.assertTrue(s.phrase_present('build-12','using build 12'))
        self.assertFalse(s.phrase_present('red','preferred'))
        message=dict(speaker='Ari',original_text='A short message about configuration.')
        facts=[dict(id='f',value='build-12')]
        good=dict(speaker='Ari',text='I use build 12 for configuration now.')
        self.assertFalse(s.render_failures(good,message,facts,facts))
        for row,reason in [(dict(good,speaker='Bela'),'speaker'),(dict(good,text=''),'empty'),
                           (dict(good,text='I use a different version now.'),'missing value'),
                           (dict(good,text=good['text']*10),'length')]:
            self.assertTrue(any(reason in r for r in s.render_failures(row,message,facts,facts)))

    def test_json_lists_fences_trailing_prose_and_incomplete_retry(self):
        row=dict(id='a',speaker='Ari',text='Hello there.')
        for value in ([row],dict(messages=[row]),row):
            raw='Here is the result:\n```json\n'+json.dumps(value)+'\n```\nDone.'
            self.assertEqual(s.parse_response(raw,'render')['messages'],[row])
        self.assertEqual(s.parse_response(json.dumps(self.facts)+' trailing','seed_plan')['facts'],self.facts)
        with self.assertRaises(ValueError):s.parse_response('{"messages":['+json.dumps(row),'render')
        class Truncated(s.SeedMockPool):
            def __init__(self):super().__init__();self.seen=[]
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);self.seen.append(p)
                if len(self.seen)==1:return '{"messages":[{"id":','mock'
                raw,f=super().chat(prompt,**kwargs)
                value=json.loads(raw)
                return '```json\n'+json.dumps(value['messages'] if p['task']=='render' else value)+'\n``` trailing',f
        pool=Truncated();models=s.seed_models(mock=True,pools=[pool]*3)
        rendered=s.render_case(self.case(),models)
        self.assertTrue(rendered['render_ok'])
        self.assertIn('no complete JSON',pool.seen[1]['retry_feedback']['reasons'][0])
        self.assertTrue(any(t['render_failures'] for t in rendered['turns']))

    def test_exhausted_render_records_every_turn_failure(self):
        class Broken(s.SeedMockPool):
            def chat(self,prompt,**kwargs):return 'incomplete {"messages":','mock'
        pool=Broken();models=s.seed_models(mock=True,pools=[pool]*3)
        c=s.render_case(self.case(),models)
        self.assertFalse(c['render_ok'])
        for t in c['turns']:
            if not t['edit_requested']:
                self.assertTrue(t['render_ok']);self.assertEqual(t['render_failures'],[])
                continue
            self.assertEqual(len(t['render_failures']),3)
            self.assertIn('no complete JSON',t['render_failures'][0]['reasons'][0])

    def test_natural_question_bans_and_answer_leaks(self):
        c=self.case()
        for q in ['Evaluate only these scoped attributes?', 'What is the effective review count?',
                  'I (Ari) reported on this. What happened?',f"Is the review count {c['answer']}?"]:
            self.assertTrue(s.question_failures(q,c),q)
        class WrongIntent(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_intent_check':return json.dumps(dict(faithful=False,reason='asks history instead of current')),'mock'
                return super().chat(prompt,**kwargs)
        pool=WrongIntent();models=s.seed_models(mock=True,pools=[pool]*3)
        with self.assertRaisesRegex(ValueError,'faithfully'):s.write_question(c,models)

    def test_answerable_without_evidence_or_inconsistent_identity_is_dropped(self):
        c=s.render_case(self.case(),self.models)
        for kind,reason in [('negative','answerable_without_evidence'),('identity','judge_disagreement_or_identity')]:
            class BadJudge(s.SeedMockPool):
                def chat(self,prompt,**kwargs):
                    raw,f=super().chat(prompt,**kwargs);p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                    if kind=='negative' and p['task']=='seed_answer' and p['control']=='negative':
                        return json.dumps(dict(answer=c['answer'],abstain=False)),f
                    if kind=='identity' and p['task']=='seed_identity':
                        return json.dumps(dict(identity_ok=False,reason='wrong holder')),f
                    return raw,f
            pool=BadJudge();models=s.seed_models(mock=True,pools=[pool]*3)
            checked=s.verify_case(c,models)
            self.assertFalse(checked['accepted']);self.assertEqual(checked['drop_reason'],reason)

    def test_skeleton_original_types_receive_same_schedule(self):
        c=self.case()
        for rid,kind in enumerate(c['planned_kinds']):
            if kind in ('same_speaker_other_topic','contentless_ack','random'):
                for i in c['layout'][rid]:
                    self.assertFalse(c['turns'][i]['fact_ids'])
                    self.assertEqual(c['turns'][i]['edit_requested'],c['turns'][i]['background_rewrite'])
        rendered=s.render_case(c,self.models)
        checked=s.verify_case(rendered,self.models)
        questions,_,groups=s.build([checked])
        report=s.gates([checked],questions,groups,False)
        self.assertFalse(report['r1_satisfied'])
        self.assertIn('pos',report['edit_provenance']['table'])

    def test_only_people_renamed_and_large_digit_free_pool(self):
        turns=[dict(tid=str(i),speaker=f'speaker{i}',text='Please check Configuration and Filename Encoding. Ubuntu works on Monday.') for i in range(48)]
        aliases={t['speaker']:[t['speaker']] for t in turns}
        renamed,_=sx.rename_all(turns,aliases)
        self.assertEqual(len({t['speaker'] for t in renamed}),48)
        for before,after in zip(turns,renamed):
            self.assertEqual(before['text'],after['text'])
            self.assertRegex(after['speaker'],r'^[A-Z][a-z]+(?: [A-Z][a-z]+)?$')
        turns=[dict(tid='a',speaker='speaker',text='Hi Alice, please check the Filename Encoding.\nRegards,\nRobert')]
        renamed,_=sx.rename_all(turns,{'speaker':['speaker']})
        self.assertNotIn('Alice',renamed[0]['text']);self.assertNotIn('Robert',renamed[0]['text'])
        self.assertIn('Filename Encoding',renamed[0]['text'])

    def test_enron_recipient_filter_and_cap_before_renaming(self):
        for n in (7,10,11,80):
            rows=[dict(subject='weekly report',date=i,body=f'I have this report number {i}.',
                **{'from':f'sender{i%2}@example.org','to':[f'person{x}@example.org' for x in range(n)]}) for i in range(3)]
            windows=sx.extract('enron',rows,stats=STATS)
            self.assertEqual(bool(windows),n<=10)
            for w in windows:
                self.assertTrue(all(len(t['recipients'])<=6 for t in w['turns']))

    def test_reextract_raw_samples_has_no_digit_pseudonyms(self):
        for source in sx.SOURCES:
            windows=sx.extract(source,sx.read_rows([str(ROOT/'seed_raw_samples'/(source+'.jsonl'))]),stats=STATS)
            self.assertTrue(windows,source)
            for w in windows:
                for t in w['turns']:
                    self.assertNotRegex(t['speaker'],r'\d')
                    self.assertNotRegex(t['text'],r'\bPerson\d+\b')
                    if source=='enron':self.assertLessEqual(len(t['recipients']),6)


class Revision2bTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=sx.extract('irc_ubuntu',sx.read_rows([str(ROOT/'seed_raw_samples/irc_ubuntu.jsonl')]),stats=STATS)[0]
        cls.facts,_=s.propose_facts(cls.skeleton,s.seed_models(mock=True))

    def extra(self):
        return dict(self.facts[2],attribute='delivery format',value='paper copy',natural_statement='My delivery format is paper copy.')

    def models(self,pool): return s.seed_models(mock=True,pools=[pool]*3)

    def test_validation_drops_each_bad_fact_and_records_reasons(self):
        extras=[None,dict(self.extra(),holder=['absent']),dict(self.extra(),value='maybe something'),
                dict(self.extra(),topic='bad=topic'),dict(self.extra(),grounding_quote='absent quote'),
                dict(self.extra(),natural_statement='No answer here.'),dict(self.extra(),attribute=None),
                dict(self.extra(),kind='invalid'),copy.deepcopy(self.facts[0])]
        dropped=[]
        clean=s.validate_facts(extras+self.facts,self.skeleton,dropped)
        self.assertEqual(len(clean),3)
        self.assertEqual(len(dropped),len(extras))
        self.assertTrue(all(d['reasons'] and d['stage']=='validation' for d in dropped))
        self.assertIn('absent quote',json.dumps(dropped))
        self.assertEqual([f['id'] for f in clean],['f0','f1','f2'])

    def test_validation_requires_count_and_update_after_dropping(self):
        with self.assertRaisesRegex(ValueError,'need 2-6'):
            s.validate_facts([self.facts[0],dict(self.facts[1],holder='Absent')],self.skeleton)
        with self.assertRaisesRegex(ValueError,'old/new same-slot'):
            s.validate_facts([self.facts[0],self.facts[2],dict(self.facts[1],holder='Absent')],self.skeleton)
        rows=[self.facts[0],self.facts[2],self.extra(),dict(self.facts[1],holder='Absent')]
        with self.assertRaisesRegex(ValueError,'old/new same-slot') as caught:
            s.validate_facts(rows,self.skeleton)
        self.assertIn('holder outside window',str(caught.exception))
        self.assertIn('remaining_facts',str(caught.exception))

    def test_plan_retry_feedback_identifies_failed_fact_and_reason(self):
        class BadOnce(s.SeedMockPool):
            def __init__(pool): super().__init__(); pool.seen=[]
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                raw,family=super().chat(prompt,**kwargs)
                if p['task']=='seed_plan':
                    pool.seen.append(p)
                    if len(pool.seen)==1:
                        r=json.loads(raw); r['facts'][1]['grounding_quote']='missing topic quote'
                        return json.dumps(r),family
                return raw,family
        pool=BadOnce(); facts,check=s.propose_facts(self.skeleton,self.models(pool))
        self.assertEqual(len(facts),3); self.assertEqual(len(pool.seen),2)
        feedback=json.dumps(pool.seen[1]['retry_feedback'])
        for text in ('grounding quote is absent','missing topic quote','input_index','old/new same-slot'):
            self.assertIn(text,feedback)
        self.assertEqual(len(check['planning_attempts']),2)
        self.assertEqual(check['planning_attempts'][0]['dropped_facts'][0]['input_index'],1)

    def test_single_fact_restatement_preserves_value_and_other_fields(self):
        class Restate(s.SeedMockPool):
            def __init__(pool): super().__init__(); pool.seen=[]
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]); pool.seen.append(p)
                raw,family=super().chat(prompt,**kwargs)
                if p['task']=='seed_plan':
                    r=json.loads(raw); r['facts'][0]['natural_statement']='I used an earlier number.'
                    return json.dumps(r),family
                if p['task']=='seed_restate':
                    r=json.loads(raw); r.update(value='999',holder='Somebody else')
                    return json.dumps(r),family
                return raw,family
        pool=Restate(); models=self.models(pool); facts,check=s.propose_facts(self.skeleton,models)
        self.assertEqual(models.calls['family0:seed_plan'],1)
        self.assertEqual(models.calls['family0:seed_restate'],1)
        self.assertTrue(s.phrase_present(facts[0]['value'],facts[0]['natural_statement']))
        self.assertEqual(facts[0]['value'],self.facts[0]['value'])
        self.assertEqual(facts[0]['holder'],self.facts[0]['holder'])
        self.assertTrue(check['planning_attempts'][0]['statement_repairs'][0]['repaired'])
        request=next(p for p in pool.seen if p['task']=='seed_restate')
        self.assertIn('fact',request); self.assertNotIn('facts',request)

    def test_failed_restatement_is_called_once_then_fact_dropped(self):
        extra=self.extra()
        class Failed(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_restate': return '{"natural_statement":','mock'
                raw,family=super().chat(prompt,**kwargs)
                if p['task']=='seed_plan':
                    r=json.loads(raw); r['facts'].append(dict(extra,natural_statement='The format changed.'))
                    return json.dumps(r),family
                return raw,family
        models=self.models(Failed()); facts,check=s.propose_facts(self.skeleton,models)
        self.assertEqual(len(facts),3)
        self.assertEqual(models.calls['family0:seed_restate'],1)
        self.assertEqual(models.calls['family0:seed_plan'],1)
        audit=check['planning_attempts'][0]
        self.assertFalse(audit['statement_repairs'][0]['repaired'])
        self.assertIn('no complete JSON',audit['dropped_facts'][0]['reasons'][0])

    def test_fact_check_keeps_plausible_new_facts_and_drops_only_nonfit(self):
        extra=self.extra()
        class Mixed(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_plan': return json.dumps(dict(facts=[extra]+self.facts)),'mock'
                if p['task']=='seed_fact_check':
                    self.assertIn('NOT required',prompt); self.assertIn('already supported by the window',prompt)
                    self.assertIn('holder attribution right',prompt); self.assertIn('not contradicting',prompt)
                    return json.dumps(dict(facts=[dict(id=f['id'],fits=i!=0,reason='incompatible speaker role' if i==0 else 'plausible new detail')
                                                  for i,f in enumerate(p['facts'])])),'mock'
                return super().chat(prompt,**kwargs)
        facts,check=s.propose_facts(self.skeleton,self.models(Mixed()))
        self.assertEqual(len(facts),3)
        self.assertEqual([f['id'] for f in facts],['f0','f1','f2'])
        drop=check['planning_attempts'][0]['dropped_facts'][0]
        self.assertEqual(drop['stage'],'fact_check'); self.assertEqual(drop['reasons'],['incompatible speaker role'])
        self.assertTrue(all(not s.phrase_present(f['value'],' '.join(t['text'] for t in self.skeleton['turns'])) for f in facts))
        case=s.make_case(self.skeleton,facts,'current','observer',0,31)
        self.assertEqual(len({f['id'] for f in case['facts']}),len(case['facts']))

    def test_fact_check_retries_when_dropping_breaks_structure(self):
        extra=self.extra()
        class BreakPair(s.SeedMockPool):
            def __init__(pool): super().__init__(); pool.plans=[]
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_plan':
                    pool.plans.append(p)
                    return json.dumps(dict(facts=self.facts+([extra] if len(pool.plans)==1 else []))),'mock'
                if p['task']=='seed_fact_check':
                    return json.dumps(dict(facts=[dict(id=f['id'],fits=not(len(pool.plans)==1 and f['id']=='f1'),
                        reason='contradicts explicit role' if f['id']=='f1' else 'fits') for f in p['facts']])),'mock'
                return super().chat(prompt,**kwargs)
        pool=BreakPair();facts,check=s.propose_facts(self.skeleton,self.models(pool))
        self.assertEqual(len(facts),3);self.assertEqual(len(check['planning_attempts']),2)
        feedback=json.dumps(pool.plans[1]['retry_feedback'])
        for text in ('old/new same-slot','contradicts explicit role','f1','fact_check'):
            self.assertIn(text,feedback)

    def test_fact_check_requires_one_typed_verdict_for_every_id(self):
        for malformed in ('missing','duplicate','unknown','nonbool','no_reason','whole_set'):
            with self.subTest(malformed=malformed):
                class BadSchema(s.SeedMockPool):
                    def __init__(pool): super().__init__(); pool.checks=[]
                    def chat(pool,prompt,**kwargs):
                        p=json.loads(prompt.split('\nPAYLOAD\n')[1]);raw,family=super().chat(prompt,**kwargs)
                        if p['task']=='seed_fact_check':
                            pool.checks.append(p)
                            if len(pool.checks)==1:
                                r=json.loads(raw)
                                if malformed=='missing': r['facts'].pop()
                                elif malformed=='duplicate': r['facts'][0]=r['facts'][1]
                                elif malformed=='unknown': r['facts'][0]['id']='unknown'
                                elif malformed=='nonbool': r['facts'][0]['fits']='true'
                                elif malformed=='no_reason': del r['facts'][0]['reason']
                                else: r=dict(grounded=True,reason='old schema')
                                return json.dumps(r),family
                        return raw,family
                pool=BadSchema();facts,_=s.propose_facts(self.skeleton,self.models(pool))
                self.assertEqual(len(facts),3);self.assertEqual(len(pool.checks),2)
                self.assertIn('f0, f1, f2',json.dumps(pool.checks[1]['retry_feedback']))

    def candidates(self,n=5):
        rows=[]
        for carrier in ('group','dyad','assistant','record'):
            for i in range(n):
                row=copy.deepcopy(self.skeleton)
                row.update(carrier=carrier,skel_id=f'{carrier}-{i}',source_conv_id=f'{carrier}-thread-{i}')
                rows.append(row)
        return rows

    def test_skeletons_replaced_in_seeded_order_until_all_three_questions_plan(self):
        rows=self.candidates(); ordered=s.pilot_candidates(rows,31)
        failures={cs[0]['skel_id'] for cs in ordered.values()}
        original=s.write_question
        def fail_one_question(case,models):
            if case['skel_id'] in failures and case['perspective']=='participant':
                raise ValueError('mock participant intent failure')
            return original(case,models)
        with patch.object(s,'write_question',side_effect=fail_one_question):
            selected,cases,report=s.plan_pilot(rows,s.seed_models(mock=True),max_attempts=4)
        self.assertEqual(len(selected),8);self.assertEqual(len(cases),24)
        self.assertTrue(all(c['plan_ok'] for c in cases));self.assertTrue(report['target_met'])
        self.assertEqual(report['total_attempts'],12)
        self.assertEqual([c['qtype'] for c in cases],s.question_sequence(24,31))
        self.assertEqual(len({c['question_id'] for c in cases}),24)
        self.assertEqual([s['skel_id'] for s in selected],[c['skel_id'] for cs in ordered.values() for c in cs[1:3]])
        for row in report['carriers'].values():
            self.assertEqual(row['attempts'],3);self.assertEqual(row['selected'],2)
            self.assertEqual(row['drop_reasons'],{'mock participant intent failure':1})
        failed=[a for a in report['attempts'] if not a['passed']]
        self.assertEqual(len(failed),4);self.assertTrue(all(len(a['failed_cases'])==3 for a in failed))

    def test_skeleton_cap_and_candidate_exhaustion(self):
        for n,cap,stop in ((5,3,'attempt_cap'),(1,3,'candidates_exhausted')):
            rows=self.candidates(n)
            with patch.object(s,'propose_facts',side_effect=ValueError('mock no update pair')) as propose:
                selected,cases,report=s.plan_pilot(rows,s.seed_models(mock=True),max_attempts=cap)
            self.assertFalse(selected);self.assertFalse(cases);self.assertFalse(report['target_met'])
            self.assertEqual(propose.call_count,min(n,cap)*4)
            for row in report['carriers'].values():
                self.assertEqual(row['stop_reason'],stop);self.assertEqual(row['shortfall'],2)
                self.assertEqual(row['drop_reasons'],{'mock no update pair':min(n,cap)})

    def test_failed_window_does_not_exclude_another_window_from_same_thread(self):
        rows=self.candidates(3)
        duplicate=copy.deepcopy(rows[0]);duplicate['skel_id']='another-window-same-thread';rows.append(duplicate)
        ordered=s.pilot_candidates(rows,31)['group']
        same_thread=[r for r in ordered if r['source_conv_id']==duplicate['source_conv_id']]
        first=same_thread[0]['skel_id']; original=s.propose_facts
        def fail_first(row,models,attempts=None):
            if row['skel_id']==first: raise ValueError('mock bad window')
            return original(row,models,attempts)
        # Put both windows first; the first fails and the second must still be tried.
        with patch.object(s,'pilot_candidates',return_value={'group':same_thread+[
                r for r in ordered if r['source_conv_id']!=duplicate['source_conv_id']]}), \
             patch.object(s,'propose_facts',side_effect=fail_first):
            selected,cases,report=s.plan_pilot(rows,s.seed_models(mock=True),max_attempts=4)
        self.assertEqual(report['carriers']['group']['attempts'],3)
        self.assertEqual(selected[0]['skel_id'],same_thread[1]['skel_id'])
        self.assertEqual(len({r['source_conv_id'] for r in selected}),2)
        # Once selected, another window from that conversation cannot fill a slot.
        selected,_,report=s.plan_pilot(rows,s.seed_models(mock=True),max_attempts=4)
        self.assertEqual(len({(r['source'],r['source_conv_id']) for r in selected}),8)

    def test_cli_writes_planning_attempt_report_and_selected_skeletons(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(s,'load_skeletons',return_value=self.candidates(3)),contextlib.redirect_stdout(io.StringIO()):
                s.main(['plan','--mock','--out',str(out)])
            report=json.loads((out/'planning_attempts.json').read_text())
            self.assertTrue(report['target_met']);self.assertEqual(report['total_attempts'],8)
            self.assertEqual(len(list(s.read_rows([str(out/'skeletons.jsonl')]))),8)
            self.assertEqual(len(json.loads((out/'plan.json').read_text())),24)


class Revision2cTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=sx.extract('irc_ubuntu',sx.read_rows([str(ROOT/'seed_raw_samples/irc_ubuntu.jsonl')]),stats=STATS)[0]
        cls.facts,_=s.propose_facts(cls.skeleton,s.seed_models(mock=True))

    def test_value_span_and_token_overlap_fallback_before_restatement(self):
        class Paraphrase(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                raw,family=super().chat(prompt,**kwargs)
                if json.loads(prompt.split('\nPAYLOAD\n')[1])['task']=='seed_plan':
                    self.assertIn('value_span',prompt); self.assertIn('exact short span',prompt)
                    r=json.loads(raw); r['facts'][2].update(value='attached to email',value_span='attached to email',
                        natural_statement='The updated list is attached to this email.')
                    return json.dumps(r),family
                return raw,family
        pool=Paraphrase();models=s.seed_models(mock=True,pools=[pool]*3)
        facts,_=s.propose_facts(self.skeleton,models)
        self.assertEqual(facts[0]['value_statement_check'],'substring')
        self.assertTrue(facts[0]['value_span_valid'])
        self.assertEqual(facts[2]['value_statement_check'],'token_overlap')
        self.assertFalse(facts[2]['value_span_valid'])
        self.assertEqual(models.calls['family0:seed_restate'],0)
        # Exactly .6 is accepted; fewer matching value tokens is rejected.
        self.assertEqual(s.value_in_statement('alpha beta gamma delta epsilon','Gamma, alpha and beta.'),'token_overlap')
        self.assertIsNone(s.value_in_statement('alpha beta gamma delta epsilon','Alpha and beta.'))
        self.assertFalse(s.phrase_present('attached to email',facts[2]['natural_statement']))

    def test_optional_fact_drop_keeps_update_and_reports_shortage(self):
        class BadOptional(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                raw,family=super().chat(prompt,**kwargs)
                if json.loads(prompt.split('\nPAYLOAD\n')[1])['task']=='seed_plan':
                    r=json.loads(raw); r['facts'][2]['grounding_quote']='absent quote'
                    return json.dumps(r),family
                return raw,family
        pool=BadOptional();models=s.seed_models(mock=True,pools=[pool]*3)
        facts,check=s.propose_facts(self.skeleton,models)
        self.assertEqual(len(facts),2); self.assertEqual(models.calls['family0:seed_plan'],1)
        audit=check['planning_attempts'][0]
        self.assertEqual(audit['negative_shortfall'],{'same_topic_wrong_attribute':1})
        self.assertEqual(audit['dropped_facts'][0]['input_index'],2)
        for operation in s.QTYPES:
            case=s.make_case(self.skeleton,facts,operation,'observer',0,31)
            case.update(mock=True,contract=s.CONTRACT)
            s.write_question(case,models)
            verified=s.verify_case(s.render_case(case,models),models)
            self.assertTrue(verified['accepted'],verified.get('drop_reason'))
            self.assertNotIn('same_topic_wrong_attribute',case['planned_kinds'])
            self.assertEqual(case['negative_shortfall']['same_topic_wrong_attribute'],1)
            questions,_,_=s.build([verified])
            self.assertGreaterEqual(questions[0]['negative_shortfall']['same_topic_wrong_attribute'],1)
            self.assertEqual(len(case['required']),1 if operation=='current' else 2)

    def test_optional_other_attribute_rejected_by_fit_is_not_a_structure_failure(self):
        class PairOnly(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_fact_check':
                    return json.dumps(dict(facts=[dict(id=f['id'],fits=i<2,reason='fits' if i<2 else 'unrelated detail')
                        for i,f in enumerate(p['facts'])])),'mock'
                return super().chat(prompt,**kwargs)
        pool=PairOnly();facts,check=s.propose_facts(self.skeleton,s.seed_models(mock=True,pools=[pool]*3))
        self.assertEqual(len(facts),2); self.assertEqual(len(check['planning_attempts']),1)
        self.assertEqual(check['planning_attempts'][0]['dropped_facts'][0]['stage'],'fact_check')

    def test_optional_other_attribute_without_a_free_host_reports_shortage(self):
        skeleton=copy.deepcopy(self.skeleton)
        skeleton['turns']=skeleton['turns'][:3]
        for i,t in enumerate(skeleton['turns']):
            t['speaker']=self.facts[0]['holder'] if i<2 else 'Another Speaker'
            t['text']='We are continuing the discussion and checking the details together. '*3
        with self.assertRaisesRegex(ValueError,'no feasible 40-60%'):
            s.make_case(skeleton,self.facts,'current','observer',0,31)

    def test_two_to_six_facts_and_update_still_required(self):
        self.assertEqual(len(s.validate_facts(self.facts[:2],self.skeleton)),2)
        rows=self.facts+[dict(self.facts[2],attribute=f'copy label {i}') for i in range(4)]
        self.assertEqual(len(s.validate_facts(rows[:6],self.skeleton)),6)
        with self.assertRaisesRegex(ValueError,'need 2-6'): s.validate_facts(rows,self.skeleton)
        for key in ('holder','topic','attribute'):
            pair=copy.deepcopy(self.facts[:2])
            pair[1][key]=next(t['speaker'] for t in self.skeleton['turns'] if t['speaker']!=pair[0]['holder']) if key=='holder' else 'another slot'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'old/new same-slot'):
                s.validate_facts(pair,self.skeleton)

    def test_question_rewrites_feed_back_judge_reasons_and_fixed_intent(self):
        class RejectTwice(s.SeedMockPool):
            def __init__(pool): super().__init__();pool.requests=[];pool.checks=0
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_question': pool.requests.append(p)
                if p['task']=='seed_intent_check':
                    pool.checks+=1
                    reject=len(pool.requests)<3 and pool.checks%2==1
                    return json.dumps(dict(faithful=not reject,reason='omits latest report' if reject else 'faithful')),'mock'
                return super().chat(prompt,**kwargs)
        pool=RejectTwice();models=s.seed_models(mock=True,pools=[pool]*3)
        case=s.make_case(self.skeleton,self.facts,'history','observer',0,31)
        before=s.digest({k:case[k] for k in ('intent','answer','facts','required','expected_relations')})
        s.write_question(case,models)
        self.assertEqual(len(case['question_attempts']),3);self.assertEqual(pool.checks,6)
        self.assertEqual(models.calls['family0:seed_question'],3)
        self.assertEqual(case['planned_semantics_sha256'],s.semantics_digest(case))
        self.assertEqual(before,s.digest({k:case[k] for k in ('intent','answer','facts','required','expected_relations')}))
        for i,p in enumerate(pool.requests[1:]):
            feedback=p['retry_feedback']
            self.assertEqual(feedback,case['question_attempts'][i])
            self.assertIn('omits latest report',json.dumps(feedback))
            self.assertEqual(feedback['intent'],case['intent'])
            self.assertTrue(feedback['question']);self.assertEqual(len(feedback['intent_checks']),2)

    def test_question_drops_only_after_three_failed_judgments_with_diagnostics(self):
        class Reject(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                if json.loads(prompt.split('\nPAYLOAD\n')[1])['task']=='seed_intent_check':
                    return json.dumps(dict(faithful=False,reason='wrong perspective')),'mock'
                return super().chat(prompt,**kwargs)
        pool=Reject();models=s.seed_models(mock=True,pools=[pool]*3)
        _,cases,report=s.plan_pilot([self.skeleton],models,per_carrier=1,max_attempts=1)
        self.assertFalse(cases)
        for case in report['attempts'][0]['failed_cases']:
            self.assertFalse(case['plan_ok']); self.assertEqual(len(case['question_attempts']),3)
            self.assertIn('wrong perspective',case['plan_error'])
            for audit in case['question_attempts']:
                self.assertEqual(audit['intent'],case['intent']);self.assertTrue(audit['question'])
                self.assertEqual([v['reason'] for v in audit['judge_reasons']],['wrong perspective']*2)
        self.assertEqual(len(report['attempts'][0]['question_planning']),3)
        self.assertEqual(models.calls['family0:seed_question'],9)

    def test_local_and_parse_failures_share_three_question_attempt_budget(self):
        class BadQuestions(s.SeedMockPool):
            def __init__(pool):super().__init__();pool.requests=[]
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_question':
                    pool.requests.append(p)
                    if len(pool.requests)==1:return '{"question":','mock'
                    if len(pool.requests)==2:return '{"question":"Evaluate only these scoped attributes?"}','mock'
                return super().chat(prompt,**kwargs)
        pool=BadQuestions();models=s.seed_models(mock=True,pools=[pool]*3)
        case=s.make_case(self.skeleton,self.facts,'current','observer',0,31)
        s.write_question(case,models)
        self.assertEqual(len(pool.requests),3)
        self.assertIn('no complete JSON',json.dumps(pool.requests[1]['retry_feedback']))
        self.assertIn('banned template phrase',json.dumps(pool.requests[2]['retry_feedback']))
        self.assertEqual(models.calls['family0:seed_intent_check'],1)

    def test_long_planning_window_caps_turns_and_total_without_mutating_source(self):
        skeleton=copy.deepcopy(self.skeleton)
        skeleton['turns']=[dict(skeleton['turns'][i%len(skeleton['turns'])],tid=str(i),
            text='reports '*10000,length_cap=80 if i==0 else 50000) for i in range(24)]
        original=copy.deepcopy(skeleton)
        self.assertEqual([t['length_cap'] for t in s.prompt_window(skeleton['turns'][:3])],[80,1500,1500])
        class Bounded(s.SeedMockPool):
            def __init__(pool):super().__init__();pool.seen={}
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1]);pool.seen[p['task']]=p
                if p['task'] in ('seed_plan','seed_fact_check'):
                    self.assertLess(len(prompt),22000) # Includes schema and fact metadata.
                return super().chat(prompt,**kwargs)
        pool=Bounded();facts,check=s.propose_facts(skeleton,s.seed_models(mock=True,pools=[pool]*3))
        window=pool.seen['seed_plan']['turns']
        self.assertEqual([t['tid'] for t in window],[t['tid'] for t in skeleton['turns']])
        self.assertEqual(len(window),24)
        self.assertLessEqual(sum(len(t['text']) for t in window),s.PROMPT_WINDOW_CHARS)
        self.assertLessEqual(len(window[0]['text']),80)
        for shown,source in zip(window,skeleton['turns']):
            self.assertLessEqual(len(shown['text']),min(source['length_cap'],s.PROMPT_TURN_CHARS))
            self.assertEqual(shown['text'],source['text'][:shown['length_cap']])
        self.assertEqual(skeleton,original)
        self.assertNotIn('excerpt',pool.seen['seed_plan']) # No duplicate full window.
        self.assertLessEqual(len(pool.seen['seed_fact_check']['window']),s.PROMPT_WINDOW_CHARS+1000)
        self.assertEqual(check['planning_attempts'][0]['prompt_window']['shown_chars'],sum(len(t['text']) for t in window))

    def test_stage_yield_counts_losses_and_marks_unrun_stages(self):
        planned=[dict(question_id=str(i),skel_id=str(i//3),plan_ok=True) for i in range(6)]
        rendered=[dict(c,render_ok=i<4) for i,c in enumerate(planned)]
        verified=[dict(c,accepted=i<2) for i,c in enumerate(rendered)]
        exported=[dict(question_id='0')]
        report=s.stage_yield(dict(input_skeletons=12,total_attempts=8),planned,rendered,verified,exported)
        self.assertEqual([(r['skeletons'],r['questions']) for r in report['rows']],[(12,None),(2,6),(2,4),(1,2),(1,1)])
        report=s.stage_yield({},[],[],[],[])
        self.assertEqual([r['questions'] for r in report['rows']],[None,0,0,0,0])
        report=s.stage_yield({},planned)
        self.assertTrue(all(r['questions'] is None for r in report['rows'][2:]))

    def test_cli_default_cap_and_yield_table_at_end_for_all_and_separate_stages(self):
        rows=[]
        for carrier in ('group','dyad','assistant','record'):
            for i in range(2):
                rows.append(dict(self.skeleton,carrier=carrier,skel_id=f'{carrier}-{i}',source_conv_id=f'{carrier}-{i}'))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);stdout=io.StringIO()
            with patch.object(s,'load_skeletons',return_value=rows),contextlib.redirect_stdout(stdout):
                s.main(['all','--mock','--skip-probes','--out',str(out)])
            selection=json.loads((out/'planning_attempts.json').read_text())
            self.assertEqual(selection['max_attempts_per_carrier'],30)
            report=json.loads((out/'stage_yield.json').read_text())
            self.assertEqual([r['stage'] for r in report['rows']],['skeletons_in','planned','rendered','verified','exported'])
            self.assertEqual([r['skeletons'] for r in report['rows']],[8]*5)
            self.assertEqual([r['questions'] for r in report['rows']],[None,24,24,24,24])
            self.assertEqual(json.loads((out/'gates.json').read_text())['stage_yield'],report)
            self.assertTrue(stdout.getvalue().splitlines()[-1].startswith('exported'))
            # A plan-only rerun must mark later stages unrun despite old artifacts.
            with patch.object(s,'load_skeletons',return_value=rows),contextlib.redirect_stdout(io.StringIO()):
                s.main(['plan','--mock','--out',str(out)])
            partial=json.loads((out/'stage_yield.json').read_text())
            self.assertTrue(all(r['questions'] is None for r in partial['rows'][2:]))
            with contextlib.redirect_stdout(io.StringIO()):
                s.main(['render','--mock','--out',str(out)])
            partial=json.loads((out/'stage_yield.json').read_text())
            self.assertEqual(partial['rows'][2]['questions'],24)
            self.assertTrue(all(r['questions'] is None for r in partial['rows'][3:]))


class Revision2dTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=sx.extract('irc_ubuntu',sx.read_rows([str(ROOT/'seed_raw_samples/irc_ubuntu.jsonl')]),stats=STATS)[0]
        cls.models=s.seed_models(mock=True)
        cls.facts,_=s.propose_facts(cls.skeleton,cls.models)

    def case(self,operation='current'):
        c=s.make_case(related_hosts(self.skeleton,self.facts),self.facts,operation,'observer',0,31)
        c.update(mock=True,models=self.models.identity,contract=s.CONTRACT)
        return s.render_case(s.write_question(c,self.models),self.models)

    def models_for(self,pool):
        return s.seed_models(mock=True,pools=[pool]*3)

    def test_small_call_payloads_are_blind_answers_and_gold_only_identity(self):
        class Capture(s.SeedMockPool):
            def __init__(self):super().__init__();self.seen=[]
            def chat(self,prompt,**kwargs):
                self.seen.append(json.loads(prompt.split('\nPAYLOAD\n')[1]));return super().chat(prompt,**kwargs)
        pool=Capture();c=s.verify_case(self.case('history'),self.models_for(pool),par=4)
        self.assertTrue(c['accepted'])
        supports=[p for p in pool.seen if p['task']=='seed_support']
        self.assertEqual(len(supports),2*len(c['visible_documents']))
        self.assertTrue(all('document' in p and 'documents' not in p for p in supports))
        answers=[p for p in pool.seen if p['task']=='seed_answer']
        self.assertTrue(answers)
        for p in answers:
            self.assertNotIn('ledger',p);self.assertNotIn('answer',p);self.assertNotIn('expected_relations',p)
            self.assertTrue(all('value' not in v for v in p['components']))
            if p['control']=='positive':self.assertEqual(p['documents'],[c['visible_documents'][i] for i in c['gold_rounds']])
            if p['control']=='negative':self.assertEqual(p['documents'],[d for i,d in enumerate(c['visible_documents']) if i not in c['gold_rounds']])
        identities=[p for p in pool.seen if p['task']=='seed_identity']
        self.assertEqual(len(identities),2*(1+len(c['host_restoration_controls'])))
        for p in identities:
            docs=c['visible_documents'] if p['control']=='gold' else next(r['documents'] for r in c['host_restoration_controls'] if p['control']==f"restore:{r['turn']}")
            self.assertEqual(p['documents'],[docs[i] for i in c['gold_rounds']])
        self.assertNotIn('seed_judge',{p['task'] for p in pool.seen})
        self.assertNotIn('seed_grade',{p['task'] for p in pool.seen})

    def test_answer_grader_only_on_loose_failure_wrong_negative_is_allowed(self):
        class Paraphrase(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_answer' and p['control'] in ('positive','negative'):
                    return json.dumps(dict(answer='twenty nine' if p['control']=='positive' else 'wrong value',abstain=False)),'mock'
                if p['task']=='seed_grade':
                    return json.dumps(dict(correct=p['answer']=='twenty nine',reason='semantic grade')),'mock'
                return super().chat(prompt,**kwargs)
        c=s.verify_case(self.case(),self.models_for(Paraphrase()),par=3)
        self.assertTrue(c['accepted']);self.assertTrue(c['negative_control']['pass_control'])
        graded=[a for a in c['verification_calls'] if a['task']=='seed_grade']
        self.assertEqual(len(graded),4)
        self.assertEqual({a['control'] for a in graded},{'positive','negative'})
        self.assertTrue(all(a['grading']=='llm' for a in c['verification_calls'] if a['task']=='seed_answer' and a['control'] in ('positive','negative')))

    def test_answer_agreement_uses_correctness_not_raw_prose(self):
        class Verbose(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                raw,f=super().chat(prompt,**kwargs);p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_answer':
                    r=json.loads(raw)
                    if not r['abstain']:r['answer']='The value is '+r['answer']+'.'
                    return json.dumps(r),f
                return raw,f
        models=s.seed_models(mock=True,pools=[s.SeedMockPool(),s.SeedMockPool(),Verbose()])
        c=s.verify_case(self.case(),models)
        self.assertTrue(c['accepted'])
        self.assertTrue(c['positive_control']['judgment']['primary_agreement'])
        self.assertEqual(len(c['positive_control']['judgment']['votes']),2)

    def test_wrong_positive_answers_and_computed_operations_are_rejected(self):
        for operation in ('current','history','count','compare'):
            case=self.case(operation)
            class Wrong(s.SeedMockPool):
                def chat(self,prompt,**kwargs):
                    p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                    if p['task']=='seed_answer' and p['control']=='positive':
                        return json.dumps(dict(answer='99' if operation in ('current','count') else '29 then 17',abstain=False)),'mock'
                    return super().chat(prompt,**kwargs)
            checked=s.verify_case(case,self.models_for(Wrong()))
            self.assertFalse(checked['accepted'],operation)
            self.assertEqual(checked['drop_reason'],'positive_or_negative_control')
        # Keep numeric boundaries in the cheap matcher.
        self.assertFalse(s.answer_correct(dict(case,answer='12 → 3'),dict(answer='1 → 23',abstain=False)))

    def test_support_prose_and_optional_quotes_do_not_create_disagreement(self):
        class NoQuotes(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                raw,f=super().chat(prompt,**kwargs);p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_support':
                    r=json.loads(raw);r.pop('quote',None);r['reason']='Different free-form explanation.'
                    return json.dumps(r),f
                return raw,f
        c=s.verify_case(self.case('history'),self.models_for(NoQuotes()))
        self.assertTrue(c['accepted'])
        self.assertTrue(all(j['primary_agreement'] for j in c['judgment']['candidates']))
        self.assertTrue(all(not r['quotes'] for r in c['judgment']['result']['candidates']))
        questions,_,groups=s.build([c]);self.assertTrue(questions);self.assertTrue(groups)
        self.assertTrue(all(not a['evidence_spans'] for a in questions[0]['answer_atoms']))
        self.assertTrue(any(r['support']=='joint' for r in c['judgment']['result']['candidates']))

    def test_normalized_quotes_export_literal_visible_spans(self):
        class Normalized(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                raw,f=super().chat(prompt,**kwargs);p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_support':
                    r=json.loads(raw)
                    if 'quote' in r:r['quote']=r['quote'].replace("'",'’').replace(' ',' \n ')
                    return json.dumps(r),f
                return raw,f
        c=s.verify_case(self.case('history'),self.models_for(Normalized()))
        self.assertTrue(c['accepted'])
        for row in c['judgment']['result']['candidates']:
            for quote in row['quotes'].values():self.assertIn(quote,c['visible_documents'][row['index']])
        self.assertTrue(s.build([c])[0])

    def test_same_partial_label_with_disjoint_components_needs_tiebreak(self):
        case=self.case('history');cid=[c['id'] for c in case['components']]
        target=next(r['index'] for r in case['expected_relations'] if r['support']!='none')
        class Partial(s.SeedMockPool):
            def __init__(self,family):super().__init__();self.family_id=family
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_support' and p['index']==target:
                    ids=[] if self.family_id==1 else [cid[0 if self.family_id==0 else 1]]
                    return json.dumps(dict(support='partial' if ids else 'none',components=ids,reason='disjoint components')),'mock'
                return super().chat(prompt,**kwargs)
        models=s.seed_models(mock=True,pools=[Partial(i) for i in range(3)])
        c=s.verify_case(case,models)
        self.assertFalse(c['accepted']);self.assertEqual(c['drop_reason'],'candidate_support_disagreement')
        self.assertEqual(len(c['judgment']['candidates'][target]['votes']),3)
        self.assertFalse(c['judgment']['candidates'][target]['consensus'])

    def test_joint_requires_every_component_and_each_planned_gold_vote(self):
        case=self.case('history');missing=case['components'][0]['id']
        class Missing(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                raw,f=super().chat(prompt,**kwargs);p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_support':
                    r=json.loads(raw)
                    if missing in r['components']:return json.dumps(dict(support='none',components=[],reason='not visible')),f
                return raw,f
        c=s.verify_case(case,self.models_for(Missing()))
        self.assertFalse(c['accepted']);self.assertEqual(c['drop_reason'],'final_answer_or_support')
        self.assertFalse(any(r['support']=='joint' for r in c['judgment']['result']['candidates']))

    def test_disagreements_are_resolved_independently_for_each_candidate(self):
        case=self.case();expected=case['expected_relations'];indices=[r['index'] for r in expected if r['support']!='none'][:2]
        self.assertEqual(len(indices),2)
        class Dissent(s.SeedMockPool):
            def __init__(self,target):super().__init__();self.target=target
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_support' and p['index']==self.target:
                    return json.dumps(dict(support='none',components=[],reason='dissent')),'mock'
                return super().chat(prompt,**kwargs)
        models=s.seed_models(mock=True,pools=[Dissent(indices[0]),s.SeedMockPool(),Dissent(indices[1])])
        c=s.verify_case(case,models,par=4)
        self.assertTrue(c['accepted'])
        self.assertEqual({a['candidate'] for a in c['verification_calls'] if a['task']=='seed_support' and a['family']==1},set(indices))
        self.assertTrue(all(len(c['judgment']['candidates'][i]['votes'])==3 for i in indices))

    def test_unexpected_full_is_equivalent_only_with_ledger_consistency(self):
        case=self.case();docs,_=s.documents(case)
        gold=next(r['index'] for r in case['expected_relations'] if r['support']=='full')
        target=next(r['index'] for r in case['expected_relations'] if r['support']=='none')
        # Add genuine visible relay evidence to an unplanned candidate.
        tid=case['layout'][target][0]
        source_tid=case['layout'][gold][0]
        case['turns'][tid]['final_text']=case['turns'][source_tid]['final_text']
        for consistent in (True,False):
            class Ledger(s.SeedMockPool):
                def chat(self,prompt,**kwargs):
                    p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                    if p['task']=='seed_equivalence':return json.dumps(dict(consistent=consistent,reason='ledger check')),'mock'
                    return super().chat(prompt,**kwargs)
            c=s.verify_case(case,self.models_for(Ledger()))
            self.assertEqual(c['accepted'],consistent)
            if consistent:
                self.assertIn(target,c['equivalent_rounds']);self.assertIn(target,c['gold_rounds'])
                self.assertNotIn(c['visible_documents'][target],c['negative_control']['documents'])
            else:self.assertEqual(c['drop_reason'],'non_gold_support_inconsistent_with_ledger')

    def test_restoration_reserializes_context_and_checks_gold_identity(self):
        case=self.case();c=s.verify_case(case,self.models)
        self.assertTrue(c['accepted'])
        for r in c['host_restoration_controls']:
            self.assertEqual(r['documents'],s.documents(case,restore=[r['turn']])[0])
            self.assertTrue(r['identity_check']['decision'])
        class BrokenIdentity(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_identity' and p['control'].startswith('restore:'):
                    return json.dumps(dict(identity_ok=False,reason='broken relay after restoration')),'mock'
                return super().chat(prompt,**kwargs)
        c=s.verify_case(case,self.models_for(BrokenIdentity()))
        self.assertFalse(c['accepted']);self.assertEqual(c['drop_reason'],'restoration_control_disagreement_or_identity')

    def test_retry_diagnostics_for_parse_quote_schema_and_recovery(self):
        class Retry(s.SeedMockPool):
            def chat(self,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_support':
                    attempt=p.get('retry_feedback',{}).get('attempt',0)
                    if attempt==0:return '{"support":','mock'
                    if attempt==1:return json.dumps(dict(support='none',components=[],quote='invented',reason='bad')),'mock'
                raw,f=super().chat(prompt,**kwargs)
                return 'Result:\n```json\n['+raw+']\n``` trailing prose',f
        c=s.verify_case(self.case(),self.models_for(Retry()),par=4)
        self.assertTrue(c['accepted'])
        for call in c['verification_calls']:
            self.assertTrue(call['ok'])
            if call['task']=='seed_support':
                self.assertEqual(len(call['attempts']),3)
                self.assertIn('no complete JSON',call['attempts'][0]['reasons'][0])
                self.assertIn('substring',call['attempts'][1]['reasons'][0])
        self.assertTrue(s.verification_yield([c])['calls_by_task']['seed_support']['failure_reasons'])

    def test_exhausted_family_is_recovered_by_gemma_but_two_failures_drop(self):
        class Broken(s.SeedMockPool):
            def chat(self,prompt,**kwargs):return '{"incomplete":','mock'
        case=self.case()
        models=s.seed_models(mock=True,pools=[s.SeedMockPool(),s.SeedMockPool(),Broken()])
        c=s.verify_case(case,models,par=4)
        self.assertTrue(c['accepted'])
        failed=[a for a in c['verification_calls'] if not a['ok']]
        self.assertTrue(failed);self.assertTrue(all(a['family']==2 and len(a['attempts'])==3 for a in failed))
        models=s.seed_models(mock=True,pools=[Broken(),s.SeedMockPool(),Broken()])
        c=s.verify_case(case,models,par=4)
        self.assertFalse(c['accepted']);self.assertEqual(c['drop_reason'],'candidate_support_disagreement')
        self.assertTrue(all(not j['consensus'] for j in c['judgment']['candidates']))
        stats=s.stage_yield({},verified=[c])['verification']
        self.assertEqual(stats['resolved'],0)
        self.assertEqual(stats['calls_by_task']['seed_support']['failed'],2*len(c['visible_documents']))
        self.assertEqual(stats['drops'],{'candidate_support_disagreement':1})

    def test_failed_answer_grader_and_identity_calls_are_not_wrong_or_abstain(self):
        case=self.case()
        for task in ('seed_answer','seed_grade','seed_identity'):
            class Broken(s.SeedMockPool):
                def chat(self,prompt,**kwargs):
                    p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                    if p['task']==task:return 'not JSON','mock'
                    if task=='seed_grade' and p['task']=='seed_answer':return json.dumps(dict(answer='not sure',abstain=False)),'mock'
                    return super().chat(prompt,**kwargs)
            c=s.verify_case(case,self.models_for(Broken()))
            self.assertFalse(c['accepted'],task)
            self.assertTrue(any(a['task']==task and not a['ok'] for a in c['verification_calls']))

    def test_par_limits_global_calls_and_really_runs_small_calls_concurrently(self):
        class Concurrent(s.SeedMockPool):
            def __init__(self):super().__init__();self.lock=threading.Lock();self.active=0;self.peak=0
            def chat(self,prompt,**kwargs):
                with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
                try:
                    time.sleep(.001)
                    return super().chat(prompt,**kwargs)
                finally:
                    with self.lock:self.active-=1
        case=self.case()
        for par in (1,4):
            pool=Concurrent();models=self.models_for(pool)
            checked=s.verify_cases([case,copy.deepcopy(case)],models,par=par)
            self.assertTrue(all(c['accepted'] for c in checked))
            self.assertLessEqual(pool.peak,par)
            self.assertEqual(pool.peak,par)

    def test_stage_yield_and_gates_include_candidate_control_and_call_failures(self):
        case=self.case();case=s.verify_case(case,self.models)
        report=s.stage_yield({},verified=[case])
        v=report['verification']
        self.assertEqual(v['candidates'],len(case['visible_documents']))
        self.assertEqual(v['resolved'],v['candidates']);self.assertEqual(v['primary_agreement_rate'],1)
        self.assertEqual(v['controls']['positive_control'],dict(checked=1,passed=1))
        self.assertEqual(v['controls']['negative_control'],dict(checked=1,passed=1))
        self.assertEqual(v['controls']['untouched_control'],dict(checked=1,passed=1))
        self.assertEqual(v['controls']['identity_check'],dict(checked=1,passed=1))
        self.assertTrue(v['calls_by_task']);self.assertEqual(v['drops'],{})
        questions,_,groups=s.build([case]);g=s.gates([case],questions,groups,run_probes=False)
        self.assertEqual(g['verification'],v);self.assertEqual(g['judge_agreement']['unit'],'candidate')
        self.assertEqual(g['judge_agreement']['n'],v['candidates'])


class Revision2eTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=sx.extract('irc_ubuntu',sx.read_rows([str(ROOT/'seed_raw_samples/irc_ubuntu.jsonl')]),stats=STATS)[0]
        cls.facts,_=s.propose_facts(cls.skeleton,s.seed_models(mock=True))

    def models(self,pool): return s.seed_models(mock=True,pools=[pool]*3)

    def case(self,operation='current'):
        models=s.seed_models(mock=True)
        c=s.make_case(related_hosts(self.skeleton,self.facts),self.facts,operation,'observer',0,31)
        c.update(mock=True,contract=s.CONTRACT)
        return s.write_question(c,models)

    def native_fixture(self,partial=False):
        sk=copy.deepcopy(self.skeleton); facts=copy.deepcopy(self.facts[:1 if partial else 2])
        for f,i in zip(facts,(0,2)):
            phase='Earlier' if i==0 else 'Now'
            f['natural_statement']=f"{phase}, {f['holder']}'s {f['topic']} {f['attribute']} is {f['value']}."
            sk['turns'][i]['text']=f['natural_statement']
            f.update(tid=sk['turns'][i]['tid'],span=f['natural_statement'],grounding_quote=f['natural_statement'])
        return sk,facts

    def native_pool(self,facts):
        class Native(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                payload=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if payload['task']=='seed_native':return json.dumps(dict(facts=facts)),'mock'
                return super().chat(prompt,**kwargs)
        return Native()

    def test_background_register_rejects_commands_numbers_pronouns_and_case(self):
        checks=[
            ('dpkg -i file.deb','use dpkg -ied file.deb','protected_tokens'),
            ('try foo_bar(x) at https://a.test urlLink','try foo_bar(y) at https://a.test link','protected_tokens'),
            ('i have 11:30 in my notes','i have 12:30 in my notes','numbers'),
            ('i really like this one','Farah really likes this one','pronouns'),
            ('you can try it now','we can try it now','pronouns'),
            ('we can try this thing','We can try that thing','casing')]
        for before,after,key in checks:
            with self.subTest(key=key,before=before):
                self.assertFalse(s.register_checks(before,after)['checks'][key])
                self.assertTrue(s.render_failures(dict(text=after,speaker='A'),dict(original_text=before,speaker='A'),[],[]))

    def test_background_similarity_and_length_band(self):
        text='we can try this again soon'
        for rewritten,key in ((text,'similarity'),('we should discuss something completely different','similarity'),
                              ('we can try this again soon with lots of extra detail','length')):
            self.assertFalse(s.register_checks(text,rewritten)['checks'][key])
        good='we can test that again soon'
        self.assertTrue(s.register_checks(text,good)['passed'])
        self.assertEqual(s.render_failures(dict(text=good,speaker='A'),dict(original_text=text,speaker='A'),[],[]),[])

    def test_exact_schedule_matches_feasible_types_and_protects_native_positives(self):
        kinds=['pos','pos','wrong_person','wrong_person','random','random','contentless_ack','contentless_ack']
        turns=[dict(tid=str(i),speaker='A',text='we can try this again soon') for i in range(8)]
        facts=[dict(id='native',native=True),dict(id='added')]
        report=s.schedule_rewrites(turns,[[i] for i in range(8)],kinds,{0:['native'],1:['added']},s.random.Random(3),facts)
        self.assertEqual(report['target_edit_rate'],.5)
        self.assertFalse(turns[0]['edit_requested']); self.assertTrue(turns[1]['edit_requested'])
        self.assertTrue(report['matching_feasible'])
        self.assertTrue(all(row['target_edit_rate']==.5 for row in report['strata'].values()))
        self.assertEqual(sum(t['background_rewrite'] for t in turns),3)

    def test_impossible_budget_fails_and_failed_edits_never_change_schedule(self):
        turns=[dict(tid=str(i),speaker='A',text='we can try this again soon') for i in range(4)]
        with self.assertRaisesRegex(ValueError,'no feasible'):
            s.schedule_rewrites(turns,[[i] for i in range(4)],['pos']*4,{i:['f'] for i in range(3)},s.random.Random(1))
        class Unchanged(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='render':
                    m=p['messages'][0]
                    return json.dumps(dict(messages=[dict(id=m['id'],speaker=m['speaker'],text=m['original_text'])])),'mock'
                return super().chat(prompt,**kwargs)
        case=self.case(); checked=s.render_case(case,self.models(Unchanged()))
        self.assertFalse(checked['render_ok'])
        self.assertFalse(checked['rewrite_schedule']['actual_matches_schedule'])
        self.assertEqual(checked['rewrite_schedule']['selected_turn_indices'],case['rewrite_schedule']['selected_turn_indices'])
        self.assertTrue(all(not t['render_ok'] for t in checked['turns'] if t['edit_requested']))

    def test_insertion_never_appends_unchanged_host(self):
        message=dict(speaker='A',original_text='Would you like to have lunch with me?')
        fact=dict(id='f',value='noon')
        reasons=s.render_failures(dict(speaker='A',text=message['original_text']+' At noon.'),message,[fact],[fact])
        self.assertTrue(any('disconnected append' in r for r in reasons))

    def test_host_selection_prefers_local_topic_over_distant_long_turns(self):
        sk=copy.deepcopy(self.skeleton); holder=self.facts[0]['holder']
        sk['turns']=[dict(tid=str(i),speaker=holder if i%2==0 else 'Other',role='user',
            text=('guitar rehearsal and music practice. ' if i in (0,2) else 'The weather is clear and the road is quiet. ')*4) for i in range(12)]
        facts=copy.deepcopy(self.facts[:2])
        for f,value in zip(facts,('acoustic','electric')):
            f.update(topic='guitar rehearsal',attribute='guitar',value=value,value_span=value,
                natural_statement='I play the '+value+' guitar.',grounding_quote='guitar rehearsal and music practice.')
        case=s.make_case(sk,facts,'history','self',0,31)
        old,new,_=s.find_update(case['facts'])
        self.assertEqual([old['source']['position'],new['source']['position']],[0,2])

    def test_local_judge_retries_with_context_and_reason(self):
        class Coherence(s.SeedMockPool):
            def __init__(pool): super().__init__();pool.bad=False;pool.requests=[]
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='render': pool.requests.append(p)
                if p['task']=='seed_content_check' and p['message']['assigned_facts'] and not pool.bad:
                    pool.bad=True
                    return json.dumps(dict(coherence=2,contradiction=True,same_voice=True,natural_value=True,
                        reason='Contradicts the reply; integrate the time into the invitation.')),'mock'
                return super().chat(prompt,**kwargs)
        pool=Coherence(); checked=s.render_case(self.case(),self.models(pool))
        self.assertTrue(checked['render_ok'])
        retry=next(p for p in pool.requests if 'retry_feedback' in p)
        self.assertIn('Contradicts the reply',str(retry['retry_feedback']))
        self.assertIn('previous_turns',retry);self.assertIn('next_turns',retry)
        self.assertTrue(any(len(t.get('content_checks',[]))==2 for t in checked['turns']))

    def test_local_judge_voice_value_and_exhaustion_are_hard_failures(self):
        for field in ('coherence','contradiction','same_voice','natural_value'):
            class Reject(s.SeedMockPool):
                def chat(pool,prompt,**kwargs):
                    p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                    if p['task']=='seed_content_check':
                        r=dict(coherence=5,contradiction=False,same_voice=True,natural_value=True,reason='bad '+field)
                        r[field]=2 if field=='coherence' else field=='contradiction'
                        return json.dumps(r),'mock'
                    return super().chat(prompt,**kwargs)
            checked=s.render_case(self.case(),self.models(Reject()))
            self.assertFalse(checked['render_ok'])
            self.assertTrue(all(len(t['render_failures'])==3 for t in checked['turns'] if t['edit_requested']))

    def test_wrong_person_generated_in_own_voice_and_never_copied(self):
        case=self.case(); wrong=next(f for f in case['facts'] if f.get('wrong_person'))
        self.assertEqual(wrong['natural_statement'],'')
        self.assertNotEqual(wrong['holder'],case['intent']['holder'])
        rendered=s.render_case(case,s.seed_models(mock=True))
        turn=next(t for t in rendered['turns'] if wrong['id'] in t['fact_ids'])
        self.assertIn('As for me',turn['text'])
        self.assertFalse(s.near_duplicate(turn['text'],wrong['holder_statement']))
        spec=dict(speaker=turn['speaker'],original_text='x'*len(wrong['holder_statement']))
        copied=s.render_failures(dict(speaker=turn['speaker'],text=wrong['holder_statement']),spec,[wrong],case['facts'])
        self.assertTrue(any('near-copies' in reason for reason in copied))
        self.assertTrue(s.near_duplicate('Something different. '+wrong['holder_statement'],wrong['holder_statement']))

    def test_wrong_person_final_dedupe_including_future_holder(self):
        case=self.case(); wrong=next(f for f in case['facts'] if f.get('wrong_person'))
        holder=next(f for f in case['facts'] if f['id'] in case['required'])
        # Near-copy detection must also catch a different name or punctuation.
        before=f"Now, {holder['holder']}'s {holder['topic']} {holder['attribute']} is {holder['value']}."
        copied=before.replace(holder['holder'],wrong['holder'])
        self.assertTrue(s.near_duplicate(before,copied))
        class Copy(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='render' and any(f.get('wrong_person') for f in p['messages'][0]['assigned_facts']):
                    m=p['messages'][0]
                    return json.dumps(dict(messages=[dict(id=m['id'],speaker=m['speaker'],text=copied)])),'mock'
                return super().chat(prompt,**kwargs)
        self.assertFalse(s.render_case(case,self.models(Copy()))['render_ok'])

    def test_native_extraction_precedes_proposals_validates_spans_and_reuses_update(self):
        sk,native=self.native_fixture();pool=self.native_pool(native);models=self.models(pool)
        facts,audit=s.propose_facts(sk,models)
        self.assertEqual(models.calls['family0:seed_native'],1)
        self.assertEqual(models.calls['family0:seed_plan'],0)
        self.assertEqual(len(facts),2)
        self.assertTrue(all(f['native'] for f in facts))
        self.assertFalse(audit['planning_attempts'][0]['inserted_facts_needed'])
        for qtype in s.QTYPES:
            case=s.make_case(sk,facts,qtype,'self',0,31);case.update(mock=True)
            case=s.write_question(case,models);rendered=s.render_case(case,models)
            for f in facts:
                t=next(t for t in rendered['turns'] if f['id'] in t['fact_ids'])
                self.assertEqual(t['tid'],f['native_tid']);self.assertEqual(t['edit_kind'],'none')
            checked=s.verify_case(rendered,models)
            self.assertTrue(checked['accepted'],checked.get('drop_reason'))
            self.assertTrue(checked['untouched_control']['expected_answerable'])
            self.assertTrue(checked['untouched_control']['pass_control'])
            self.assertTrue(checked['negative_control']['pass_control'])
            questions,_,_=s.build([checked])
            positives=[p for p in questions[0]['candidate_provenance'] if p['final_support']['support']!='none']
            native_positives=[p for p in positives if p['native_fact_ids']]
            self.assertTrue(native_positives)
            self.assertTrue(all(not p['was_edited'] for p in native_positives))
            self.assertLess(s.provenance_table(questions)['table']['pos']['edited_rate'],1)

    def test_native_span_failures_retry_exact_tid_and_literal_value(self):
        sk,native=self.native_fixture()
        for field,value in (('tid','missing'),('span','not a literal quote'),('value','999'),('value_span','wrong span')):
            broken=copy.deepcopy(native);broken[0][field]=value
            models=self.models(self.native_pool(broken))
            # Invalid native facts are dropped (not fatal); the broken one must never be
            # accepted as native evidence, and extraction is called once.
            try:
                facts,_=s.propose_facts(sk,models)
            except ValueError:
                facts=[]
            self.assertEqual(models.calls['family0:seed_native'],1)
            bad=broken[0]
            self.assertFalse(any(f.get('native') and f.get('native_tid')==bad.get('tid') and f.get('native_span')==bad.get('span')
                                 and f.get('value')==bad.get('value') for f in facts))

    def test_partial_native_only_adds_missing_structure_and_preserves_original(self):
        sk,native=self.native_fixture(partial=True);models=self.models(self.native_pool(native))
        facts,_=s.propose_facts(sk,models)
        self.assertEqual(len(facts),3)
        self.assertTrue(facts[0]['native'])
        self.assertEqual(sum(f.get('native',False) for f in facts),1)
        self.assertEqual(models.calls['family0:seed_plan'],1)
        case=s.make_case(sk,facts,'history','observer',0,31);case.update(mock=True)
        checked=s.verify_case(s.render_case(s.write_question(case,models),models),models)
        self.assertTrue(checked['accepted'],checked.get('drop_reason'))
        self.assertFalse(checked['untouched_control']['expected_answerable'])

    def test_daily_dialog_real_time_proposals_reused_without_duplicate_insertions(self):
        sk=copy.deepcopy(self.skeleton)
        texts=['Is 11:30 AM OK ?', "Sorry , I didn't hear you . Can you say that again please ?",
               'I said , 11:30 AM .', "Oh , I'm busy then . Can we meet a little later ?",
               'OK , how about 12:30 PM ?', 'OK . Where ?', "How about Bill's Seafood Restaurant ?", "Oh , Where is that ?"]
        sk['turns']=[dict(tid=str(i),speaker='Pia' if i%2==0 else 'Jules',role='user',text=text) for i,text in enumerate(texts)]
        native=[]
        for i,value in ((2,'11:30 AM'),(4,'12:30 PM')):
            native.append(dict(kind='plan',holder='Pia',topic='lunch',attribute='proposed time',value=value,value_span=value,
                natural_statement=texts[i],grounding_quote=texts[i],tid=str(i),span=texts[i]))
        models=self.models(self.native_pool(native));facts,_=s.propose_facts(sk,models)
        self.assertEqual([f['value'] for f in facts],['11:30 AM','12:30 PM'])
        case=s.make_case(sk,facts,'history','self',0,31)
        self.assertEqual(len([f for f in case['facts'] if f['holder']=='Pia']),2)
        self.assertTrue(all(not t['edit_requested'] for t in case['turns'] if t['tid'] in ('2','4')))
        self.assertEqual(models.calls['family0:seed_plan'],0)

    def test_jargon_heuristic_and_fact_judge_reject_odd_values(self):
        self.assertTrue(s.natural_value_failures('linguistic blending synthesis'))
        self.assertTrue(s.natural_value_failures('conceptual integration framework'))
        for value in ("Bill's Seafood Restaurant",'12:30 PM','paper copy','linguistic analysis'):
            self.assertEqual(s.natural_value_failures(value),[])
        facts=copy.deepcopy(self.facts);facts[2].update(value='linguistic blending synthesis',natural_statement='I prefer linguistic blending synthesis.')
        dropped=[];clean=s.validate_facts(facts,self.skeleton,dropped)
        self.assertEqual(len(clean),2);self.assertIn('abstract noun stack',str(dropped))

    def test_question_tails_vocatives_and_self_perspective_are_banned(self):
        case=self.case();case['asker']='Pia'
        for text in ('What changed, considering the earlier and latest reports?',
                     'What changed in order from old to new?', 'What is the effective choice?',
                     'Which scoped value changed?', 'Pia, what changed?', 'What changed, Pia?', 'Pia what changed?'):
            self.assertTrue(s.question_failures(text,case),text)
        case['intent'].update(asker='Pia',holder='Pia')
        self.assertTrue(s.question_failures("What is Pia's choice now?",case))
        self.assertFalse(s.question_failures('What is my choice now?',case))

    def test_question_naturalness_rejection_feeds_back_then_recovers(self):
        class Natural(s.SeedMockPool):
            def __init__(pool):super().__init__();pool.bad=False;pool.requests=[]
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_question':pool.requests.append(p)
                if p['task']=='seed_intent_check' and not pool.bad:
                    pool.bad=True;return json.dumps(dict(faithful=False,reason='Unnatural report-order tail.')),'mock'
                return super().chat(prompt,**kwargs)
        pool=Natural();s.write_question(self.case(),self.models(pool))
        self.assertEqual(len(pool.requests),2)
        self.assertIn('Unnatural report-order tail',str(pool.requests[1]['retry_feedback']))
        self.assertIn('natural',s.INTENT_PROMPT)

    def test_content_review_contains_every_turn_and_register_summary_even_failures(self):
        case=s.render_case(self.case(),s.seed_models(mock=True))
        case['turns'][0]['render_error']='quarantine example'
        with tempfile.TemporaryDirectory() as tmp:
            s.dump_content_review([case],tmp)
            obj=json.loads((Path(tmp)/'content_review.json').read_text())
            variant=obj['windows'][case['skel_id']][0]
            self.assertEqual(len(variant['turns']),len(case['turns']))
            self.assertTrue(variant['register_summary'])
            text=(Path(tmp)/'content_review.md').read_text()
            self.assertIn('| Before | After |',text)
            self.assertIn('quarantine example',text)
            self.assertIn('none',text)
            for t in case['turns']:
                self.assertTrue(any(r['tid']==t['tid'] and r['original_text']==t['original_text'] and
                    r['final_text']==t['final_text'] for r in variant['turns']))


class Revision2fTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=sx.extract('irc_ubuntu',sx.read_rows([str(ROOT/'seed_raw_samples/irc_ubuntu.jsonl')]),stats=STATS)[0]

    def candidates(self,n=8):
        return [dict(copy.deepcopy(self.skeleton),skel_id=f'parallel-{i}',source_conv_id=f'thread-{i}') for i in range(n)]

    def test_parallel_selection_matches_serial_including_slot_dependent_failure(self):
        rows=self.candidates(); original_facts=s.plan_facts; original_question=s.write_question
        class Concurrent(s.SeedMockPool):
            def __init__(self): super().__init__();self.lock=threading.Lock();self.active=0;self.peak=0
            def chat(self,prompt,**kwargs):
                with self.lock: self.active+=1;self.peak=max(self.peak,self.active)
                try:
                    time.sleep(.002)
                    return super().chat(prompt,**kwargs)
                finally:
                    with self.lock:self.active-=1
        def question(case,models):
            if case['perspective']=='participant' and (case['skel_id']==rows[0]['skel_id'] or
                    (case['skel_id']==rows[1]['skel_id'] and case['id']=='c4')):
                raise ValueError('slot-dependent mock failure')
            return original_question(case,models)
        def facts(row,models):
            if row['skel_id']==rows[0]['skel_id']: time.sleep(.06)
            return original_facts(row,models)
        outputs=[]
        with patch.object(s,'pilot_candidates',return_value={'group':rows}), \
             patch.object(s,'write_question',side_effect=question),patch.object(s,'plan_facts',side_effect=facts):
            for par in (1,4):
                pool=Concurrent();models=s.seed_models(mock=True,pools=[pool]*3)
                outputs.append(s.plan_pilot(rows,models,max_attempts=8,par=par,progress=False))
                self.assertLessEqual(pool.peak,par)
                self.assertGreaterEqual(pool.peak,min(par,3))
        self.assertEqual(outputs[0][:2],outputs[1][:2])
        selected,cases,report=outputs[1]
        self.assertEqual([r['skel_id'] for r in selected],['parallel-1','parallel-2'])
        self.assertEqual([c['qtype'] for c in cases],s.question_sequence(24,31)[:6])
        self.assertLessEqual(report['total_attempts'],6) # Resolved prefix (3) plus at most par-1 speculative rows.
        self.assertEqual(report['carriers']['group']['stop_reason'],'target_met')

    def test_checkpoint_and_incremental_report_exist_while_earlier_attempt_is_blocked(self):
        rows=self.candidates(5);release=threading.Event();original=s.plan_facts
        observed=threading.Event();errors=[]
        def facts(row,models):
            if row['skel_id']=='parallel-0':
                if not release.wait(5): raise RuntimeError('test timed out waiting for incremental checkpoint')
            return original(row,models)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def watch():
                try:
                    deadline=time.monotonic()+5
                    while time.monotonic()<deadline:
                        path=out/'planning_attempts.json'
                        report=json.loads(path.read_text()) if path.exists() else {}
                        attempts=report.get('attempts',[])
                        if attempts:
                            self.assertNotEqual(attempts[0]['skel_id'],'parallel-0')
                            self.assertTrue((out/'planning_ckpt'/(attempts[0]['skel_id']+'.json')).exists())
                            self.assertEqual(json.loads((out/'plan.json').read_text()),[])
                            observed.set();break
                        time.sleep(.005)
                except BaseException as exc: errors.append(exc)
                finally: release.set()
            watcher=threading.Thread(target=watch);watcher.start()
            output=io.StringIO()
            with patch.object(s,'pilot_candidates',return_value={'group':rows}), \
                 patch.object(s,'plan_facts',side_effect=facts),contextlib.redirect_stdout(output):
                selected,_,report=s.plan_pilot(rows,s.seed_models(mock=True),par=3,out=out)
            watcher.join()
            self.assertFalse(errors,errors);self.assertTrue(observed.is_set())
            self.assertEqual([r['skel_id'] for r in selected],['parallel-0','parallel-1'])
            self.assertLessEqual(report['total_attempts'],3)
            self.assertEqual(len(list((out/'planning_ckpt').glob('*.json'))),report['total_attempts'])
            self.assertIn('carrier=group skel_id=parallel-',output.getvalue())
            self.assertIn('pass reason=ok completed=',output.getvalue())

    def test_interrupted_planning_resumes_failures_and_successes_without_repeating_calls(self):
        rows=self.candidates(6);original=s.plan_facts
        def crash(row,models):
            if row['skel_id']=='parallel-2': raise RuntimeError('simulated interruption')
            if row['skel_id']=='parallel-0':
                return dict(proposals=[],grounding=None,attempts=[],error='permanent mock failure')
            return original(row,models)
        with tempfile.TemporaryDirectory() as tmp,patch.object(s,'pilot_candidates',return_value={'group':rows}):
            out=Path(tmp)
            with patch.object(s,'plan_facts',side_effect=crash),self.assertRaisesRegex(RuntimeError,'interruption'):
                s.plan_pilot(rows,s.seed_models(mock=True),par=1,out=out,progress=False)
            partial=json.loads((out/'planning_attempts.json').read_text())
            self.assertEqual(partial['total_attempts'],2)
            self.assertFalse(partial['attempts'][0]['passed'])
            def resume(row,models):
                self.assertNotIn(row['skel_id'],('parallel-0','parallel-1'))
                return original(row,models)
            with patch.object(s,'plan_facts',side_effect=resume):
                selected,cases,report=s.plan_pilot(rows,s.seed_models(mock=True),par=3,out=out,progress=False)
            self.assertEqual([r['skel_id'] for r in selected],['parallel-1','parallel-2'])
            self.assertTrue(all(a['cache_hit'] for a in report['attempts'][:2]))
            # Same run and concurrency: even question-slot reconciliations are cached.
            models=s.seed_models(mock=True)
            with patch.object(s,'plan_facts',side_effect=AssertionError('facts repeated')), \
                 patch.object(s,'plan',side_effect=AssertionError('question planning repeated')):
                for par in (1,3,4):
                    again=s.plan_pilot(rows,models,par=par,out=out,progress=False)
                    self.assertEqual(again[:2],(selected,cases))
            self.assertFalse(models.calls)

    def test_planning_fingerprints_invalidate_inputs_prompts_models_and_seed(self):
        rows=self.candidates(1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(s,'pilot_candidates',side_effect=lambda xs,seed:{'group':xs}):
            def run(xs=rows,models=None,seed=31):
                return s.plan_pilot(xs,models or s.seed_models(mock=True),per_carrier=1,
                    max_attempts=1,seed=seed,out=tmp,progress=False)
            run()
            for change in ('input','prompt','model','seed'):
                run()
                changed=copy.deepcopy(rows);models=s.seed_models(mock=True);seed=31
                if change=='input':changed[0]['turns'][0]['text']+=' extra context'
                if change=='model':models.identity=['changed model']*3
                if change=='seed':seed=77
                with patch.object(s,'PLAN_PROMPT',s.PLAN_PROMPT+(' changed' if change=='prompt' else '')), \
                     patch.object(s,'plan_facts',wraps=s.plan_facts) as facts:
                    run(changed,models,seed)
                self.assertEqual(facts.call_count,1,change)

    def test_parallel_duplicate_conversations_and_attempt_cap_match_serial(self):
        rows=self.candidates(5)
        rows[1]['source_conv_id']=rows[0]['source_conv_id']
        original=s.plan_facts
        for failed in (False,True):
            def facts(row,models):
                if failed and row['skel_id']=='parallel-0':
                    return dict(proposals=[],grounding=None,attempts=[],error='failed first window')
                return original(row,models)
            with patch.object(s,'pilot_candidates',return_value={'group':rows}),patch.object(s,'plan_facts',side_effect=facts):
                serial=s.plan_pilot(rows,s.seed_models(mock=True),max_attempts=3,par=1,progress=False)
                parallel=s.plan_pilot(rows,s.seed_models(mock=True),max_attempts=3,par=4,progress=False)
            self.assertEqual(serial[:2],parallel[:2])
            self.assertLessEqual(parallel[2]['total_attempts'],3)
            self.assertEqual(len({r['source_conv_id'] for r in parallel[0]}),2)

    def test_render_and_verify_resume_and_invalidate_question_checkpoints(self):
        models=s.seed_models(mock=True)
        plans=s.plan(self.candidates(1),models)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rendered=s.render_cases(plans,models,par=3,out=out)
                verified=s.verify_cases(rendered,models,par=3,out=out)
            verified=json.loads(json.dumps(verified)) # Persisted diagnostic tuples are JSON arrays.
            self.assertTrue(all(c['accepted'] for c in verified))
            for stage in ('render','verify'):
                self.assertEqual(len(list((out/(stage+'_ckpt')).glob('*.json'))),3)
                self.assertIn(stage+' carrier=group',stdout.getvalue())
                self.assertEqual(json.loads((out/(stage+'.json')).read_text()),rendered if stage=='render' else verified)
            fresh=s.seed_models(mock=True)
            with patch.object(s,'render_case',side_effect=AssertionError('render repeated')), \
                 patch.object(s,'verify_case',side_effect=AssertionError('verify repeated')):
                self.assertEqual(s.render_cases(plans,fresh,par=1,out=out,progress=False),rendered)
                self.assertEqual(s.verify_cases(rendered,fresh,par=1,out=out,progress=False),verified)
            self.assertFalse(fresh.calls)
            # A bad/truncated checkpoint only recomputes its own question.
            path=out/'render_ckpt'/(plans[1]['question_id']+'.json');path.write_text('{')
            with patch.object(s,'render_case',wraps=s.render_case) as render:
                s.render_cases(plans,models,par=3,out=out,progress=False)
            self.assertEqual(render.call_count,1)
            # Visibility parameters participate in the verification fingerprint.
            with patch.object(s,'verify_case',wraps=s.verify_case) as verify:
                s.verify_cases(rendered,models,max_chars=20,par=3,out=out,progress=False)
            self.assertEqual(verify.call_count,3)

    def test_question_stage_resumes_after_interruption_and_checkpoints_failed_results(self):
        cases=[dict(question_id=f'q{i}',carrier='group',skel_id=f's{i}') for i in range(3)]
        def interrupted(case):
            if case['question_id']=='q1': raise RuntimeError('interrupted question stage')
            return dict(case,render_ok=case['question_id']!='q0',plan_error='mock render failure')
        with tempfile.TemporaryDirectory() as tmp:
            models=s.seed_models(mock=True)
            with self.assertRaisesRegex(RuntimeError,'interrupted question'):
                s.checkpointed_cases('render',interrupted,cases,models,1,tmp,False)
            self.assertTrue((Path(tmp)/'render_ckpt/q0.json').exists())
            def resumed(case):
                self.assertNotEqual(case['question_id'],'q0')
                return dict(case,render_ok=True)
            result=s.checkpointed_cases('render',resumed,cases,models,2,tmp,False)
            self.assertFalse(result[0]['render_ok'])
            self.assertTrue(all(c['render_ok'] for c in result[1:]))
            self.assertEqual([c['question_id'] for c in result],['q0','q1','q2'])

    def test_cli_all_uses_par_and_resume_performs_no_inference(self):
        rows=[dict(copy.deepcopy(self.skeleton),carrier=carrier,skel_id=f'{carrier}-{i}',
                   source_conv_id=f'{carrier}-thread-{i}')
              for carrier in ('group','dyad','assistant','record') for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp,patch.object(s,'load_skeletons',return_value=rows):
            args=['all','--mock','--par','4','--skip-probes','--out',tmp]
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(s,'plan_pilot',wraps=s.plan_pilot) as planning, \
                 patch.object(s,'render_cases',wraps=s.render_cases) as rendering, \
                 patch.object(s,'verify_cases',wraps=s.verify_cases) as verifying, \
                 patch('builtins.print',wraps=print) as printing:
                s.main(args)
            self.assertEqual(planning.call_args.args[5],4)
            self.assertEqual(rendering.call_args.args[2],4)
            self.assertEqual(verifying.call_args.args[5],4)
            lines=[c for c in printing.call_args_list if c.args and
                   str(c.args[0]).startswith(('planning carrier=','render carrier=','verify carrier='))]
            self.assertEqual(len(lines),8+24+24)
            self.assertTrue(all(c.kwargs.get('flush') is True for c in lines))
            before=json.loads((Path(tmp)/'questions.json').read_text())
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(s.SeedMockPool,'chat',side_effect=AssertionError('completed CLI work repeated')):
                s.main(args)
            self.assertEqual(json.loads((Path(tmp)/'questions.json').read_text()),before)
            self.assertEqual(json.loads((Path(tmp)/'all_model_usage.json').read_text())['calls'],{})


class Revision2kTests(unittest.TestCase):
    def fixture(self):
        sk=sx.extract(
            'irc_ubuntu',sx.read_rows([str(ROOT/'seed_raw_samples/irc_ubuntu.jsonl')]),stats=STATS)[0]
        sk['turns']=[dict(tid=str(i),speaker='Ari' if i%2==0 else 'Kiran',role='user',
            text='For the backend language, keep the API stable while we test the deployment carefully.') for i in range(12)]
        facts=[dict(id='f'+str(i),kind='item-config',holder='Ari',topic='backend',attribute='language',
            value=value,value_span=value,natural_statement='My backend language is '+value+'.',
            grounding_quote='backend language') for i,value in enumerate(('Node.js','PHP'))]
        return sk,facts

    def case(self,sk,facts,operation='history'):
        models=s.seed_models(mock=True)
        c=s.make_case(sk,facts,operation,'observer',0,31);c.update(mock=True,contract=s.CONTRACT)
        return s.write_question(c,models)

    def native(self,sk,f,tid):
        f.update(native=True,native_tid=str(tid),native_span=f['natural_statement'])
        sk['turns'][tid]['text']=f['natural_statement']+' Keep the API stable.'

    def test_all_native_hosts_protected_from_facts_and_background(self):
        sk,facts=self.fixture();self.native(sk,facts[0],0)
        extra=dict(facts[0],id='f2',attribute='deployment',value='staging',natural_statement='My backend deployment is staging.')
        self.native(sk,extra,4);facts.append(extra)
        c=self.case(sk,facts);rendered=s.render_case(c,s.seed_models(mock=True))
        self.assertTrue(rendered['render_ok'])
        for tid in ('0','4'):
            t=next(t for t in rendered['turns'] if t['tid']==tid)
            self.assertFalse(t['edit_requested']);self.assertEqual(t['edit_kind'],'none')
            self.assertEqual(t['text'],sk['turns'][int(tid)]['text'])
            self.assertTrue(all(next(f for f in facts if f['id']==fid).get('native') for fid in t['fact_ids']))
        # The scheduler also defends against direct callers assigning an inserted fact.
        with self.assertRaisesRegex(ValueError,'native evidence'):
            s.schedule_rewrites(copy.deepcopy(sk['turns']),[[i] for i in range(12)],['pos']*12,
                {0:['f1']},s.random.Random(0),facts)

    def test_stale_plan_cannot_rewrite_native_evidence(self):
        sk,facts=self.fixture();self.native(sk,facts[0],0);c=self.case(sk,facts)
        t=next(t for t in c['turns'] if t['tid']=='0');t['edit_requested']=True
        rendered=s.render_case(c,s.seed_models(mock=True))
        self.assertFalse(rendered['render_ok'])
        self.assertEqual(rendered['turns'][0]['text'],sk['turns'][0]['text'])
        self.assertEqual(rendered['turns'][0]['edit_kind'],'none')

    def test_update_pair_two_families_accept_real_replacement(self):
        sk,facts=self.fixture();models=s.seed_models(mock=True);audit={}
        s.check_update_pair(facts,'backend language discussion',models,audit)
        self.assertEqual([c['family'] for c in audit['update_pair_checks']],[2,0])
        self.assertEqual(models.calls['family2:seed_update_pair'],1)
        self.assertEqual(models.calls['family0:seed_update_pair'],1)
        self.assertIn('REPLACES',s.UPDATE_PAIR_PROMPT)
        self.assertIn('actual statements',s.UPDATE_PAIR_PROMPT)

    def test_update_pair_rejects_vague_refinement_body_part_and_family_disagreement(self):
        sk,facts=self.fixture()
        for old,new,reason in (('make jabber better','add display pictures','refinement of vague goal'),
                ('Extreme pain in my foot','ankle still hurts like mad','different body part'),
                ('Node.js','PHP','not same holder in actual statement')):
            pair=copy.deepcopy(facts)
            for f,value in zip(pair,(old,new)): f.update(value=value,natural_statement=value)
            for reject_family in (0,2):
                class Judge(s.SeedMockPool):
                    def chat(pool,prompt,**kwargs):
                        p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                        if p['task']=='seed_update_pair':
                            return json.dumps(dict(valid=int(pool.family)!=reject_family,reason=reason)),'mock'
                        return super().chat(prompt,**kwargs)
                models=s.seed_models(mock=True,pools=[Judge(str(i)) for i in range(3)]);audit={}
                with self.assertRaisesRegex(ValueError,reason):s.check_update_pair(pair,'window',models,audit)
                self.assertEqual(len(audit['update_pair_checks']),2)

    def test_invalid_inserted_update_rejected_during_planning_with_reason(self):
        sk,_=self.fixture();attempts=[]
        class Reject(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_update_pair':return json.dumps(dict(valid=False,reason='elaboration is not replacement')),'mock'
                return super().chat(prompt,**kwargs)
        pool=Reject();models=s.seed_models(mock=True,pools=[pool]*3)
        with self.assertRaisesRegex(ValueError,'elaboration is not replacement'):s.propose_facts(sk,models,attempts)
        self.assertEqual(len(attempts),3)
        self.assertTrue(all('invalid update pair' in a['error'] and len(a['update_pair_checks'])==2 for a in attempts))

    def test_native_pair_also_requires_semantic_check(self):
        sk,facts=self.fixture()
        native=[]
        for f,i in zip(facts,(0,2)):
            self.native(sk,f,i)
            native.append(dict(f,tid=str(i),span=f['natural_statement'],grounding_quote=f['natural_statement']))
        class Reject(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='seed_native':return json.dumps(dict(facts=native)),'mock'
                if p['task']=='seed_update_pair':return json.dumps(dict(valid=False,reason='different specific attributes')),'mock'
                return super().chat(prompt,**kwargs)
        pool=Reject();models=s.seed_models(mock=True,pools=[pool]*3);attempts=[]
        with self.assertRaisesRegex(ValueError,'different specific attributes'):s.propose_facts(sk,models,attempts)
        self.assertEqual(models.calls['family0:seed_plan'],0)
        self.assertIn('different specific attributes',attempts[0]['error'])

    def test_wrong_person_requires_minimal_integration_and_keeps_host_point(self):
        sk,facts=self.fixture();c=self.case(sk,facts);rendered=s.render_case(c,s.seed_models(mock=True))
        wrong=next(f for f in c['facts'] if f.get('wrong_person'))
        t=next(t for t in rendered['turns'] if wrong['id'] in t['fact_ids'])
        self.assertGreaterEqual(s.register_checks(t['original_text'],t['text'])['char_similarity'],.5)
        self.assertIn('keep the API stable',t['text'])
        wholesale=dict(speaker=t['speaker'],text='True, PHP powers everything now.')
        reasons=s.render_failures(wholesale,t,[wrong],c['facts'])
        self.assertTrue(any('similarity <0.5' in r for r in reasons))
        self.assertIn("original host's own point",s.CONTENT_PROMPT)

    def test_content_judge_rejects_lost_point_even_with_high_similarity(self):
        sk,facts=self.fixture();c=self.case(sk,facts)
        class LostPoint(s.SeedMockPool):
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='render' and any(f.get('wrong_person') for f in p['messages'][0]['assigned_facts']):
                    m=p['messages'][0]
                    text=m['original_text'].replace('keep the API stable','abandon API stability')+' For my own setup I use PHP with the backend, too.'
                    self.assertGreaterEqual(s.register_checks(m['original_text'],text)['char_similarity'],.5)
                    return json.dumps(dict(messages=[dict(id=m['id'],speaker=m['speaker'],text=text)])),'mock'
                if p['task']=='seed_content_check' and 'abandon API stability' in p['text']:
                    self.assertIn('keep the API stable',p['message']['original_text'])
                    return json.dumps(dict(coherence=2,contradiction=True,same_voice=True,natural_value=True,
                        reason="lost host's own point about API stability")),'mock'
                return super().chat(prompt,**kwargs)
        pool=LostPoint();rendered=s.render_case(c,s.seed_models(mock=True,pools=[pool]*3))
        self.assertFalse(rendered['render_ok'])
        wrong=next(t for t in rendered['turns'] if any(f.get('wrong_person') and f['id'] in t['fact_ids'] for f in rendered['facts']))
        self.assertIn("lost host's own point",str(wrong['render_failures']))

    def test_relay_uses_related_host_and_rejects_oh_and(self):
        sk,facts=self.fixture()
        for i in range(6,12):sk['turns'][i]['text']='The weather forecast predicts sunshine for tomorrow. '*4
        c=self.case(sk,facts);relay=next(f for f in c['facts'] if f.get('equivalent_to'))
        self.assertLess(relay['source']['position'],6)
        t=next(t for t in c['turns'] if relay['id'] in t['fact_ids'])
        row=dict(speaker=t['speaker'],text=t['original_text']+" Oh, and Ari mentioned PHP for the backend.")
        self.assertTrue(any('disconnected relay' in r for r in s.render_failures(row,t,[relay],c['facts'])))

    def test_topic_scores_prefer_specific_topic_and_adjacent_right_speaker(self):
        sk,facts=self.fixture();f=facts[0]
        sk['turns'][0]['text']='My toothache is getting worse and I need a dentist. '*5
        sk['turns'][1]['text']='The backend language affects the API deployment.'
        layout=s.round_layout(sk['turns'])
        self.assertGreater(s.host_topic_score(sk['turns'],layout,1,f),s.host_topic_score(sk['turns'],layout,0,f))
        for t in sk['turns']:t['text']='The toothache needs a dentist and a filling.'
        self.assertEqual(s.host_topic_score(sk['turns'],layout,4,f),0)

    def test_new_turn_fallback_is_short_related_and_accounted_through_export(self):
        sk,facts=self.fixture()
        for t in sk['turns']:t['text']='The weather is clear and the road is quiet. '*3
        # The offline answer fixture recognizes canonical statements only.
        facts[0].update(value='Python',value_span='Python',natural_statement="Earlier, Ari's backend language is Python.")
        self.native(sk,facts[0],2)
        c=self.case(sk,facts);inserts=[t for t in c['turns'] if t.get('planned_insert')]
        self.assertEqual(len(inserts),1);insert=inserts[0]
        self.assertEqual(insert['speaker'],'Ari');self.assertEqual(insert['insert_after_tid'],'2')
        i=c['turns'].index(insert);self.assertEqual(c['turns'][i-1]['tid'],'2')
        self.assertTrue(insert['edit_requested']);self.assertEqual(insert['original_text'],'')
        rendered=s.render_case(c,s.seed_models(mock=True))
        self.assertTrue(rendered['render_ok'],[(t['tid'],t.get('render_error')) for t in rendered['turns'] if not t['render_ok']])
        self.assertEqual(rendered['turns'][i]['edit_kind'],'insert')
        self.assertEqual(rendered['rewrite_schedule']['actual_insertion_rate'],1/13)
        self.assertEqual(c['rewrite_schedule']['target_insertion_rate'],1/13)
        self.assertFalse(c['rewrite_schedule']['matching_feasible'])
        self.assertTrue(rendered['rewrite_schedule']['actual_matches_schedule'])
        self.assertTrue(.4<=rendered['rewrite_schedule']['actual_edit_rate']<=.6)
        untouched,_=s.documents(rendered,untouched=True);restored,_=s.documents(rendered,restore=[i])
        self.assertNotIn('PHP',' '.join(untouched));self.assertNotIn('PHP',restored[c['layout'].index([i])])
        checked=s.verify_case(rendered,s.seed_models(mock=True));self.assertTrue(checked['accepted'],checked.get('drop_reason'))
        questions,_,_=s.build([checked]);p=next(p for p in questions[0]['candidate_provenance'] if p['was_inserted'])
        self.assertTrue(p['was_edited']);self.assertEqual(p['turns'][0]['edit_kind'],'insert')
        self.assertTrue(p['original_restoration'])
        self.assertGreater(s.provenance_table(questions)['table']['pos']['inserted_rate'],0)

    def test_no_related_content_rejects_and_long_insert_rejects(self):
        sk,facts=self.fixture()
        for t in sk['turns']:t['text']='The toothache needs a dentist and a filling. '*3
        with self.assertRaisesRegex(ValueError,'no related'):self.case(sk,facts)
        spec=dict(speaker='Ari',original_text='',edit_mode='insert')
        self.assertTrue(any('must be short' in r for r in s.render_failures(
            dict(speaker='Ari',text='Node.js '+'word '*100),spec,facts[:1],facts)))

    def test_email_blocks_are_not_hosts_or_renderer_input(self):
        sk,facts=self.fixture();sk['source']='enron';sk['carrier']='record'
        signature='Ari Silva\nCadwalader, Wickersham & Taft\n100 Maiden Lane\nTel.: [PHONE]'
        sk['turns'][0]['text']=signature
        for t in sk['turns'][1:]:t['text']='From: Ari\nSubject: backend\n\n'+t['text']+'\n\n'+signature
        c=self.case(sk,facts);self.assertFalse(c['turns'][0]['edit_requested'])
        self.assertNotIn('0',[f['source']['tid'] for f in c['facts']])
        class Capture(s.SeedMockPool):
            def __init__(pool):super().__init__();pool.requests=[]
            def chat(pool,prompt,**kwargs):
                p=json.loads(prompt.split('\nPAYLOAD\n')[1])
                if p['task']=='render':pool.requests.append(p)
                return super().chat(prompt,**kwargs)
        pool=Capture();rendered=s.render_case(c,s.seed_models(mock=True,pools=[pool]*3))
        self.assertTrue(rendered['render_ok'])
        for p in pool.requests:
            texts=[p['messages'][0]['original_text']]+[t['text'] for t in p['previous_turns']+p['next_turns']]
            self.assertNotIn('Cadwalader',' '.join(texts));self.assertNotIn('From:',' '.join(texts))
        for t in rendered['turns']:
            if t['edit_kind']=='rewrite':self.assertTrue(t['text'].endswith(signature))
        # A one-line signature and a body repeated in Subject must also work.
        self.assertEqual(s.email_body('Ari Silva Harbor Partners 100 Elm Lane Tel.: [PHONE]'),'')
        text='Subject: backend\n\nbackend\n\nBest,\nAri Silva'
        begin,end=s.email_body_span(text)
        self.assertEqual(text[:begin]+'changed'+text[end:],'Subject: backend\n\nchanged\n\nBest,\nAri Silva')

    def test_foreign_values_already_in_original_window_are_allowed(self):
        _,facts=self.fixture();message=dict(speaker='Ari',original_text='My backend language is Node.js today.',
            window_original_text='My backend language is Node.js today. Kiran uses PHP for the backend.')
        row=dict(speaker='Ari',text='My backend language is Node.js; Kiran uses PHP.')
        self.assertFalse(any('foreign value leak' in r for r in s.render_failures(row,message,facts[:1],facts)))
        del message['window_original_text']
        self.assertTrue(any('foreign value leak' in r for r in s.render_failures(row,message,facts[:1],facts)))


if __name__=='__main__': unittest.main()

