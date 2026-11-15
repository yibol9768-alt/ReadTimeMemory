"""Revision-3 CPU regressions: competitive clusters, real-reader boundary, patch integration."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import world3 as v
from world3_adapter import groups_for, mock_pool, final_audit, PANEL_QUOTAS
from world3_probes import (universal_probes, learned_scores, competition_check,
                           ideal_prediction, parse_document, center_scores, DIMS)
from world3_structure import STRUCTURES, structure_hash, observed_structure, solve_relations
from world3_reader import LLMReader
import s23_world3_sources as sources


class Revision3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hasattr(cls,"worlds"): return
        cls.questions=[];cls.groups=[];cls.worlds=[];cls.reports=[]
        for i,domain in enumerate(list(v.DOMAINS)*2):
            m=v.ModelFamilies(mock=True)
            w=v.verify(v.render(v.make_plan(100+i,20,seed=73,domain=domain),m,par=4),m)
            cls.worlds.append(w)
            report={};qs=v.build_questions(w,build_report=report)
            if len(qs)!=20: raise AssertionError(report['quarantine'])
            cls.questions+=qs
            for q in qs:
                gs,rs=groups_for(q,mock_pool(q),no_set=True,p=100000)
                cls.groups+=gs;cls.reports+=rs

    def test_competitive_dimensions_and_query_ideal(self):
        self.assertEqual(len(self.groups),240)
        for g in self.groups:
            self.assertEqual(len(set(g['docs'])),16)
            self.assertEqual(__import__('collections').Counter(g['neg_src']),PANEL_QUOTAS)
            self.assertTrue(g['competition']['pass'],g['competition'])
            self.assertTrue(g['ideal_rule']['correct'],(g['qid'],g['ideal_rule']))
            self.assertTrue(all(g['negative_subtypes'][k] for k in ('vetoed','other_time','condition_or_future')))

    def test_blind_attacks_pass_on_final_groups(self):
        audit=final_audit(self.groups,self.questions,self.reports)
        print('WORLD3_240_AUDIT '+json.dumps(dict(groups=len(self.groups),
            cpu_contract_pass=audit['cpu_contract_pass'],ideal_rule_accuracy=audit['ideal_rule_accuracy'],
            checks=audit['checks'],shortcut_probes={k:r['unique_rate'] for k,r in audit['shortcut_probes'].items()})),flush=True)
        self.assertTrue(audit['cpu_contract_pass'],(audit['checks'],{k:r['unique_rate'] for k,r in audit['shortcut_probes'].items()}))
        self.assertEqual(audit['ideal_rule_accuracy'],1.)
        self.assertFalse(audit['release_ready'])
        for k,r in audit['shortcut_probes'].items():
            self.assertLessEqual(r['unique_rate'],.1625,(k,r))
        self.assertEqual(audit['shortcut_probes']['group_center']['searched_rules'],512)
        for k in ('logistic_relative','boosted_relative'):
            self.assertTrue(audit['shortcut_probes'][k]['validation']['validated'])

    def test_ideal_solver_does_not_read_answer_or_gold_metadata(self):
        q=copy.deepcopy(self.questions[0]);g=self.groups[0]
        q['answer']='corrupt';q['answer_atoms']=[];q['gold_stages']=[]
        prediction,answer=ideal_prediction(q,g['docs'],set())
        self.assertEqual(prediction,[0]);self.assertNotEqual(answer,'corrupt')
        shuffled=list(range(16));random.Random(9).shuffle(shuffled)
        prediction,_=ideal_prediction(q,[g['docs'][i] for i in shuffled],set())
        self.assertEqual(prediction,[shuffled.index(0)])

    def test_center_is_permutation_invariant_including_ties(self):
        import itertools
        g=self.groups[0];rows=[parse_document(d) for d in g['docs']]
        order=list(range(16));random.Random(8).shuffle(order)
        for dims in itertools.permutations(DIMS):
            for tie in ('all','first','min','max'):
                a=center_scores(rows,dims,tie)
                b=center_scores([rows[i] for i in order],dims,tie)
                self.assertEqual([a[i] for i in order],b)

    def test_learner_detects_leak_and_holds_out_entire_ledger(self):
        gs=[]
        for i in range(6):
            g=copy.deepcopy(self.groups[i]);g.update(ledger_id=f'L{i}',qid=f'Q{i}')
            g['docs']=['secret_signal positive']+['ordinary document neutral']*15;gs.append(g)
        scores,validation=learned_scores(gs)
        self.assertTrue(validation['validated'])
        for model,values in scores.items():
            self.assertTrue(all(max(s)==s[0] and s.count(s[0])==1 for s in values),model)
        altered=copy.deepcopy(gs);altered[0]['label']=7
        changed,_=learned_scores(altered)
        for model in scores:
            self.assertEqual(scores[model][0],changed[model][0])

    def test_native_domains_realized_hashes_and_edge_necessity(self):
        self.assertEqual(len({structure_hash(d) for d in STRUCTURES}),6)
        sets={}
        for w in self.worlds:
            native=[c for c in w['cases'] if c['query'].get('domain_program')]
            self.assertTrue(native)
            hashes={observed_structure(w,c)['hash'] for c in native};sets.setdefault(w['domain'],set()).update(hashes)
            for c in native:
                query=c['query'];solved=v.solve(w,query)
                self.assertEqual(len(solved['atom_ids']),c['expected_evidence'])
                rows=[next(f for f in w['facts'] if f['id']==fid) for fid in solved['atom_ids']]
                changed=copy.deepcopy(rows);changed[0]['links']=['missing endpoint']
                with self.assertRaisesRegex(ValueError,'edge'):
                    solve_relations(w['domain'],changed,query['domain_program'])
        observed=list(sets.values())
        for i,a in enumerate(observed):
            for b in observed[:i]:self.assertFalse(a&b,(a,b))

    def test_reader_measurements_use_actual_answers_and_no_gold_payload(self):
        q=self.questions[0];w=self.worlds[0]
        class RecordingReader:
            mock=False;identity={'test':'recorded answerer'}
            def __init__(self):self.calls=[]
            def answer(self,question,docs):
                self.calls.append((question,list(docs)))
                result=v.solve_visible(w,q['query_spec'],docs)
                return dict(answer=result.get('answer',''),abstain=not result['available'])
            def grade(self,question,reference,atoms,response):
                return dict(correct=response['answer']==reference,coverage=[s in '\n'.join(self.calls[-1][1]) for s in atoms])
        reader=RecordingReader();result=v.gates(w,[q],v.ModelFamilies(mock=True),reader=reader)[0]
        self.assertEqual(result['reader']['measurement'],'real_llm_reader')
        self.assertEqual(reader.calls[0][1],[])
        self.assertEqual(len(reader.calls),3+len(q['answer_atoms']))
        self.assertEqual(result['blind'],0.);self.assertEqual(result['suff'],1.)
        self.assertTrue(result['reader']['passed'])
        # Models remain mock, so an injected real reader cannot promote this fixture.
        self.assertEqual(result['completion'],'mock_only')
        class Wrong(RecordingReader):
            def answer(self,question,docs):
                self.calls.append((question,list(docs)));return dict(answer='wrong',abstain=False)
        wrong=v.gates(w,[q],v.ModelFamilies(mock=True),reader=Wrong())[0]
        self.assertEqual(wrong['suff'],0.);self.assertFalse(wrong['reader']['passed'])
        mock=v.gates(w,[q],v.ModelFamilies(mock=True))[0]
        self.assertTrue(mock['reader']['mock']);self.assertEqual(mock['completion'],'mock_only')

    def test_llm_answer_payload_has_no_reference(self):
        class Capture:
            mock=False;identity=['fixture']
            def ask(self,family,instructions,payload,**kwargs):
                self.payload=payload
                return dict(answer='abstain',abstain=True)
        models=Capture();reader=LLMReader(models)
        reader.answer('question',['evidence'])
        self.assertEqual(set(models.payload),{'task','question','documents'})


class PatchIntegrationTests(unittest.TestCase):
    def test_patch_applies_to_current_sources_and_routes_builders(self):
        root=Path(__file__).resolve().parent
        hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ('s23.py','preflight_s23.py')}
        text=(root/'s23_v3_patch.md').read_text().split('```diff\n',1)[1].split('```',1)[0]
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d)
            for p in hashes:(tmp/p).write_bytes((root/p).read_bytes())
            (tmp/'change.patch').write_text(text)
            subprocess.run(['patch','-p1','-i',str(tmp/'change.patch')],cwd=tmp,capture_output=True,text=True,check=True)
            for p in hashes:compile((tmp/p).read_text(),p,'exec')
            spec=importlib.util.spec_from_file_location('s23_patched_fixture',tmp/'s23.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with patch.dict(os.environ,{k:'' for k in sources.ENV},clear=False):
                self.assertEqual(module.w3.stage1(),'tr_w20.jsonl')
                calls=[]
                with patch.object(module,'flops',side_effect=lambda *a: calls.append(a) or 1):
                    self.assertEqual(module.s1_flops('yi',0),2)
                self.assertEqual(calls[0][0],'tr_w20.jsonl')
            # Every actual build entry point calls the separate-stage reservation helper.
            for name in ('build','build_all','prepare_budget'):
                import inspect
                self.assertIn('w3.reserved(',inspect.getsource(getattr(module,name)))
            from contextlib import ExitStack
            from types import SimpleNamespace
            def row(qid):
                return json.dumps(dict(qid=qid, query='query', docs=[qid], label=0, teacher=[1]))+'\n'
            def reserve(n2,n3,seed):
                return [row('syn2_'+str(i)) for i in range(n2)], [row('syn3_'+str(i)) for i in range(n3)]
            args=SimpleNamespace(arm='yi',fold='beam',seed=0,r=0.,cmd='plan',all_data=False)
            with ExitStack() as stack:
                stack.enter_context(patch.object(module.w3,'enabled',return_value=True))
                routed=stack.enter_context(patch.object(module.w3,'reserved',side_effect=reserve))
                stack.enter_context(patch.object(module,'wd',return_value=str(tmp/'build')))
                stack.enter_context(patch.object(module,'PER_STAGE',4))
                stack.enter_context(patch.object(module,'source',side_effect=lambda kind,b,*a,**k:kind+'_'+b))
                stack.enter_context(patch.object(module,'teacher_blocks',side_effect=lambda path,n2,n3,seed:([row(path+'_'+str(i)) for i in range(n2)],[])))
                stack.enter_context(patch.object(module,'take',return_value=[]))
                p2,p3=module.build(args)
                self.assertEqual(routed.call_args.args,(4,4,0))
                self.assertTrue(any(json.loads(r)['qid'].startswith('syn2_') for r in Path(p2).read_text().splitlines()))
                self.assertTrue(all(json.loads(r)['qid'].startswith('syn3_') for r in Path(p3).read_text().splitlines()))
                args.all_data=True;args.s2_groups=module.S2_LEVELS[0];args.r,args.s3_groups=module.S3_CELLS[0]
                stack.enter_context(patch.object(module,'validate_train_files',return_value=None))
                stack.enter_context(patch.object(module,'all_benchmark_rows',return_value=([],[],[],[])))
                module.build_all(args)
                self.assertEqual(routed.call_args.args,(args.s2_groups//2,args.s3_groups-int(args.s3_groups*args.r)//5*5,0))
                args.cmd='budget';args.frac=1.;args.teacher='rr8';args.teacher_config=''
                stack.enter_context(patch.object(module,'budget_dir',return_value=str(tmp/'budget')))
                stack.enter_context(patch.object(module,'_question_ids',return_value=set()))
                stack.enter_context(patch.object(module,'md5',return_value='fixture_md5'))
                stack.enter_context(patch.object(module,'choice_file',return_value='choice.json'))
                stack.enter_context(patch.object(module,'lines',side_effect=lambda path:[row('real').replace('["real"]','["'+path+'"]')]))
                stack.enter_context(patch.object(module.w3,'provenance_paths',return_value=['s1.jsonl','s2.jsonl','s3.jsonl']))
                sample=dict(bench={b:dict(qids=['real']) for b in module.BENCH},qid_md5='fixture')
                result=module.prepare_budget(args,sample)
                self.assertEqual(routed.call_args.args,(4000,4000,0))
                self.assertEqual((len(result['syn2']),len(result['syn3'])),(4000,4000))
        for p,h in hashes.items():self.assertEqual(hashlib.sha256((root/p).read_bytes()).hexdigest(),h)

    def test_reserved_sources_default_validation_and_no_repetition(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths=[]
            for stage in range(3):
                path=root/f's{stage}.jsonl';paths.append(path)
                path.write_text(''.join(json.dumps(dict(qid=f'{stage}_{i}',query='q',docs=[str(i)],ledger_id=f'ledger{stage}',split='train',mock=True))+'\n' for i in range(5)))
            env={k:str(p) for k,p in zip(sources.ENV,paths)};env['S23_WORLD3_ALLOW_MOCK']='1'
            with patch.object(sources,'ROOT',root),patch.dict(os.environ,env):
                with patch.dict(os.environ, {'S23_WORLD3_ALLOW_MOCK':'0'}):
                    with self.assertRaisesRegex(ValueError,'mock'): sources.stage1_count()
                self.assertEqual(sources.stage1_count(),5)
                self.assertEqual(sources.container_stage1(),'/w/s0.jsonl')
                a,b=sources.reserved(4,3,8);self.assertEqual((len(a),len(b)),(4,3))
                self.assertFalse({json.loads(r)['ledger_id'] for r in a}&{json.loads(r)['ledger_id'] for r in b})
                self.assertIn(sources.tag(),sources.checkpoint(0))
                with self.assertRaisesRegex(ValueError,'insufficient'):sources.reserved(6,3,8)
                rows=[json.loads(s) for s in paths[2].read_text().splitlines()]
                for r in rows:r['ledger_id']='ledger1'
                paths[2].write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaisesRegex(ValueError,'overlap'):sources.reserved(3,3,8)
            with patch.dict(os.environ,{sources.ENV[0]:str(paths[0]),sources.ENV[1]:'',sources.ENV[2]:''}):
                with self.assertRaisesRegex(ValueError,'together'):sources.stage1()


if __name__=='__main__':unittest.main()
