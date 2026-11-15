"""Fourth audit blockers: NL gold equality, actual edge traversal, types and routing."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch

import world3 as v
import test_world3_v3 as fixtures
from world3_semantics import parse_question, check_alignment
from world3_probes import ideal_prediction, parse_document
from world3_structure import RELATION_TYPES
import natural_stats as ns
import s23_world3_sources as sources


class SemanticAndGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Revision3Tests.setUpClass()
        cls.fixture=fixtures.Revision3Tests

    def test_natural_question_alone_selects_exact_gold_and_negatives_never_match(self):
        byqid={q['question_id']:q for q in self.fixture.questions}
        for g in self.fixture.groups:
            q=copy.deepcopy(byqid[g['qid']]);parsed=parse_question(q['question'])
            self.assertEqual(parsed['requests'],q['query_spec']['requests'])
            q['query_spec']={'requests':[]};q['answer']='poison';q['answer_atoms']=[];q['gold_stages']=[]
            pred,answer=ideal_prediction(q,g['docs'],set())
            self.assertEqual(pred,[0]);self.assertNotEqual(answer,'poison')
            # There is no hidden query_spec fallback, even for unsupported questions.
            q['question']='What is the square root of two?'
            with self.assertRaises(ValueError):ideal_prediction(q,g['docs'],set())

    def test_parser_rejects_contradictory_headline_and_omitted_attribute_limit(self):
        w=self.fixture.worlds[0]
        c=next(c for c in w['cases'] if not c['query'].get('domain_program') and c['qtype']=='direct')
        question=c['question'].replace('What are ', 'Summarize all effective outcomes for ',1)
        self.assertFalse(check_alignment(question,c['query'],w['facts'])['ok'])
        q=copy.deepcopy(c['query']);q['operation']='aggregation'
        text=v.question_text(dict(c,query=q,explicit_as_of=True))
        self.assertIn('only these attributes:',text)
        self.assertTrue(check_alignment(text,q,w['facts'])['ok'])
        text=text.replace('for only these attributes: ','for all attributes including ')
        self.assertFalse(check_alignment(text,q,w['facts'])['ok'])

    def test_each_native_question_has_two_typed_edges_and_every_edge_changes_answer(self):
        for w in self.fixture.worlds:
            native=[c for c in w['cases'] if c['query'].get('domain_program')]
            self.assertGreaterEqual(len(native)/len(w['cases']),.4)
            self.assertTrue(any(c['qtype']=='multihop' for c in native))
            for c in native:
                q=parse_question(c['question']);answer=v.solve(w,q)
                self.assertEqual(len(q['requests']),1)
                root=q['requests'][0]['entity']
                for atom in answer['atom_ids']:
                    entity=next(f['entity'] for f in w['facts'] if f['id']==atom)
                    if entity!=root:self.assertNotIn(entity,c['question'])
                rows={f['id']:f for f in w['facts'] if f['id'] in answer['atom_ids']}
                edges=[(f['id'],link) for f in rows.values() for link in f.get('links',[])]
                self.assertGreaterEqual(len(edges),2)
                self.assertGreaterEqual(len({rows[fid]['attr'] for fid,link in edges}),2)
                # Remove one actual visible edge, then re-solve. A changed computed
                # answer and reachable atom set, rather than a topology assertion.
                for fid,link in edges:
                    changed={'facts':[dict(f,links=[x for x in f.get('links',[]) if x!=link]) if f['id']==fid else f for f in w['facts']]}
                    result=v.solve(changed,q)
                    self.assertNotEqual(result['answer'],answer['answer'],(w['domain'],c['id'],fid,link))
                    self.assertNotEqual(set(result['atom_ids']),set(answer['atom_ids']))

    def test_cluster_combinations_sizes_and_all_attribute_types(self):
        sizes=set();patterns=set();cardinalities=set()
        for w in self.fixture.worlds:
            self.assertTrue(v.validate_types(w));self.assertTrue(v.validate_ledger(w['facts']))
            for c in w['cases']:
                byid={f['id']:f for f in w['facts']}
                for fid,docs in c['competition_documents'].items():
                    sizes.update(map(len,docs));rows=[byid[d[0]] for d in docs]
                    # Short carriers may show a tagged old/inactive contrast
                    # instead of its current anchor. Check the original mirror
                    # chain invariant against ALL sixteen actual current anchors.
                    anchors=[byid[x] for x in c.get('competition_anchors',{}).get(fid,[d[0] for d in docs])]
                    self.assertEqual(len(anchors),len(rows))
                    for anchor,doc,tags in zip(anchors,docs,c['panel_subtypes'][fid]):
                        if doc[0]!=anchor['id']:
                            self.assertIn(c['carrier'],('group','dyad','assistant'))
                            self.assertEqual(len(doc),1)
                            self.assertEqual(len(tags),1)
                            contrast=byid[doc[0]]
                            self.assertEqual(v.slot(contrast),v.slot(anchor))
                            if tags[0]=='other_time':
                                self.assertLess(contrast['event_time'],anchor['event_time'])
                            elif tags[0]=='vetoed':
                                self.assertNotIn(contrast['status'],v.EFFECTIVE)
                            else:
                                self.assertEqual(tags[0],'condition_or_future')
                                self.assertTrue(contrast['event_time']>c['query']['as_of'] or
                                    contrast['condition']!=c['query']['context'] or contrast['status'] not in v.EFFECTIVE)
                    original=anchors[0]
                    if not original.get('machine') and original.get('supersedes'):
                        expected=byid[original['supersedes']]['status']
                        for mirror in anchors[1:]:
                            self.assertIsNotNone(mirror['supersedes'])
                            self.assertEqual(byid[mirror['supersedes']]['status'],expected)
                    cards=tuple(len({f[k] for f in rows}) for k in ('entity','holder','scope','attr'))
                    cardinalities.add(cards)
                    patterns.add(tuple(tuple(f[k] for k in ('holder','scope','attr')) for f in rows))
        self.assertTrue({1,2,3,4}<=sizes)
        self.assertGreater(len(cardinalities),10)
        self.assertTrue(any(c!=(2,2,2,2) for c in cardinalities))
        w=copy.deepcopy(self.fixture.worlds[0]);f=next(f for f in w['facts'] if f.get('machine'))
        f['attr']=w['ontology']['availability'];f['value']=w['ontology']['unavailable']
        with self.assertRaisesRegex(ValueError,'machine'):v.validate_types(w)
        w=copy.deepcopy(self.fixture.worlds[0]);w['facts'][0]['value']='impossible_value'
        with self.assertRaisesRegex(ValueError,'value'):v.validate_types(w)


# Apply the unified diff directly to strings, without creating or modifying s23.py.
def apply_memory_patch(original,patch_text,name):
    section=patch_text.split('--- a/'+name+'\n+++ b/'+name+'\n',1)[1].split('--- a/',1)[0]
    lines=original.splitlines(True);out=[];cursor=0
    for m in re.finditer(r'^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*\n(.*?)(?=^@@|\Z)',section,re.M|re.S):
        start=int(m[1])-1;out.extend(lines[cursor:start]);cursor=start
        for line in m[2].splitlines(True):
            if line.startswith((' ','-')):
                if cursor>=len(lines) or lines[cursor]!=line[1:]:raise AssertionError('patch context mismatch')
                cursor+=1
            if line.startswith((' ','+')):out.append(line[1:])
    out.extend(lines[cursor:]);return ''.join(out)


class StartRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent
        original=(root/'s23.py').read_text();diff=(root/'s23_v3_patch.md').read_text().split('```diff\n')[1].split('```')[0]
        cls.before=hashlib.sha256((root/'s23.py').read_bytes()).hexdigest()
        cls.module=types.ModuleType('s23_memory_fixture');cls.module.__file__=str(root/'s23.py')
        exec(compile(apply_memory_patch(original,diff,'s23.py'),cls.module.__file__,'exec'),cls.module.__dict__)

    def test_all_start_and_run_seed0_seed7_enabled_and_disabled(self):
        m=self.module
        class StopBeforeTraining(Exception):pass
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);plan=dict(s2='s2',s3='s3',s2_md5='hash',s3_md5='hash',teacher_overlap=0,group_exposures={},s2_rows=32,s3_rows=32)
            (p/'plan.json').write_text(json.dumps(plan))
            for enabled in (False,True):
                for seed in (0,7):
                    for arm in ('yi','bing','jia'):
                        a=types.SimpleNamespace(seed=seed,arm=arm,gpus='fixture',smoke_steps=0)
                        old=(f'/w/ckpt/s23_s1_yi_s{seed}' if seed else '/w/ckpt/d3_e')
                        expected='/w/ckpt/world3_fixture_s'+str(seed) if enabled and arm=='yi' else old if arm=='yi' else f'/w/ckpt/s23_s1_bing_s{seed}' if arm=='bing' else m.A0
                        with patch.object(m.w3,'enabled',return_value=enabled),patch.object(m.w3,'checkpoint',side_effect=lambda s:f'ckpt/world3_fixture_s{s}'),patch.object(m,'wd',return_value=d),patch.object(m,'stage23_gpus',return_value=['fixture']),patch.object(m,'md5',return_value='hash'),patch.object(m,'lines',return_value=[]),patch.object(m,'assert_teacher_disjoint'),patch.object(m,'torchrun',side_effect=StopBeforeTraining) as launch:
                            if arm=='yi':self.assertEqual(m.all_start(a),expected)
                            with self.assertRaises(StopBeforeTraining):m.run(a)
                            args=launch.call_args.args[2]
                            self.assertEqual(args[args.index('--model')+1],expected)
        self.assertEqual(hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),self.before)

    def test_s1_producer_and_run_consumer_use_same_checkpoint(self):
        from contextlib import ExitStack,redirect_stdout
        import io
        m=self.module
        fake=types.ModuleType('preflight_s23');fake.checkpoint_recipe=lambda *a,**k:None
        fake.training_rows=lambda path,n:n
        for seed in (0,7):
            a=types.SimpleNamespace(arm='yi',seed=seed,gpus='fixture',smoke_steps=0)
            with ExitStack() as stack:
                stack.enter_context(patch.dict(__import__('sys').modules,{'preflight_s23':fake}))
                stack.enter_context(patch.object(m,'preflight'))
                stack.enter_context(patch.object(m.os,'makedirs'))
                stack.enter_context(patch.object(m,'stage23_gpus',return_value=['a','b','c','d']))
                launch=stack.enter_context(patch.object(m,'torchrun'))
                stack.enter_context(patch.object(m.w3,'enabled',return_value=True))
                stack.enter_context(patch.object(m.w3,'checkpoint',side_effect=lambda s:f'ckpt/world3_fixture_s{s}'))
                stack.enter_context(patch.object(m.w3,'stage1',return_value='new_s1.jsonl'))
                stack.enter_context(patch.object(m.w3,'container_stage1',return_value='/w/new_s1.jsonl'))
                stack.enter_context(patch.object(m.w3,'stage1_count',return_value=64))
                stack.enter_context(redirect_stdout(io.StringIO()))
                m.s1(a)
                first,second=[c.args for c in launch.call_args_list]
                self.assertEqual('/w/'+second[1],m.all_start(a))
                self.assertEqual(first[1],second[1]+'_w20')
                self.assertEqual(first[2][first[2].index('--data')+1],'/w/new_s1.jsonl')
                self.assertEqual(second[2][second[2].index('--model')+1],'/w/'+first[1])

    def test_all_entrypoints_route_through_all_start_and_s1_writes_same_checkpoint(self):
        m=self.module
        import inspect
        for name in ('all_grid','all_run','budget_stage1_actual','lobo_run','budget_run'):
            if hasattr(m,name):self.assertIn('all_start(',inspect.getsource(getattr(m,name)),name)
        # Every remaining legacy yi checkpoint literal belongs to the disabled
        # fallback in run/all_start. Source routing produces and consumes one path.
        source=Path(m.__file__).read_text()
        self.assertIn('def s1(',source)
        import ast
        tree=ast.parse(apply_memory_patch(source,Path('s23_v3_patch.md').read_text().split('```diff\n')[1].split('```')[0],'s23.py'))
        s1=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='s1')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='checkpoint' for n in ast.walk(s1)))


class NaturalnessTests(unittest.TestCase):
    def test_forest_detects_leak_and_holds_out_complete_ledger(self):
        import numpy as np
        from world3_extra_probes import forest_scores
        from world3_probes import rate
        groups=[dict(ledger_id=str(i//10),label=0,docs=['leak']+['ordinary']*15) for i in range(40)]
        xs=[np.array([[1.]]+[[0.]]*15) for g in groups]
        scores=forest_scores(groups,xs)
        self.assertEqual(rate(groups,scores)['unique_rate'],1.)
        changed=[dict(g,label=7) if g['ledger_id']=='0' else g for g in groups]
        self.assertEqual(scores[:10],forest_scores(changed,xs)[:10])

    def test_real_file_contract_deduplicates_and_is_label_blind(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool=root/'pool_beam';pool.mkdir()
            q=dict(question_id='sample',haystack_sessions=[[dict(role='user',speaker='Ari',content=f'We discussed Orion in meeting {i}. '+('notes '*i)) for i in range(20)]],haystack_dates=['January-20-2025'])
            qfile=root/'q_train_beam.json';qfile.write_text(json.dumps([q]))
            (pool/'sample.json').write_text(json.dumps(dict(fused=[0,0]+list(range(20)))))
            a,source=ns.pool_groups([qfile],[pool]);self.assertEqual(len(a[0]['docs']),16)
            self.assertEqual(source['missing'],[])
            q['answer']='poison';q['query_spec']={};q['gold_stages']=['bad'];qfile.write_text(json.dumps([q]))
            b,_=ns.pool_groups([qfile],[pool]);self.assertEqual(a,b)
            self.assertEqual(ns.summarize(a)['same_speaker_pair_share'],1.)
            report=ns.compare(a,ns.mock_groups(n=3),mock=True)
            self.assertFalse(report['naturalness_pass']);self.assertFalse(report['real_baseline_measured'])
            self.assertIn('非真实测量',ns.markdown(report))
            with self.assertRaisesRegex(ValueError,'missing'):ns.pool_groups([qfile],[])

class VisibleEdgeTests(unittest.TestCase):
    def test_visible_edge_text_controls_discovery_in_every_domain(self):
        for i,domain in enumerate(v.DOMAINS):
            # Test the complete text reader boundary without manufacturing the
            # contrast as hidden query metadata. Panels are covered separately.
            with patch.object(v,'install_competition_panels'):
                w=v.make_plan(9000+i,20,seed=19,domain=domain)
            c=next(c for c in w['cases'] if c['qtype']=='multihop')
            w['cases']=[c];w['facts']=[f for f in w['facts'] if f['source']['case']==c['id']]
            models=v.ModelFamilies(mock=True);w=v.verify(v.render(w,models),models)
            q=v.build_questions(w)[0]
            solved=v.solve(w,parse_question(q['question']))
            rows=[f for f in w['facts'] if f['id'] in solved['atom_ids']]
            edge_rows=[f for f in rows if f.get('links')]
            self.assertGreaterEqual(len({f['attr'] for f in edge_rows}),2)
            for row in edge_rows:
                for link in row['links']:
                    changed=copy.deepcopy(q)
                    replacement=v.core(dict(row,links=[x for x in row['links'] if x!=link]))
                    for session in changed['haystack_sessions']:
                        for turn in session:
                            turn['content']=turn['content'].replace(row['statement'],replacement)
                    # Poison all structured query metadata: the answer must come
                    # from the question and modified visible edge text alone.
                    changed['query_spec']={};changed['answer_atoms']=[]
                    _,answer=ideal_prediction(changed,[],set())
                    self.assertNotEqual(answer,solved['answer'],(domain,row['id'],link))


class DomainOperandTests(unittest.TestCase):
    def test_restaurant_uses_dish_demand_and_linked_inventory(self):
        from world3_structure import solve_relations
        rows=[dict(entity='dish',attr='dish_ingredient_units',value='2',links=['ingredient']),
              dict(entity='ingredient',attr='ingredient_supplier_units',value='4',links=['supplier']),
              dict(entity='supplier',attr='supplier_delivery_units',value='6',links=[])]
        answer=solve_relations('restaurant',rows,{})
        self.assertIn('menu demand=2; supplier inventory=10; portions=5',answer)


if __name__=='__main__':unittest.main()
