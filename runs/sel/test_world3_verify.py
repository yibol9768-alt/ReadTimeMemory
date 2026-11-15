"""Offline message-level verification, evidence binding and typed quota regressions."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import random
import tempfile
import unittest

import world3 as v
from world3_adapter import groups_for, mock_pool


class Judge(v.MockPool):
    def __init__(self, family, reject=(), spans=None):
        super().__init__(str(family))
        self.reject=set(reject)
        self.spans=spans
        self.seen=[]

    def chat(self, prompt, **kwargs):
        payload=json.loads(prompt.split('\nPAYLOAD\n',1)[1])
        if payload['task']=='verify':
            if len(payload['messages'])!=1:
                raise AssertionError('verification must call each judge per message')
            m=payload['messages'][0]
            self.seen.append(m['id'])
            if m['id'] in self.reject:
                return json.dumps(dict(ok=False,reason='wrong qualifier')),self.family
            if self.spans is not None:
                return json.dumps(dict(ok=True,no_extra_facts=True,
                    quotes={m['id']+':'+fid:self.spans(m,fid) for fid in m.get('anchor_sets',{})})),self.family
        return super().chat(prompt,**kwargs)


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures={}
        for revision in (2,3):
            plan=v.make_plan(2,8,seed=73,domain='software',revision=revision)
            case=next(c for c in plan['cases'] if c['qtype']=='direct' and not c['date_window'])
            plan['cases']=[case]
            plan['facts']=[f for f in plan['facts'] if f['source']['case']==case['id']]
            cls.fixtures[revision]=v.render(plan,v.ModelFamilies(mock=True),par=4)

    def verify(self, world, rejects=None, spans=None, checkpoint=None):
        pools=[Judge(i,(rejects or {}).get(i,()),spans) for i in range(3)]
        models=v.ModelFamilies(mock=True,pools=pools)
        stream=io.StringIO()
        with redirect_stdout(stream):
            result=v.verify(world,models,checkpoint)
        line=stream.getvalue().strip()
        self.assertTrue(line.startswith('VERIFY_SUMMARY '))
        self.assertEqual(json.loads(line.split(' ',1)[1]),result['verify_summary'])
        return result,models,pools

    def required(self, world):
        required=set(v.solve(world,world['cases'][0]['query'])['atom_ids'])
        return [m for m in world['rounds'] if required.intersection(m['atoms'])]

    def test_each_message_uses_both_non_rendering_families_and_resumes_independently(self):
        for revision,world in self.fixtures.items():
            with self.subTest(revision=revision),tempfile.TemporaryDirectory() as root:
                # Old block/case caches must never supply votes for this contract.
                v.write(Path(root)/('verify_'+world['cases'][0]['id']+'_block0.json'),{'votes':{}})
                verified,models,pools=self.verify(world,checkpoint=root)
                self.assertTrue(verified['cases'][0]['semantic_ok'])
                self.assertEqual(sum(models.calls.values()),2*len(world['rounds']))
                for m in world['rounds']:
                    self.assertEqual({i for i,p in enumerate(pools) if m['id'] in p.seen},
                        {(m['family']+1)%3,(m['family']+2)%3})
                before=dict(models.calls)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(verified,v.verify(verified,models,root))
                self.assertEqual(before,dict(models.calls))
                changed=copy.deepcopy(world)
                changed['rounds'][0]['text']+=' Extra spacing. '
                with redirect_stdout(io.StringIO()):v.verify(changed,models,root)
                self.assertEqual(sum(models.calls.values()),sum(before.values())+2)
                self.assertEqual(len(list(Path(root).glob('verify_m*.json'))),len(world['rounds']))

    def test_one_judge_rejecting_any_required_equivalent_quarantines_case(self):
        world=self.fixtures[2]
        required=self.required(world)
        self.assertGreaterEqual(len(required),2)  # includes the explicit restatement
        for m in required:
            with self.subTest(message=m['id']):
                verified,_,_=self.verify(world,{(m['family']+1)%3:[m['id']]})
                row=next(r for r in verified['rounds'] if r['id']==m['id'])
                self.assertEqual(row['verification'],[False,True])
                self.assertTrue(row['verify_required'])
                self.assertFalse(verified['cases'][0]['semantic_ok'])
                self.assertEqual(verified['verify_summary']['required_failed'],1)
                self.assertIn('wrong qualifier',str(row['verify_failure_reasons']))
                self.assertEqual(v.build_questions(verified),[])

    def test_failed_optional_messages_drop_and_same_type_quota_refills_or_reports_shortage(self):
        world=self.fixtures[2]
        baseline,_,_=self.verify(world)
        question=v.build_questions(baseline)[0]
        candidates=question['negative_candidates']
        meta=[m for sess in question['haystack_sessions'] for m in sess if m['role']=='user']
        kind=next(k for k,quota in v.NEG_QUOTAS.items() if len(candidates[k])>quota)
        victim=meta[candidates[kind][0]]['id']
        verified,_,_=self.verify(world,{i:[victim] for i in range(3)})
        self.assertTrue(verified['cases'][0]['semantic_ok'])
        q=v.build_questions(verified)[0]
        self.assertNotIn(victim,{m['id'] for sess in q['haystack_sessions'] for m in sess})
        selected,labels,shortage=v.sample_negatives(q['negative_candidates'],random.Random(5))
        self.assertEqual(labels.count(kind),v.NEG_QUOTAS[kind])
        self.assertFalse(shortage[kind])
        self.assertTrue(all(i in q['negative_candidates'][label] for i,label in zip(selected,labels)))
        victims=[meta[i]['id'] for i in candidates[kind]]
        verified,_,_=self.verify(world,{i:victims for i in range(3)})
        self.assertTrue(verified['cases'][0]['semantic_ok'])
        q=v.build_questions(verified)[0]
        _,labels,shortage=v.sample_negatives(q['negative_candidates'],random.Random(5))
        self.assertEqual(shortage[kind],v.NEG_QUOTAS[kind])
        self.assertNotIn(kind,labels)
        self.assertEqual(q['negative_shortfall'][kind],v.NEG_QUOTAS[kind])
        self.assertEqual(v.quota_audit(verified,[q])['negative_shortfall'][q['question_id']][kind],v.NEG_QUOTAS[kind])

    def test_competition_failure_reports_panel_shortage_without_random_substitution(self):
        world=self.fixtures[3]
        case=world['cases'][0]
        fid=next(iter(case['competition_documents']))
        atom=case['competition_documents'][fid][1][0]
        victim=next(m for m in world['rounds'] if atom in m['atoms'])
        verified,_,_=self.verify(world,{i:[victim['id']] for i in range(3)})
        self.assertTrue(verified['cases'][0]['semantic_ok'])
        self.assertEqual(verified['verify_summary']['dropped_non_required'],1)
        q=v.build_questions(verified)[0]
        self.assertEqual(q['negative_shortfall'][fid],1)
        groups,reports=groups_for(q,mock_pool(q),no_set=True,p=100000)
        self.assertEqual(groups,[])
        self.assertEqual(reports[0]['drop'],'structural_quota_shortfall')
        self.assertEqual(reports[0]['shortfall']['competition_panel'],1)

    def test_short_whitespace_normalized_spans_accept_full_message_anchors_and_qualifiers(self):
        world=copy.deepcopy(self.fixtures[3])
        for m in world['rounds']:
            if m['core']:
                # Neither the locator nor the anchors alone carries qualifiers.
                m['text']='Support\n\tlocator. '+m['text'].replace('its status was executed','it ran')
        self.assertTrue(any('it ran' in m['text'] for m in world['rounds']))
        verified,_,_=self.verify(world,spans=lambda m,fid:'Support locator.')
        self.assertTrue(verified['cases'][0]['semantic_ok'])
        for m in verified['rounds']:
            self.assertTrue(m['ok'],m['verify_failure_reasons'])
            for quotes in m['evidence_quotes'].values():self.assertEqual(quotes,['Support locator.']*2)
        # Locator reuse cannot make different versions equivalent downstream.
        bindings=v.evidence_bindings(verified)
        required=self.required(verified)[0]
        for fid in required['atoms']:
            self.assertNotIn('Support locator.',bindings[fid])
            fact=next(f for f in verified['facts'] if f['id']==fid)
            self.assertFalse(v.atom_present(v.bound_atom(verified,fact,bindings),'Support locator.'))

    def test_bad_quote_or_absent_message_anchor_fails_locally_even_if_judges_say_yes(self):
        for mode in ('absent_quote','empty_quote','missing_anchor','changed_punctuation'):
            with self.subTest(mode=mode):
                world=copy.deepcopy(self.fixtures[3])
                victim=self.required(world)[0]
                quote=lambda m,fid:m['text']
                if mode=='absent_quote':quote=lambda m,fid:'THIS IS NOT IN THE MESSAGE'
                elif mode=='empty_quote':quote=lambda m,fid:' \n\t '
                elif mode=='changed_punctuation':quote=lambda m,fid:m['text'].replace('.',',',1)
                else:victim['anchor_sets'][victim['atoms'][0]].append('missing-anchor-xyz')
                verified,_,_=self.verify(world,spans=quote)
                row=next(m for m in verified['rounds'] if m['id']==victim['id'])
                self.assertFalse(row['ok'])
                self.assertTrue(any(row['verify_failure_reasons'].values()))
                self.assertFalse(verified['cases'][0]['semantic_ok'])

    def test_render_failure_skips_judges_for_that_message_only_and_missing_carriers_fail(self):
        world=copy.deepcopy(self.fixtures[3])
        ids={m['id'] for m in self.required(world)}
        victim=next(m for m in world['rounds'] if m['id'] not in ids)
        victim['render_error']='no usable row'
        verified,models,pools=self.verify(world)
        self.assertEqual(sum(models.calls.values()),2*(len(world['rounds'])-1))
        self.assertTrue(verified['cases'][0]['semantic_ok'])
        self.assertTrue(all(victim['id'] not in p.seen for p in pools))
        self.assertIn('no usable row',str(verified['verify_summary']['failures']))
        world['rounds']=[m for m in world['rounds'] if m['id'] not in ids]
        verified,_,_=self.verify(world)
        self.assertFalse(verified['cases'][0]['semantic_ok'])
        self.assertTrue(verified['cases'][0]['verify_missing_atoms'])

    def test_cli_reverifies_old_contract_without_rerendering_and_resumes_new_contract(self):
        world=copy.deepcopy(self.fixtures[3])
        world.update(verified=True,verifier_identity=v.ModelFamilies(mock=True).identity)
        world['cases'][0]['semantic_ok']=False
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'w00002.json'
            v.write(path,world)
            argv=['verify','--mock','--out',root,'--start','2']
            with redirect_stdout(io.StringIO()):v.main(argv)
            updated=v.read(path)
            self.assertEqual(updated['verify_contract'],v.VERIFY_CONTRACT)
            self.assertTrue(updated['cases'][0]['semantic_ok'])
            self.assertEqual(sum(updated['llm_calls_last_step'].values()),2*len(world['rounds']))
            self.assertTrue(all(key.endswith(':verify') for key in updated['llm_calls_last_step']))
            with redirect_stdout(io.StringIO()):v.main(argv)
            self.assertEqual(v.read(path)['llm_calls_last_step'],{})

    def test_lost_dated_hard_negatives_report_shortage_without_aborting_build(self):
        plan=v.make_plan(2,8,seed=73,domain='software',revision=2)
        case=next(c for c in plan['cases'] if c['date_window'])
        plan['cases']=[case]
        plan['facts']=[f for f in plan['facts'] if f['source']['case']==case['id']]
        with redirect_stdout(io.StringIO()):world=v.render(plan,v.ModelFamilies(mock=True))
        baseline,_,_=self.verify(world)
        q=v.build_questions(baseline)[0]
        meta=[m for sess in q['haystack_sessions'] for m in sess if m['role']=='user']
        victims=[meta[i]['id'] for i in q['window_hard_negatives']]
        verified,_,_=self.verify(world,{i:victims for i in range(3)})
        self.assertTrue(verified['cases'][0]['semantic_ok'])
        report={}
        self.assertEqual(v.build_questions(verified,build_report=report),[])
        self.assertEqual(report['quarantine'][0]['reason'],'dated question lacks in-window hard negatives')
        self.assertIn('negative_shortfall',report['quarantine'][0])


if __name__=='__main__':unittest.main()
