"""Small real-reader interface fixtures; no network or inference server."""
import copy
import tempfile
import unittest
import world3 as v
from world3_reader import LLMReader


class ReaderInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan=v.make_plan(2020,20,seed=42,domain='software')
        plan['cases']=plan['cases'][:1]
        plan['facts']=[f for f in plan['facts'] if f['source']['case']==plan['cases'][0]['id']]
        models=v.ModelFamilies(mock=True)
        cls.world=v.verify(v.render(plan,models),models)
        cls.q=v.build_questions(cls.world)[0]

    def test_full_evidence_deletion_raw_scores_and_resume(self):
        q=self.q;w=self.world
        class Recording:
            mock=False;identity={'fixture':'recording'}
            def __init__(self):self.calls=[]
            def answer(self,question,docs):
                self.calls.append((question,list(docs)))
                out=v.solve_visible(w,q['query_spec'],docs)
                return dict(answer=out.get('answer',''),abstain=not out['available'])
            def grade(self,question,reference,atoms,response):
                return dict(correct=response['answer']==reference,coverage=[a in '\n'.join(self.calls[-1][1]) for a in atoms])
        reader=Recording();models=v.ModelFamilies(mock=True)
        with tempfile.TemporaryDirectory() as d:
            gate=v.gates(w,[q],models,d,reader)[0]
            before=len(reader.calls);self.assertGreater(before,3)
            self.assertEqual(v.gates(w,[q],models,d,reader)[0],gate)
            self.assertEqual(len(reader.calls),before)
            self.assertEqual(reader.calls[0][1],[])
            self.assertEqual(len(reader.calls[1][1]),len(gate['reader']['evidence_rounds']))
            self.assertLess(len(reader.calls[1][1]),len(v.visible_documents(q)))
            self.assertEqual((gate['blind'],gate['suff']),(0.,1.))
            self.assertTrue(gate['reader']['passed'])
            self.assertEqual(gate['completion'],'mock_only')
            reader.identity={'fixture':'changed'}
            v.gates(w,[q],models,d,reader)
            self.assertGreater(len(reader.calls),before)

    def test_missing_reader_and_mock_cannot_complete_release(self):
        mock=v.gates(self.world,[self.q],v.ModelFamilies(mock=True))[0]
        self.assertTrue(mock['reader']['mock']);self.assertEqual(mock['reader']['measurement'],'mock_oracle_only')
        class Failure:
            mock=False;identity={'fixture':'failure'}
            def answer(self,*args):raise ValueError('reader unavailable')
        bad=v.gates(self.world,[self.q],v.ModelFamilies(mock=True),reader=Failure())[0]
        self.assertIsNone(bad['blind']);self.assertIsNone(bad['suff'])
        self.assertFalse(bad['reader']['complete']);self.assertFalse(bad['order'])


class NativeQuestionTests(unittest.TestCase):
    def test_question_exposes_every_program_selector(self):
        for i,domain in enumerate(v.DOMAINS):
            world=v.make_plan(3000+i,20,seed=91,domain=domain)
            native=[c for c in world['cases'] if c['query'].get('domain_program')]
            self.assertTrue(native)
            for c in native:
                for r in c['query']['requests']:
                    self.assertIn(f"{r['holder']}'s {r['scope']} {r['attr']} for {r['entity']}",c['question'])
                self.assertIn(c['query']['as_of'],c['question'])
                for k,value in c['query']['context'].items():self.assertIn(f'{k}={value}',c['question'])


if __name__=='__main__':unittest.main()
