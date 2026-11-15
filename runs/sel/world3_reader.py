"""Measured reader gates. The answerer never receives references, atoms or gold IDs."""
import re
from world3_style import atom_present


def normalize(answer):
    return re.sub(r'\s+', ' ', answer).strip().casefold()


class LLMReader:
    def __init__(self, models, family=0):
        self.models, self.family = models, family
        self.mock = models.mock
        self.identity = dict(models=models.identity, family=family, protocol='reader-v1')

    def answer(self, question, documents):
        return self.models.ask(self.family,
            'Answer the question using ONLY the supplied documents. Without sufficient evidence, abstain. '
            'Return JSON {answer:string,abstain:bool}. Do not guess facts about private conversations.',
            dict(task='reader', question=question, documents=documents))

    def grade(self, question, reference, atoms, response):
        # A separate call/family scores a frozen answer. References never go to answer().
        if response['abstain']:
            return dict(correct=False, coverage=[False]*len(atoms))
        return self.models.ask(self.family+1,
            'Grade this frozen response semantically against the reference and required facts. '
            'Return JSON {correct:bool,coverage:[bool]}. coverage has one entry per required fact. '
            'Mark a fact only if the answer explicitly reports it correctly. correct requires every fact '
            'and the computed conclusion, without contradictions. Equivalent wording is allowed.',
            dict(task='reader_grade', question=question, reference=reference, facts=atoms, response=response['answer']),
            validator=lambda x: isinstance(x.get('correct'),bool) and isinstance(x.get('coverage'),list)
                and len(x['coverage'])==len(atoms) and all(isinstance(v,bool) for v in x['coverage']))


def measure_reader(w, q, docs, necessity, reader):
    from world3 import solve_visible
    atoms = [a['statement'] for a in q['answer_atoms']]
    removed = sorted({i for a in necessity['atoms'].values() for i in a['removed_rounds']})
    evidence_rounds = sorted(set(removed) | set(q.get('support_rounds', [])))
    # Sufficiency is measured with the complete gold evidence set, including
    # equivalent/context carriers. Whole-haystack redundancy is checked separately
    # by necessity_check. Avoid silently exceeding the answerer's context window.
    conditions = {'blind':[], 'full':[docs[i] for i in evidence_rounds],
                  'delete_all':[docs[i] for i in evidence_rounds if i not in removed]}
    for fid,row in necessity['atoms'].items():
        conditions['delete:'+fid] = [docs[i] for i in evidence_rounds if i not in set(row['removed_rounds'])]
    trials = {}
    for name,evidence in conditions.items():
        try:
            if reader.mock:
                # Explicit oracle mock tests the interface only; it is not a measurement.
                from world3_semantics import parse_question
                specification=parse_question(q['question']) if q.get('revision',2)>=3 else q['query_spec']
                observed=solve_visible(w,specification,evidence)
                answer=observed.get('answer','')
                response=dict(answer=answer,abstain=not observed['available'])
                grade=dict(correct=normalize(answer)==normalize(q['answer']),
                    coverage=[any(atom_present(a,d) for d in evidence) for a in q['answer_atoms']])
            else:
                response=reader.answer(q['question'],evidence)
                grade=reader.grade(q['question'],q['answer'],atoms,response)
            trials[name]=dict(response=response,grade=grade,documents=len(evidence),
                evidence_fingerprint=__import__('hashlib').sha256('\n'.join(evidence).encode()).hexdigest())
        except (ValueError,KeyError,TypeError,OSError) as exc:
            trials[name]=dict(error=str(exc),documents=len(evidence))
    complete=all('grade' in t for t in trials.values())
    blind=float(trials['blind']['grade']['correct']) if complete else None
    suff=float(trials['full']['grade']['correct']) if complete else None
    deletion={fid:sum(trials['delete:'+fid]['grade']['coverage'])/len(atoms) for fid in necessity['atoms']} if complete else {}
    deletion_ok=complete and all(not trials['delete:'+a['id']]['grade']['coverage'][i]
        and not trials['delete:'+a['id']]['grade']['correct'] for i,a in enumerate(q['answer_atoms']))
    passed=complete and blind<=.10 and suff>=.90 and deletion_ok and not trials['delete_all']['grade']['correct']
    return dict(mock=reader.mock, measurement='mock_oracle_only' if reader.mock else 'real_llm_reader',
        identity=reader.identity, evidence_policy='complete_gold_equivalents_and_context_carriers',
        evidence_rounds=evidence_rounds, complete=complete, passed=passed, blind=blind,suff=suff,
        deletion=max(deletion.values(),default=None), atom_deletion=deletion, trials=trials)
