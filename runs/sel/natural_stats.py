"""CPU naturalness comparison; run on H20 beside q_train_*.json and pool_*.

python natural_stats.py --questions 'q_train_*.json' --pools 'pool_*' \
    --groups world3/groups.jsonl --out natural_stats.json
python natural_stats.py --mock --groups world3/groups.jsonl --out mock_stats.json

Sampling is label free: 16 distinct candidates sampled from each top-P fused pool.
Entity overlap uses the SAME conservative lexical named-mention heuristic for both
sources (not world3's privileged ledger parser). Report extraction coverage too.
--entity-map can supply audited {qid:{round_index:[entity_names]}} annotations for
real pools; source/method is recorded. Missing pools fail unless --allow-missing.
No models, GPUs, SSH, third party packages, or services are needed.
"""
import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path
import random
import re
import unittest
from fmt import doc_str, round_prev, split_round
from k1 import rounds_of

STOP=set('On Decision It Links Regarding User Assistant Document Conditions January February March April May June July August September October November December I The This That We You Thanks'.split())


def entities(text):
    # Ignore decision IDs, dates and renderer headers. Include quoted names and
    # proper-name spans, with all-uppercase opaque IDs normalized as entity mentions.
    text=re.sub(r'(?:User|Assistant):\s*(?:\[[^\]]*\]\s*)?(?:[A-Z][A-Za-z]*:\s*)?', '', text)
    body=re.sub(r'Decision [A-Z]{8}|decision [A-Z]{8}|<[^>]*>|\[[^\]]*\]', '', text)
    mentions=re.findall(r'\b[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*\b',body)
    return {x.casefold() for x in mentions if x not in STOP and len(x)>1}


# Explicit correspondences by source form, not an aggregate benchmark mixture.
CARRIER_BASELINES={'assistant':('beam','lme'), 'record':('ep',),
                  'group':('ev',), 'dyad':('loc',)}


def text_speaker(text,benchmark=None):
    """Read only a known benchmark's name prefix on the first user line."""
    if benchmark not in ('ev','loc'):return None
    # Unwrap only doc_str's anchored header, including its optional context.
    # Never search for User: in prose or skip into an assistant's reply.
    if text.startswith('<Document>:'):
        header=re.match(r'<Document>: \([^\r\n)]*\) (?:\[follow-up to: [^\r\n]*\]\n)?User: ',text)
        if not header:return None
        text=text[header.end():]
    line=text.split('\n',1)[0].removesuffix('\r')
    name_pattern=r"[^\W\d_]+(?:[-'’][^\W\d_]+)*(?: [^\W\d_]+(?:[-'’][^\W\d_]+)*)*"
    prefix=r'\[Group [0-9]+\] ' if benchmark=='ev' else ''
    m=re.match(prefix+'('+name_pattern+r'):(?:[ \t]|$)',line)
    if m and len(m[1])<=64 and m[1] not in (
            'User','Assistant','Document','Title','Links','Conditions','Decision'):
        return m[1]
    return None


def turn_speaker(turn,benchmark=None):
    name=turn.get('speaker') or turn.get('name')
    if name:return name,'metadata'
    if turn['role']=='user':
        name=text_speaker(turn.get('content',''),benchmark)
        if name:return name,'text_prefix'
    return turn['role'].capitalize(),'role_fallback'


def document_speakers(group):
    fallbacks=group.get('speakers',[None]*len(group['docs']))
    return [s or text_speaker(d,group.get('benchmark')) for d,s in zip(group['docs'],fallbacks)]


def summarize(groups):
    lengths=[];words=[];pairs=same_entity=same_speaker=0;covered=total=0;counts=collections.Counter()
    for g in groups:
        docs=g['docs'];speakers=document_speakers(g)
        es=g.get('entities') or [entities(d) for d in docs]
        lengths.extend(map(len,docs));words.extend(len(d.split()) for d in docs)
        counts.update([len(docs)]);covered+=sum(bool(e) for e in es);total+=len(es)
        for i in range(len(docs)):
            for j in range(i):
                pairs+=1;same_entity+=bool(set(es[i])&set(es[j]))
                same_speaker+=speakers[i] is not None and speakers[i]==speakers[j]
    def quantile(xs,p):
        if not xs:return None
        xs=sorted(xs);return xs[round((len(xs)-1)*p)]
    return dict(groups=len(groups),candidates=total,candidate_counts=dict(counts),pairs=pairs,
        same_entity_pair_share=same_entity/max(1,pairs),same_speaker_pair_share=same_speaker/max(1,pairs),
        text_speaker_coverage=sum(text_speaker(d,g.get('benchmark')) is not None for g in groups for d in g['docs'])/max(1,total),
        speaker_coverage=sum(s is not None for g in groups for s in document_speakers(g))/max(1,total),
        entity_coverage=covered/max(1,total),chars_mean=sum(lengths)/max(1,total),
        chars_p10=quantile(lengths,.1),chars_p50=quantile(lengths,.5),chars_p90=quantile(lengths,.9),
        words_p50=quantile(words,.5))


def pool_groups(question_files,pool_dirs,P=600,size=16,seed=0,entity_map=None,allow_missing=False):
    groups=[];missing=[];matched=[];inputs={};pool_hashes={}
    for file in question_files:
        benchmark=Path(file).stem.removeprefix('q_train_')
        inputs[str(file)]=hashlib.sha256(Path(file).read_bytes()).hexdigest()
        questions=json.loads(Path(file).read_text())
        for q in questions:
            qid=q['question_id']
            matches=[Path(d)/(qid+'.json') for d in pool_dirs if (Path(d)/(qid+'.json')).exists()]
            if not matches:missing.append(qid);continue
            if len(matches)>1:
                # Avoid silently comparing the wrong benchmark pool.
                suffix=Path(file).stem.removeprefix('q_train_')
                preferred=[p for p in matches if p.parent.name in ('pool_'+suffix,'pool_train_'+suffix)]
                if len(preferred)!=1: raise ValueError('ambiguous pool for '+qid)
                matches=preferred
            pp=matches[0];pool=json.loads(pp.read_text());matched.append(str(pp));pool_hashes[str(pp)]=hashlib.sha256(pp.read_bytes()).hexdigest()
            rounds=rounds_of(q);prev=round_prev(q)
            indices=list(dict.fromkeys(pool['fused'][:P]))
            if any(not isinstance(i,int) or not 0<=i<len(rounds) for i in indices):raise ValueError('invalid pool indices: '+qid)
            if len(indices)<size:missing.append(qid+':short_pool');continue
            rng=random.Random(f'{seed}:{qid}');chosen=rng.sample(indices,size)
            docs=[doc_str(rounds[i][1],*split_round(rounds[i][2]),prev[i]) for i in chosen]
            # Metadata alignment matches rounds_of, including unpaired assistant turns.
            meta=[]
            for session in q['haystack_sessions']:
                i=0
                while i<len(session):
                    turn=session[i];meta.append(turn_speaker(turn,benchmark))
                    i+=2 if turn['role']=='user' and i+1<len(session) and session[i+1]['role']=='assistant' else 1
            row=dict(qid=qid,benchmark=benchmark,docs=docs,speakers=[meta[i][0] for i in chosen],
                     speaker_sources=[meta[i][1] for i in chosen])
            if entity_map is not None:
                annotations=entity_map.get(qid,{})
                row['entities']=[annotations.get(str(i),[]) for i in chosen]
            groups.append(row)
    if missing and not allow_missing:raise ValueError(f'{len(missing)} missing/short pools; examples: {missing[:5]}')
    return groups,dict(question_sha256=inputs,pool_sha256=pool_hashes,matched_pool_files=matched,missing=missing,P=P,size=size,seed=seed,
        speaker_method='metadata speaker/name first; ev [Group N] Name: or loc Name: on first user line only; role fallback only when missing',
        entity_method='audited external annotations' if entity_map is not None else 'shared lexical named mentions heuristic')


def mock_groups(seed=17,n=120):
    """Heterogeneous conversation fixture; never labeled as a real baseline."""
    rng=random.Random(seed);out=[]
    for k in range(n):
        docs=[];speakers=[]
        for j in range(16):
            person=rng.choice(['Ari','Bela','Cleo','Davi']);project=rng.choice(['Maple','Orion','River','Atlas','Juniper'])
            text=f'{person}: We discussed {project}. '+rng.choice(['Please keep the notes. ','The booking needs another review. ','I can attend on Tuesday. '])*rng.randint(1,35)
            docs.append(doc_str('January-20-2025',text,''));speakers.append(person)
        out.append(dict(docs=docs,speakers=speakers,benchmark='mock'))
    return out


def compare(synthetic,baseline,mock=False):
    a,b=summarize(synthetic),summarize(baseline)
    metrics=['same_entity_pair_share','same_speaker_pair_share','entity_coverage','chars_mean','chars_p10','chars_p50','chars_p90']
    rows=[dict(metric=k,world3=a[k],baseline=b[k],delta=None if a[k] is None or b[k] is None else a[k]-b[k]) for k in metrics]
    # Descriptive checks flag large shifts. They never certify synthetic fixtures
    # as natural, and are explicitly separate from the shortcut acceptance line.
    shifts=[r['metric'] for r in rows if r['delta'] is not None and ((abs(r['delta'])>.20) if 'share' in r['metric'] else ('chars' in r['metric'] and (r['world3']/max(1,r['baseline'])>2 or r['world3']/max(1,r['baseline'])<.5)))]
    by_benchmark={k:summarize([g for g in baseline if g.get('benchmark','unknown')==k])
                  for k in sorted({g.get('benchmark','unknown') for g in baseline})}
    by_carrier={k:summarize([g for g in synthetic if g.get('carrier','unknown')==k])
                for k in sorted({g.get('carrier','unknown') for g in synthetic})}
    correspondence=[]
    for carrier,benchmarks in CARRIER_BASELINES.items():
        if carrier not in by_carrier:continue
        for benchmark in benchmarks:
            if benchmark not in by_benchmark:continue
            aa,bb=by_carrier[carrier],by_benchmark[benchmark]
            correspondence.append(dict(carrier=carrier,benchmark=benchmark,
                table=[dict(metric=k,world3=aa[k],baseline=bb[k],
                            delta=None if aa[k] is None or bb[k] is None else aa[k]-bb[k]) for k in metrics]))
    speaker_sources=dict(collections.Counter(s for g in baseline for s in g.get('speaker_sources',[])))
    stratified_shifts=[]
    for pair in correspondence:
        for r in pair['table']:
            if r['delta'] is None:continue
            if (abs(r['delta'])>.20 if 'share' in r['metric'] else
                'chars' in r['metric'] and not .5<=r['world3']/max(1,r['baseline'])<=2):
                stratified_shifts.append(pair['carrier']+'/'+pair['benchmark']+'/'+r['metric'])
    return dict(stratified_large_shifts=stratified_shifts,by_benchmark=by_benchmark,by_carrier=by_carrier,
        carrier_baseline_comparison=correspondence,speaker_sources=speaker_sources,
        carrier_baseline_mapping=CARRIER_BASELINES,
        mock=mock,measurement='mock_comparison_only' if mock else 'real_benchmark_pool_comparison',
        world3=a,baseline=b,table=rows,large_shifts=shifts,real_baseline_measured=not mock,
        groups_semantic_sha256=hashlib.sha256(json.dumps(synthetic,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        naturalness_pass=not mock and not any(g.get('mock',False) for g in synthetic) and bool(correspondence) and not stratified_shifts)



def reported_length_comparison(synthetic,report_paths):
    """Reuse saved H20 LENGTHS only; old role-based speaker shares are invalid.

    No raw pools are present locally. This mode never claims to have remeasured
    speakers, entities or real prose quality, and preserves source hashes.
    """
    report=compare(synthetic,[],mock=True)
    stats={};sources={}
    for file in report_paths:
        p=Path(file);name=p.stem.removeprefix('natural_real_')
        values={k:None for k in summarize([])}
        for line in p.read_text().splitlines():
            cells=[c.strip() for c in line.strip('|').split('|')]
            if len(cells)>=3 and cells[0] in ('chars_mean','chars_p10','chars_p50','chars_p90'):
                values[cells[0]]=float(cells[2])
        if any(values[k] is None for k in ('chars_mean','chars_p10','chars_p50','chars_p90')):
            raise ValueError('missing reported length quantiles: '+str(p))
        stats[name]=values;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    pairs=[]
    for carrier,benchmarks in CARRIER_BASELINES.items():
        if carrier not in report['by_carrier']:continue
        for benchmark in benchmarks:
            if benchmark not in stats:continue
            aa,bb=report['by_carrier'][carrier],stats[benchmark]
            pairs.append(dict(carrier=carrier,benchmark=benchmark,table=[
                dict(metric=k,world3=aa[k],baseline=bb[k],delta=aa[k]-bb[k])
                for k in ('chars_mean','chars_p10','chars_p50','chars_p90')]))
    report.update(by_benchmark=stats,carrier_baseline_comparison=pairs,table=[],
        reported_lengths_only=True,measurement='saved_H20_length_reports_only',
        real_baseline_measured=False,naturalness_pass=False,
        source=dict(length_report_sha256=sources,speaker_stats_remeasured=False,
                    note='Historical length statistics only; old speaker and entity shares excluded.'))
    return report

def markdown(report):
    head='H20 已保存长度统计（未重测说话人）' if report.get('reported_lengths_only') else 'MOCK 基准夹具（非真实测量）' if report['mock'] else '真实基准候选池'
    lines=[head,'']
    if report['table']:lines+=['| 统计 | world3 | '+head+' | 差值 |','|---|---:|---:|---:|']
    for r in report['table']:
        fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
        lines.append('| '+r['metric']+' | '+' | '.join(fmt(r[k]) for k in ('world3','baseline','delta'))+' |')
    keys=('groups','candidates','chars_mean','chars_p10','chars_p50','chars_p90',
          'same_entity_pair_share','same_speaker_pair_share','speaker_coverage','text_speaker_coverage')
    fmt=lambda x:'N/A' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
    for title,rows in [('基准分层',report.get('by_benchmark',{})),('world3 载体分层',report.get('by_carrier',{}))]:
        lines.extend(['',title,'','| 分层 | '+' | '.join(keys)+' |',
                      '|---|'+'---:|'*len(keys)])
        for name,stats in rows.items():lines.append('| '+name+' | '+' | '.join(fmt(stats[k]) for k in keys)+' |')
    lines.extend(['','载体与对应基准（字符中位数）','','| 载体 | 基准 | world3 p50 | 基准 p50 | 差值 |','|---|---|---:|---:|---:|'])
    for pair in report.get('carrier_baseline_comparison',[]):
        r=next(r for r in pair['table'] if r['metric']=='chars_p50')
        lines.append('| '+pair['carrier']+' | '+pair['benchmark']+' | '+' | '.join(fmt(r[k]) for k in ('world3','baseline','delta'))+' |')
    if report.get('speaker_sources'):lines.extend(['','说话人来源：'+json.dumps(report['speaker_sources'],ensure_ascii=False)])
    return '\n'.join(lines)+'\n' 


class SpeakerParsingTests(unittest.TestCase):
    """Run with: python -B -m unittest natural_stats.SpeakerParsingTests -v."""

    def test_benchmark_prefixes_and_renderer_context(self):
        for benchmark in ('ev','loc'):
            for name in ('Nadia','Mary Jane','Wei Zhang','Anne-Marie',"O'Neil",'O’Neil','李明'):
                body=('[Group 31] ' if benchmark=='ev' else '')+name+': Hello.'
                for text in (body,doc_str('January-20-2025',body,'Alex: Reply.'),
                             doc_str('January-20-2025',body,'',prev='Other: User: Prior context.')):
                    with self.subTest(benchmark=benchmark,name=name,text=text):
                        self.assertEqual(text_speaker(text,benchmark),name)
                self.assertEqual(turn_speaker(dict(role='user',content=body),benchmark),
                                 (name,'text_prefix'))

    def test_prefixes_are_limited_to_their_benchmark(self):
        for benchmark in (None,'unknown','lme','beam','ep','mock'):
            for body in ('Nadia: Hello.','[Group 1] Nadia: Hello.','Title: Weekly plan.'):
                with self.subTest(benchmark=benchmark,body=body):
                    self.assertIsNone(text_speaker(body,benchmark))
                    self.assertEqual(turn_speaker(dict(role='user',content=body),benchmark),
                                     ('User','role_fallback'))
        self.assertIsNone(text_speaker('Nadia: Hello.','ev'))
        self.assertIsNone(text_speaker('[Group 1] Nadia: Hello.','loc'))
        self.assertIsNone(text_speaker('[planning/task3:thread1] Ari: Hello.','ev'))

    def test_no_name_titles_lists_and_code(self):
        bodies=('Please review this.\nTitle: Weekly plan.','Title: Weekly plan.',
                '1. **Brisk Walking**: Try this.','* Bergamot: Notes.',
                '- Schedule: Notes.','## Weekly plan: Notes.',
                'def update_chart(data):\n    pass',
                '```python\ndef update_chart(data):\n    pass\n```')
        for benchmark in (None,'lme','beam','ep','ev','loc'):
            for body in bodies:
                with self.subTest(benchmark=benchmark,body=body):
                    self.assertIsNone(text_speaker(body,benchmark))
                    self.assertIsNone(text_speaker(doc_str('date',body,'Title: Reply.'),benchmark))
                    self.assertEqual(turn_speaker(dict(role='user',content=body),benchmark),
                                     ('User','role_fallback'))

    def test_only_first_line_and_no_assistant_reply(self):
        for benchmark,prefix in (('ev','[Group 1] Nadia:'),('loc','Nadia:')):
            for body in ('Please review this.\n'+prefix+' Body.', '\n'+prefix+' Body.'):
                with self.subTest(benchmark=benchmark,body=body):
                    self.assertIsNone(text_speaker(body,benchmark))
                    self.assertIsNone(text_speaker(doc_str('date',body,prefix+' Reply.'),benchmark))
            for role in ('assistant','system'):
                self.assertEqual(turn_speaker(dict(role=role,content=prefix+' Reply.'),benchmark),
                                 (role.capitalize(),'role_fallback'))
            self.assertIsNone(text_speaker(doc_str('date','Please review.',prefix+' Reply.'),benchmark))
            # rounds_of renders an unpaired assistant as the document's user body.
            self.assertIsNone(text_speaker(doc_str('date','Assistant: '+prefix+' Reply.',''),benchmark))
            self.assertIsNone(text_speaker('<Document>: malformed\nUser: '+prefix+' Reply.',benchmark))

    def test_metadata_has_priority_for_turns_and_documents(self):
        for benchmark,body in (('ev','[Group 1] Nadia: Hello.'),('loc','Nadia: Hello.')):
            for role in ('user','assistant'):
                self.assertEqual(turn_speaker(dict(role=role,speaker='Ari',name='Bela',content=body),benchmark),
                                 ('Ari','metadata'))
                self.assertEqual(turn_speaker(dict(role=role,name='Bela',content=body),benchmark),
                                 ('Bela','metadata'))
            docs=[doc_str('date',body,'')]*3
            self.assertEqual(document_speakers(dict(benchmark=benchmark,docs=docs,speakers=['Ari',None,'User'])),
                             ['Ari','Nadia','User'])
            self.assertEqual(document_speakers(dict(benchmark=benchmark,docs=docs)),['Nadia']*3)

    def test_no_name_summary_preserves_roles_and_coverage(self):
        docs=[doc_str('date','Please review.\nTitle: Weekly plan.','Alex: Reply.'),
              doc_str('date','def update_chart(data):\n    pass','[Group 1] Nadia: Reply.')]
        for benchmark in ('lme','beam','ep','ev','loc'):
            group=dict(benchmark=benchmark,docs=docs,speakers=['User','User'])
            stats=summarize([group])
            self.assertEqual(stats['same_speaker_pair_share'],1.)
            self.assertEqual(stats['text_speaker_coverage'],0.)
            self.assertEqual(stats['speaker_coverage'],1.)
            group.pop('speakers')
            self.assertEqual(document_speakers(group),[None,None])
            self.assertEqual(summarize([group])['speaker_coverage'],0.)

    def test_pool_pipeline_keeps_benchmark_metadata_and_unpaired_roles(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);files=[];pools=[]
            for benchmark in ('ev','loc','lme'):
                prefix='[Group 1] ' if benchmark=='ev' else ''
                turns=[dict(role='user',content=prefix+'Nadia: Hello.'),
                       dict(role='assistant',content=prefix+'Alex: Reply.'),
                       dict(role='assistant',content=prefix+'Bela: Unpaired reply.'),
                       dict(role='user',speaker='Ari',content=prefix+'Nadia: Conflicting prefix.'),
                       dict(role='user',content='Please review.\nTitle: Weekly plan.')]
                q=dict(question_id=benchmark,haystack_sessions=[turns],haystack_dates=['date'])
                file=root/('q_train_'+benchmark+'.json');file.write_text(json.dumps([q]));files.append(file)
                pool=root/('pool_'+benchmark);pool.mkdir();pools.append(pool)
                (pool/(benchmark+'.json')).write_text(json.dumps(dict(fused=[0,1,2,3])))
            groups,_=pool_groups(files,pools,size=4)
            for group in groups:
                expected={'Assistant':'role_fallback','Ari':'metadata','User':'role_fallback'}
                if group['benchmark'] in ('ev','loc'):expected['Nadia']='text_prefix'
                self.assertEqual(dict(zip(group['speakers'],group['speaker_sources'])),expected)
                self.assertEqual(document_speakers(group),group['speakers'])
                self.assertEqual(summarize([group])['text_speaker_coverage'],
                                 .5 if group['benchmark'] in ('ev','loc') else 0.)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--questions',default='q_train_*.json');ap.add_argument('--pools',default='pool_*')
    ap.add_argument('--groups',required=True);ap.add_argument('--out',default='natural_stats.json')
    ap.add_argument('--P',type=int,default=600);ap.add_argument('--size',type=int,default=16);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--baseline-reports',help='glob of saved natural_real_*.md; lengths only, never speaker remeasurement')
    ap.add_argument('--entity-map');ap.add_argument('--allow-missing',action='store_true');ap.add_argument('--mock',action='store_true')
    a=ap.parse_args(argv)
    synthetic=[json.loads(x) for x in Path(a.groups).read_text().splitlines() if x.strip()]
    if a.baseline_reports:
        if a.mock:raise ValueError('--mock and --baseline-reports are separate baseline modes')
        paths=sorted(glob.glob(a.baseline_reports))
        if not paths:raise ValueError('no saved baseline length reports')
        report=reported_length_comparison(synthetic,paths)
    elif a.mock:baseline=mock_groups(a.seed);source=dict(fixture=True)
    else:
        files=sorted(glob.glob(a.questions));dirs=[p for p in sorted(glob.glob(a.pools)) if Path(p).is_dir()]
        if not files or not dirs:raise ValueError('no q_train/pool inputs; use --mock only for offline fixtures')
        baseline,source=pool_groups(files,dirs,a.P,a.size,a.seed,json.loads(Path(a.entity_map).read_text()) if a.entity_map else None,a.allow_missing)
    if not a.baseline_reports:
        report=compare(synthetic,baseline,a.mock);report['source']=source
    report['groups_sha256']=hashlib.sha256(Path(a.groups).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    Path(a.out).with_suffix('.md').write_text(markdown(report))
    print(markdown(report))
    return report

if __name__=='__main__':main()
