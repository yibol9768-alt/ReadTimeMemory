"""Question-free attacks and an independent text-to-query ideal solver.

Attacks consume ONLY docs. Labels enter held-out training and aggregate metrics,
never feature extraction. Candidate position is never a feature.
"""
import collections
import datetime as dt
import hashlib
import itertools
from functools import lru_cache
import re
import numpy as np
from world3_domains import DOMAINS
from world3_structure import STRUCTURES, RELATION_TYPES

ROW = re.compile(r"On (\d{4}-\d\d-\d\d), (\w+)'s (personal|team) (.+?)=([^\n]+?) was ([a-z_]+) \(([^\n]*?)\)\. Decision ([A-Z]{8})\.(?: It replaces decision ([A-Z]{8})\.)?(?: Links: ([^\n]*?)\.)?")
NATURAL_ROW = re.compile(r"On (\d{4}-\d\d-\d\d), (\w+)'s (personal|team) (.+?)(?: \((\d{4}-\d\d-\d\d)\))? had (.+?); its status was ([a-z ]+)\."
    r"(?: This applied (.+?)\.)?(?: This was the (.+?) update\.)?(?: It replaced the (.+?) update\.| It replaced the entry from (\d{4}-\d\d-\d\d)\.)?(?: It depended on ([^\n]*?)\.)?")
DIMS = ('entity', 'attr', 'holder_scope', 'condition')
LIVE = {'approved', 'executed'} | {s['live'] for s in STRUCTURES.values()}


@lru_cache(maxsize=32768)
def parse_document(document):
    attrs = set()
    for o in DOMAINS.values():
        attrs.update(o['attributes']); attrs.update(o[k] for k in ('skill', 'requirement', 'location', 'availability', 'clearance'))
        attrs.update(o['clearance'] + ' / ' + a for a in ('equipment', 'room', 'schedule', 'archive', 'materials'))
    attrs.update(s['relation'] for s in STRUCTURES.values()); attrs.update(a for names in RELATION_TYPES.values() for a in names); attrs.add('coordinator')
    attrs = sorted(attrs, key=len, reverse=True)
    record = re.search(r'<Document>:\s*\(([^)]+)\)', document)
    try: recorded = dt.datetime.strptime(record[1], '%B-%d-%Y').date().isoformat()
    except (ValueError, TypeError): recorded = '9999-12-31'
    rows = []
    for m in ROW.finditer(document):
        date, holder, scope, slot, value, status, cond, proposal, replaces, links = m.groups()
        attr = next((a for a in attrs if slot.endswith(' ' + a)), None)
        # Generic fallback for synthetic leakage regression fixtures.
        if attr is None:
            entity, _, attr = slot.rpartition(' ')
        else: entity = slot[:-len(attr)-1]
        condition = dict(p.split('=', 1) for p in cond.split(', ')) if cond != 'always' else {}
        rows.append(dict(id=proposal + ':' + date + ':' + status, chain=proposal, proposal=proposal,
            holder=holder, scope=scope, holder_scope=(holder, scope), entity=entity, attr=attr, value=value,
            status=status, condition=condition, condition_key=tuple(sorted(condition.items())),
            event_time=date, record_time=recorded, seq=len(rows), supersedes=None,
            replacement={'proposal': replaces} if replaces else None,
            links=[] if not links or links == 'none' else links.split(', '), statement=m[0]))
    for m in NATURAL_ROW.finditer(document):
        date,holder,scope,entity,version_date,description,status,cond,proposal,replaces,replacement_date,links=m.groups()
        attr=next((a for a in attrs if description.startswith(a.replace('_',' ')+' ')),None)
        if attr is None:continue
        value=description[len(attr.replace('_',' '))+1:]
        condition={} if not cond or cond=='in all settings' else dict(p.split(' was ',1) for p in cond.removeprefix('when the ').split(' and the '))
        if proposal is None:
            proposal=hashlib.sha256(repr((holder,scope,entity,attr,value,condition,version_date,replacement_date)).encode()).hexdigest()
        status=status.replace(' ','_')
        rows.append(dict(id=proposal+':'+date+':'+status,chain=proposal,proposal=proposal,
            holder=holder,scope=scope,holder_scope=(holder,scope),entity=entity,attr=attr,value=value,
            status=status,condition=condition,condition_key=tuple(sorted(condition.items())),
            event_time=date,record_time=recorded,seq=len(rows),supersedes=None,
            replacement={'proposal':replaces} if replaces else ({'event_time':replacement_date} if replacement_date else None),
            links=links.split(', ') if links else [],statement=m[0]))
    return rows


def key(row, dim):
    return row['condition_key'] if dim == 'condition' else row[dim]


def rate(groups, scores):
    unique, tie = 0, 0.
    for g, values in zip(groups, scores):
        best = [i for i, v in enumerate(values) if v == max(values)]
        if g['label'] in best:
            unique += len(best) == 1; tie += 1 / len(best)
    n = len(groups); chance = sum(1/len(g['docs']) for g in groups) / max(1, n)
    return dict(n=n, unique_correct=unique, unique_rate=unique/max(1,n), tie_accuracy=tie/max(1,n),
        chance=chance, threshold=chance+.10, **{'pass': bool(n) and unique/max(1,n) <= chance+.10})


def center_scores(rows, order, tie_break='all', newest=True):
    flat = [r for rs in rows for r in rs]
    pool = sorted(flat, key=lambda r: hashlib.sha256(r['statement'].encode()).hexdigest())
    for dim in order:
        counts = collections.Counter(key(r, dim) for r in pool)
        if not counts: break
        modes = [k for k, n in counts.items() if n == max(counts.values())]
        if tie_break == 'min': modes = [min(modes)]
        if tie_break == 'max': modes = [max(modes)]
        if tie_break == 'first': modes = modes[:1]
        pool = [r for r in pool if key(r, dim) in modes]
    ids = {r['id'] for r in pool}
    def score(r):
        # Matches the audit's tuple rule, including preference for a live newer row.
        return (r['id'] in ids, r['status'] in LIVE, r['event_time'] if newest else '')
    return [max((score(r) for r in rs), default=(False,False,'')) for rs in rows]


def group_features(docs):
    rows = [parse_document(d) for d in docs]
    # Main clause is last if a follow-up includes an earlier partial/full clause.
    primary = [rs[-1] if rs else None for rs in rows]
    features = []
    for j, (d, r) in enumerate(zip(docs, primary)):
        nums = [len(d)/1000, len(d.split())/200, len(rows[j]), d.count('Assistant:')]
        if r:
            for dim in DIMS:
                nums.append(sum(x is not None and key(x,dim)==key(r,dim) for x in primary)/len(docs))
            # All pair-conditioned frequencies, centrality, event rank and live rank.
            for a,b in itertools.combinations(DIMS,2):
                nums.append(sum(x is not None and key(x,a)==key(r,a) and key(x,b)==key(r,b) for x in primary)/len(docs))
            dates = [x['event_time'] for x in primary if x]
            nums += [sum(t < r['event_time'] for t in dates)/len(docs), sum(t==r['event_time'] for t in dates)/len(docs),
                     float(r['status'] in LIVE), float(bool(r['replacement'])), len(r['condition']),len(r['links'])]
            nums += [float(r['status']==status) for status in sorted(LIVE | {'proposed','vetoed','rolled_back'})]
            entity_words = set(r['entity'].lower().split())
            nums += [sum(len(entity_words & set(link.lower().split())) / max(1, len(entity_words | set(link.lower().split())))
                         for link in r['links']) / max(1, len(r['links']))]
        tokens = re.findall(r'[a-z_]+|\d+', d.lower())
        nums += [len(set(tokens))/max(1,len(tokens))]
        nums += [0.] * (40-len(nums))
        words = np.zeros(256)
        for w in set(re.findall(r'[a-z_]+|\d+', d.lower())):
            h = int(hashlib.sha256(w.encode()).hexdigest()[:8],16)
            words[h%256] += 1 if (h//256)%2 else -1
        features.append(nums + words.tolist())
    return np.array(features,dtype=float)


def learned_scores(groups):
    """Leave-one-ledger-out logistic regression + nonlinear boosted stumps.

    Standardization and fitting use training ledgers only. No qid fallback is
    claimed to be ledger holdout. A single-ledger smoke audit is unvalidated.
    """
    folds = sorted({g['ledger_id'] for g in groups})
    xs = [group_features(g['docs']) for g in groups]
    results = {k: [None]*len(groups) for k in ('logistic_relative', 'boosted_relative')}
    for fold in folds:
        train = [i for i,g in enumerate(groups) if g['ledger_id']!=fold]
        test = [i for i,g in enumerate(groups) if g['ledger_id']==fold]
        if not train:
            for k in results:
                for i in test: results[k][i] = [0.]*len(groups[i]['docs'])
            continue
        x = np.concatenate([xs[i] for i in train]); y = np.concatenate([np.arange(len(groups[i]['docs']))==groups[i]['label'] for i in train]).astype(float)
        mean, std = x.mean(axis=0), x.std(axis=0)+.1
        z = (x-mean)/std
        weights = np.where(y>0, 15., 1.)
        beta = np.zeros(z.shape[1]); intercept=0.
        for _ in range(100):
            p = 1/(1+np.exp(-np.clip(z@beta+intercept,-25,25)))
            error = (p-y)*weights
            beta -= .05*((z.T@error)/len(y)+.04*beta)
            intercept -= .05*error.mean()
        for i in test: results['logistic_relative'][i] = np.round(((xs[i]-mean)/std)@beta+intercept,8).tolist()
        # Gradient boosting on group statistics + hashed text, using residual squared loss.
        pred = np.full(len(y),y.mean()); stumps=[]
        cuts = [(f,t) for f in range(x.shape[1]) for t in np.unique(np.quantile(x[:,f],[.25,.5,.75])) if x[:,f].min()<=t<x[:,f].max()]
        masks = np.array([x[:,f]<=t for f,t in cuts])
        for _ in range(32):
            residual = y-pred
            if not cuts: break
            left_n = masks.sum(axis=1); right_n=len(y)-left_n
            left = (masks@residual)/np.maximum(left_n,1)
            right = ((~masks)@residual)/np.maximum(right_n,1)
            gains = left_n*left**2+right_n*right**2
            k=int(gains.argmax()); f,t=cuts[k]; a,b=.3*left[k],.3*right[k]
            pred += np.where(masks[k],a,b); stumps.append((f,t,a,b))
        for i in test:
            results['boosted_relative'][i] = np.round(sum((np.where(xs[i][:,f]<=t,a,b) for f,t,a,b in stumps), np.full(len(xs[i]),y.mean())),8).tolist()
    return results, dict(method='leave_one_ledger_out', ledgers=len(folds), validated=len(folds)>1,
        features='candidate_text_hashes_and_group_frequencies_pairs_ranks', candidate_position=False)


def universal_probes(groups):
    rows = [[parse_document(d) for d in g['docs']] for g in groups]
    center = []
    # Every nonempty ordered subset (64), both date policies, deterministic tie policies.
    for size in range(1,len(DIMS)+1):
        for order in itertools.permutations(DIMS,size):
            for tie in ('all','min','max','first'):
                for newest in (False,True):
                    stats=rate(groups,[center_scores(rs,order,tie,newest) for rs in rows])
                    stats.update(order=order,tie_policy=tie,newest=newest); center.append(stats)
    best=max(center,key=lambda x:x['unique_rate']) if center else rate(groups,[])
    best.update(method='best_global_ordered_subset_center_rule', searched_rules=len(center))
    latest=[]
    for valid in (False,True):
        scores=[[max(((r['status'] in LIVE if valid else True,r['event_time']) for r in rs),default=(False,'')) for rs in g] for g in rows]
        stats=rate(groups,scores); stats['live_first']=valid;latest.append(stats)
    learning,validation=learned_scores(groups)
    out={'group_center':best,'latest_state':max(latest,key=lambda x:x['unique_rate'])}
    for name,scores in learning.items():
        out[name]=rate(groups,scores);out[name]['validation']=validation
        out[name]['pass'] = out[name]['pass'] and validation['validated']
    return out


def projected_rows(document, catalog=None):
    """Canonical text parser, or an explicitly verified anchor projection.

    The latter is a semantic-verifier binding, not an independent NLP reader.
    The catalog includes distractors equally and has no query/gold annotations.
    """
    rows=parse_document(document)
    if not catalog:return rows
    from world3_style import atom_present
    canonical={f['statement'] for f in catalog}
    rows=[r for r in rows if r['statement'] not in canonical]
    signatures={(r['proposal'],r['event_time'],r['status']) for r in rows}
    return rows+[dict(f,holder_scope=(f['holder'],f['scope']),condition_key=tuple(sorted(f['condition'].items())))
            for f in catalog if (f['proposal'],f['event_time'],f['status']) not in signatures and atom_present(f,document)]


def competition_check(docs, positive, catalog=None):
    rows=[projected_rows(d,catalog) for d in docs]
    if not rows[positive]: return {'pass':False,'reason':'unparsed_positive'}
    target=rows[positive][0]
    details={}
    for dim in DIMS:
        c=collections.Counter(key(r,dim) for rs in rows for r in rs)
        own=c[key(target,dim)];other=max((n for v,n in c.items() if v!=key(target,dim)),default=0)
        details[dim]=dict(positive_same=own,competing=other)
    return {'pass':all(v['competing']>0 for v in details.values()),'dimensions':details,
        'policy':'distinct_competing_categories; counts are descriptive, blind probes enforce chance+10'}


def ideal_prediction(q, docs, covered):
    from world3 import solve, visible_documents
    # Recover rows solely from visible evidence, without answer_atoms/gold metadata.
    rows={}
    for d in visible_documents(q):
        for r in projected_rows(d,q.get('evidence_catalog')):
            if len(r['statement']) > len(rows.get(r['id'], {}).get('statement', '')):
                rows[r['id']] = dict(r)
    byproposal={r['proposal']:r for r in rows.values()}
    for r in rows.values():
        replacement=r.get('replacement')
        if replacement:
            old=byproposal.get(replacement.get('proposal'))
            if old is None and replacement.get('event_time'):
                old=next((x for x in rows.values() if all(x[k]==r[k] for k in ('holder','scope','entity','attr','condition_key'))
                          and x['event_time']==replacement['event_time'] and x['status'] in LIVE),None)
            if old:
                r['supersedes']=old['id']
                r['replacement']=dict(replacement,proposal=old['proposal'])
    from world3_semantics import parse_question
    solved=solve({'facts':list(rows.values())},parse_question(q['question']))
    from world3_style import atom_present
    needed=[rows[fid] for fid in solved['atom_ids'] if not atom_present(rows[fid],q.get('_selected_text',''))]
    return [i for i,d in enumerate(docs) if any(atom_present(f,d) for f in needed)],solved['answer']
