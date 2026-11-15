"""Additional text-only probes retained from the third independent review.

Every rule is selected globally; no per-question oracle rule selection. Features
never access questions, query_spec, answer_atoms, entities, speakers or labels.
Only held-out fitting and final evaluation consume labels.
"""
import collections
import hashlib
import itertools
import re
import numpy as np
from world3_probes import parse_document, rate, LIVE


def tokens(d):return set(re.findall(r'[a-z_]+|\d+',d.lower()))


def fingerprint(d):
    # Replace dates, decision IDs and opaque entity suffixes, preserving syntax.
    d=re.sub(r'\b\d{4}-\d\d-\d\d\b|\b[A-Z]{8}\b',' ID ',d)
    d=re.sub(r'\b(?:Ari|Bela|Cleo|Davi|Esme|Farah)\b',' PERSON ',d)
    return tokens(d)


def fit(groups,xs):
    result=[None]*len(groups)
    for fold in sorted({g['ledger_id'] for g in groups}):
        train=[i for i,g in enumerate(groups) if g['ledger_id']!=fold];test=[i for i,g in enumerate(groups) if g['ledger_id']==fold]
        if not train:
            for i in test:result[i]=[0.]*len(groups[i]['docs'])
            continue
        x=np.concatenate([xs[i] for i in train]);y=np.concatenate([np.arange(len(groups[i]['docs']))==groups[i]['label'] for i in train]).astype(float)
        mean=x.mean(0);std=x.std(0)+.1;z=(x-mean)/std;beta=np.zeros(x.shape[1]);intercept=0
        for _ in range(100):
            p=1/(1+np.exp(-np.clip(z@beta+intercept,-25,25)));error=(p-y)*np.where(y,15.,1.)
            beta-=.05*(z.T@error/len(y)+.04*beta);intercept-=.05*error.mean()
        for i in test:result[i]=np.round(((xs[i]-mean)/std)@beta+intercept,8).tolist()
    return result


def _linear_and_rule_probes(groups):
    scores=collections.defaultdict(list);features=collections.defaultdict(list)
    status=sorted(LIVE|{'proposed','vetoed','rolled_back','off_shift','quarantined','expired','failed','recalled','resigned','yanked','cancelled','consumed'})
    for g in groups:
        docs=g['docs'];rows=[parse_document(d) for d in docs];sets=list(map(tokens,docs));fps=list(map(fingerprint,docs))
        n=len(docs);sim=np.array([[len(a&b)/max(1,len(a|b)) if i!=j else 0 for j,b in enumerate(sets)] for i,a in enumerate(sets)])
        length=[];dates=[];carrier=[];finger=[]
        for i,(d,rs) in enumerate(zip(docs,rows)):
            length.append([len(d),len(d.split()),len(rs),len(sets[i]),d.count('\n'),d.count('approve'),d.count('confirm')])
            days=[int(r['event_time'].replace('-','')) for r in rs]
            dates.append([min(days,default=0),max(days,default=0),len(set(days)),len(set(r['status'] for r in rs))])
            carrier.append([d.count(k) for k in ('[group','[dyad','[assistant','[record','planning',':thread','Assistant:')])
            v=np.zeros(128)
            for w in fps[i]:v[int(hashlib.md5(w.encode()).hexdigest()[:8],16)%128]+=1
            finger.append(v)
        for name,x in [('length_style',length),('dates',dates),('carrier',carrier),('template_fingerprint',finger)]:features[name].append(np.array(x,float))
        for sign in (-1,1):
            suffix='min' if sign<0 else 'max'
            scores['length_'+suffix].append([sign*len(d) for d in docs])
            scores['mean_overlap_'+suffix].append((sign*sim.mean(1)).round(10).tolist())
            scores['nearest_overlap_'+suffix].append((sign*sim.max(1)).round(10).tolist())
            scores['template_frequency_'+suffix].append([sign*fps.count(x) for x in fps])
            for rank in range(1,n):scores[f'similarity_rank_{rank}_{suffix}'].append((sign*np.sort(sim,axis=1)[:,-rank]).round(10).tolist())
        for st in status:scores['status_'+st].append([sum(r['status']==st for r in rs) for rs in rows])
        # Full 120 orders with independent holder/scope axes, including conditional modes.
        for order in itertools.permutations(('entity','holder','scope','attr','condition_key')):
            remaining=[r for rs in rows for r in rs]
            for dim in order:
                counts=collections.Counter(r[dim] for r in remaining)
                if counts:remaining=[r for r in remaining if counts[r[dim]]==max(counts.values())]
            ids={r['id'] for r in remaining}
            scores['split_center_'+'_'.join(order)].append([max(((r['id'] in ids,r['status'] in LIVE,r['event_time']) for r in rs),default=(False,False,'')) for rs in rows])
    out={}
    for category,prefix in [('length_rule','length_'),('overlap_rule','mean_overlap_'),('nearest_overlap','nearest_overlap_'),('similarity_ranks','similarity_rank_'),('template_frequency','template_frequency_'),('native_state_markers','status_'),('split_holder_scope_center','split_center_')]:
        results=[dict(rate(groups,values),rule=name) for name,values in scores.items() if name.startswith(prefix)]
        out[category]=max(results,key=lambda r:r['unique_rate']);out[category]['searched_rules']=len(results)
    for name,xs in features.items():
        out['heldout_'+name]=rate(groups,fit(groups,xs));out['heldout_'+name]['validation']='leave_one_ledger_out'
    return out


def forest_scores(groups,xs,trees=40,depth=5):
    """Balanced, nonlinear bootstrap trees; hold out complete ledgers."""
    results=[None]*len(groups)
    for fi,fold in enumerate(sorted({g['ledger_id'] for g in groups})):
        train=[i for i,g in enumerate(groups) if g['ledger_id']!=fold];test=[i for i,g in enumerate(groups) if g['ledger_id']==fold]
        if not train:
            for i in test:results[i]=[0.]*len(groups[i]['docs'])
            continue
        x=np.concatenate([xs[i] for i in train]);y=np.concatenate([np.arange(len(groups[i]['docs']))==groups[i]['label'] for i in train]).astype(float)
        rng=np.random.default_rng(fi+19);posit=np.where(y==1)[0];neg=np.where(y==0)[0]
        def build(ids,remaining):
            yy=y[ids];prior=yy.mean()
            if remaining==0 or yy.min()==yy.max() or len(ids)<20:return prior
            fs=rng.choice(x.shape[1],min(16,x.shape[1]),replace=False)
            cuts=np.quantile(x[ids][:,fs],[.25,.5,.75],axis=0).T
            masks=x[ids][:,fs].T[:,None,:]<=cuts[:,:,None]
            n=masks.sum(-1);a=(masks*yy).sum(-1);b=yy.sum()-a
            gain=a*a/np.maximum(n,1)+b*b/np.maximum(len(ids)-n,1)
            gain[(n<8)|(n>len(ids)-8)]=-1
            k,t=np.unravel_index(gain.argmax(),gain.shape)
            if gain[k,t]<0:return prior
            selected=masks[k,t]
            return (int(fs[k]),float(cuts[k,t]),build(ids[selected],remaining-1),build(ids[~selected],remaining-1))
        def predict(node,arr):
            if not isinstance(node,tuple):return np.full(len(arr),node)
            k,t,a,b=node;result=np.empty(len(arr));mask=arr[:,k]<=t
            result[mask]=predict(a,arr[mask]);result[~mask]=predict(b,arr[~mask]);return result
        forest=[build(np.r_[rng.choice(posit,len(posit),replace=True),rng.choice(neg,len(posit),replace=True)],depth) for _ in range(trees)]
        for i in test:results[i]=np.round(sum(predict(t,xs[i]) for t in forest)/trees,8).tolist()
    return results



def length_only_probes(groups):
    """Only character counts enter features; carrier routes evaluation strata.

    Global and within-carrier fits both hold out complete ledgers. Only total,
    user-side and assistant-side character counts enter features; role markers
    delimit the sides but their identities, words and candidate positions do not.
    """
    def evaluate(gs):
        if not gs:return {}
        xs=[]
        for g in gs:
            from fmt import SFX
            lengths=[]
            for d in g['docs']:
                body=d.removesuffix(SFX)
                user,sep,assistant=body.partition('\nAssistant: ')
                lengths.append([len(d),len(user),len(assistant) if sep else 0])
            a=np.array(lengths,float)
            relative=(a-a.mean(0))/(a.std(0)+1)
            xs.append(np.column_stack((a,a*a/10000,relative,relative**2)))
        rules={name:rate(gs,[[sign*len(d) for d in g['docs']] for g in gs])
               for name,sign in [('shortest',-1),('longest',1)]}
        rules['heldout_chars']=dict(rate(gs,fit(gs,xs)),validation='leave_one_ledger_out')
        # Binned lengths can detect a middle-length shortcut missed by extremes.
        rules['heldout_chars_forest']=dict(rate(gs,forest_scores(gs,xs,trees=20,depth=3)),validation='leave_one_ledger_out')
        positive=[len(g['docs'][g['label']]) for g in gs]
        negative=[len(d) for g in gs for i,d in enumerate(g['docs']) if i!=g['label']]
        def stats(a):return dict(mean=float(np.mean(a)),p10=float(np.quantile(a,.1)),
                                p50=float(np.median(a)),p90=float(np.quantile(a,.9)))
        return dict(n=len(gs),rules=rules,positive=stats(positive),negative=stats(negative),
                    **{'pass':all(r['pass'] for r in rules.values())})
    overall=evaluate(groups)
    strata={c:evaluate([g for g in groups if g['carrier']==c]) for c in sorted({g['carrier'] for g in groups})}
    return dict(overall=overall,by_carrier=strata,
                feature_contract='total/user/assistant character counts only; no question, words or candidate position',
                **{'pass':bool(groups) and overall['pass'] and all(s['pass'] for s in strata.values())})

def extra_probes(groups):
    from world3_probes import group_features
    result=_linear_and_rule_probes(groups)
    scores=forest_scores(groups,[group_features(g['docs']) for g in groups])
    result['heldout_forest_40_depth5']=rate(groups,scores)
    result['heldout_forest_40_depth5']['validation']='leave_one_ledger_out'
    result['heldout_forest_40_depth5']['per_domain']={d:rate([g for g in groups if g['domain']==d],
        [s for g,s in zip(groups,scores) if g['domain']==d]) for d in sorted({g['domain'] for g in groups})}
    length_report=length_only_probes(groups)
    strongest=max((r for cell in [length_report['overall']]+list(length_report['by_carrier'].values())
                   for r in cell['rules'].values()),key=lambda r:r['unique_rate'])
    result['length_only_stratified']=dict(strongest,details=length_report)
    result['length_only_stratified']['pass']=length_report['pass']
    result['heldout_forest_40_depth5']['pass'] &= all(r['pass'] for r in result['heldout_forest_40_depth5']['per_domain'].values())
    return result
