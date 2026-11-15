"""Reproducible CPU-only mock acceptance across domains and independent seeds."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import world3 as v
from world3_adapter import groups_for,mock_pool,final_audit,export_s23
from world3_extra_probes import extra_probes
import natural_stats


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='codex_tmp/world3_v4_cpu/final')
    ap.add_argument('--seeds',default='31,137,941');ap.add_argument('--ledger-counts',default='4,4,4')
    a=ap.parse_args(argv);root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    seeds=list(map(int,a.seeds.split(',')));counts=list(map(int,a.ledger_counts.split(',')))
    if len(seeds)!=len(counts):raise ValueError('one ledger count per seed')
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders
    backend=Tokenizer(models.BPE(vocab={c:i for i,c in enumerate(pre_tokenizers.ByteLevel.alphabet())},merges=[]))
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False);backend.decoder=decoders.ByteLevel()
    backend.save(str(root/'tokenizer.json'));tokenizer=v.load_tokenizer(root/'tokenizer.json')
    tokenizer.encode=lru_cache(maxsize=8192)(tokenizer.encode);original_decode=tokenizer.decode
    @lru_cache(maxsize=8192)
    def decode_cached(ids):return original_decode(list(ids))
    tokenizer.decode=lambda ids:decode_cached(tuple(ids))
    manifest=dict(mock=True,measurement='CPU_mock_only',seeds=seeds,ledgers_per_domain=counts,questions_per_ledger=20,
        tokenizer='serialized byte-BPE fixture; not production tokenizer',max_d=5000,
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('world3.py','world3_adapter.py','world3_probes.py','world3_structure.py','world3_semantics.py','world3_extra_probes.py','world3_reader.py','natural_stats.py','s23_v3_patch.md')})
    v.write(root/'manifest.json',manifest)
    for seed,count in zip(seeds,counts):
        target=root/f'seed{seed}';target.mkdir(exist_ok=True);questions=[];groups=[];reports=[];ledgers=[]
        for i,domain in enumerate(list(v.DOMAINS)*count):
            m=v.ModelFamilies(mock=True);w=v.verify(v.render(v.make_plan(i,20,seed=seed,domain=domain),m,par=4),m)
            r={};qs=v.build_questions(w,build_report=r,tokenizer=tokenizer,token_limit=5000)
            if len(qs)!=20:raise AssertionError(r['quarantine'])
            for q in qs:
                gs,rs=groups_for(q,mock_pool(q),no_set=True,p=100000,tokenizer=tokenizer,max_d=5000);groups+=gs;reports+=rs
            questions+=qs;ledgers.append(v.quota_audit(w,qs))
            print('seed',seed,domain,i,'questions',len(qs),'groups',len(groups),flush=True)
            tokenizer.encode.cache_clear();decode_cached.cache_clear()
        (target/'groups.jsonl').write_text(''.join(json.dumps(g,ensure_ascii=False)+'\n' for g in groups))
        v.write(target/'ledgers.json',ledgers)
        audit=final_audit(groups,questions,reports);audit['additional_shortcut_probes']={k.removeprefix('extra_'):r for k,r in audit['shortcut_probes'].items() if k.startswith('extra_')}
        audit['checks']['additional_shortcut_probes']=all(r['pass'] for r in audit['additional_shortcut_probes'].values())
        audit['cpu_contract_pass']=all(audit['checks'].values());v.write(target/'audit.json',audit)
        comparison=natural_stats.compare(groups,natural_stats.mock_groups(),mock=True)
        v.write(target/'natural_stats.json',comparison);(target/'natural_stats.md').write_text(natural_stats.markdown(comparison))
        export_s23(groups,target/'s23_export',audit,seed)
        print('CHECKS',seed,audit['checks'],flush=True)
        print('PROBES',seed,{k:r['unique_rate'] for k,r in audit['shortcut_probes'].items()},flush=True)
        print('EXTRA',seed,{k:r['unique_rate'] for k,r in audit['additional_shortcut_probes'].items()},flush=True)
        assert len(groups)==count*120 and audit['cpu_contract_pass'] and audit['ideal_rule_accuracy']==1.
    print('ALL_SEEDS_PASS',flush=True)

if __name__=='__main__':main()
