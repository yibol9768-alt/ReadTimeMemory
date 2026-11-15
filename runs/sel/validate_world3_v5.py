"""CPU-only fifth audit: recheck immutable v4 windows plus v5 acceptance.

No GPU, SSH, download or external model calls. V4 facts are regenerated with the
frozen backup generator and matched to the actual saved questions and positives.
"""
import argparse
import concurrent.futures
import subprocess
import sys
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from world3_semantics import parse_question,check_alignment
import world3 as v


def recheck_v4_windows(out):
    root=Path('codex_tmp/world3_v4_cpu/final_verified')
    snapshot=Path('bak_world3_v4/world3.py')
    spec=importlib.util.spec_from_file_location('frozen_world3_v4',snapshot)
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    rows=[];counts=collections.Counter();input_hashes={}
    for seed in (31,137,941):
        file=root/f'seed{seed}'/'groups.jsonl'
        input_hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        groups=[json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        worlds={}
        for g in groups:
            if not g['date_window']:continue
            wid=int(g['ledger_id'].rsplit(':',1)[1])
            if wid not in worlds:worlds[wid]=frozen.make_plan(wid,20,seed=seed,domain=g['domain'])
            w=worlds[wid];c=next(c for c in w['cases'] if f"w{wid}_v3_{c['qtype']}_{c['id']}"==g['qid'])
            text=g['query'].split('<Query>: ',1)[1].split('\n<Selected>:\n',1)[0]
            assert text==c['question'],(seed,g['qid'],'snapshot mismatch')
            old=frozen.solve(w,c['query']);parsed=parse_question(text);actual=v.solve(w,parsed)
            assert actual==old,(seed,g['qid'],'original gold differs')
            byid={f['id']:f for f in w['facts']}
            assert any(byid[fid]['statement'] in g['docs'][g['label']] for fid in actual['atom_ids'])
            assert parsed['date_window']==g['date_window']
            repaired_spec=dict(c['query'],date_window=g['date_window'])
            assert check_alignment(text,repaired_spec,w['facts'])['ok']
            changed=re.sub(r'Use records between [\d-]+ and [\d-]+\.',
                           'Use records between 2030-01-01 and 2030-01-02.',text)
            assert parse_question(changed)!=parsed
            assert not check_alignment(changed,repaired_spec,w['facts'])['ok']
            try:v.solve(w,parse_question(changed))
            except ValueError:rejected=True
            else:raise AssertionError((seed,g['qid'],'empty window retained gold'))
            rows.append(dict(seed=seed,qid=g['qid'],window=g['date_window'],
                original_gold_unchanged=True,atom_ids=actual['atom_ids'],empty_window_rejected=rejected))
            counts[seed]+=1
        assert counts[seed]==72,counts
        print('V4_DATE_RECHECK',seed,counts[seed],flush=True)
    report=dict(total=len(rows),per_seed=dict(counts),rows=rows,input_sha256=input_hashes,
                backup_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                record_window_semantics='inclusive record_time filter before all resolution and traversal')
    assert len(rows)==216
    v.write(Path(out)/'original_216_date_recheck.json',report)
    return report



def acceptance_worker(root,seed):
    target=Path(root)/f'worker{seed}';target.mkdir(parents=True,exist_ok=True)
    with (target/'validation.log').open('w') as log:
        result=subprocess.run([sys.executable,'-u','validate_world3_v4.py','--out',str(target),
                               '--seeds',str(seed),'--ledger-counts','4'],stdout=log,stderr=subprocess.STDOUT)
    return seed,result.returncode

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='codex_tmp/world3_v5_cpu/release_complete')
    ap.add_argument('--dates-only',action='store_true')
    ap.add_argument('--acceptance-only',action='store_true')
    a=ap.parse_args(argv)
    if not a.acceptance_only:recheck_v4_windows(a.out)
    if not a.dates_only:
        import natural_stats
        root=Path(a.out)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            statuses=list(executor.map(lambda seed:acceptance_worker(root,seed),(31,137,941)))
        if any(code for seed,code in statuses):raise AssertionError(('acceptance worker failures',statuses))
        manifests={}

        for seed in (31,137,941):
            target=root/f'worker{seed}'/f'seed{seed}'
            groups=[json.loads(x) for x in (target/'groups.jsonl').read_text().splitlines()]
            report=natural_stats.reported_length_comparison(groups,sorted(Path('codex_tmp').glob('natural_real_*.md')))
            v.write(target/'reported_h20_lengths.json',report)
            (target/'reported_h20_lengths.md').write_text(natural_stats.markdown(report))
            manifests[str(seed)]=v.read(root/f'worker{seed}'/'manifest.json')
        manifest=dict(mock=True,release_ready=False,build_contract=v.BUILD_CONTRACT,
                      seeds=[31,137,941],groups=1440,workers=manifests,source_sha256={})
        for name in ('world3.py','world3_semantics.py','world3_style.py','world3_extra_probes.py','natural_stats.py','validate_world3_v4.py','validate_world3_v5.py','test_world3_v4.py','test_world3_v5.py'):
            manifest['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        v.write(root/'manifest.json',manifest)

if __name__=='__main__':main()
