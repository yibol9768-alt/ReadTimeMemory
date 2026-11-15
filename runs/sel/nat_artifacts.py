"""Refilter lineage, pool reuse, and archival of native-only stale artifacts."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import fmt


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def validate(env='env_nat2.json', pooldir='pool_nat'):
    report = json.loads(Path(env + '.report.json').read_text())
    parent = Path(report['parent_env'])
    assert report['env_md5'] == md5(env), 'refilter env changed'
    assert report['parent_md5'] == md5(parent), 'refilter parent changed'
    assert Path(pooldir, 'env_md5.txt').read_text().strip() == report['parent_md5'], 'pool parent mismatch'
    assert report['fmt_md5'] == md5(fmt.__file__), 'fmt changed: refilter again'
    before = {q['question_id']: q for q in json.loads(parent.read_text())}
    after = json.loads(Path(env).read_text())
    assert after, 'no native questions survived refilter'
    assert len({q['question_id'] for q in after}) == len(after), 'duplicate refilter ids'
    for q in after:
        old = before[q['question_id']]
        for key, value in old.items():
            if key not in ('support_stages', 'answer_values', 'answer_evidence'):
                assert q[key] == value, f'pool input changed: {q["question_id"]}/{key}'
        if q['split'] == 'train':
            assert Path(pooldir, q['question_id'] + '.json').is_file(), f'missing pool: {q["question_id"]}'
    assert any(q['question_type'] == 'single_fact' and q['split'] == 'train' for q in after), 'no training single facts'
    return report


def publish(tmp, root):
    report = validate()
    dest = Path('tr_nat.jsonl')
    from preflight_s23 import training_rows
    training_rows(tmp)
    stale = dest.exists() and md5(dest) != md5(tmp)
    recipe = Path('s23/nat_s1_s0/recipe.json')
    if recipe.exists():
        stale |= json.loads(recipe.read_text())['source_md5'] != md5(tmp)
    for fold in ('loc', 'ev'):
        plan = Path(root, f'nat_{fold}_s0/plan.json')
        if plan.exists():
            stale |= json.loads(plan.read_text())['native']['source_md5'] != md5(tmp)
    if stale:
        archive = Path('s23/nat_archive') / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        paths = [Path('s23/nat_s1_s0'), Path('ckpt/s23_s1_nat_s0'), Path('ckpt/s23_s1_nat_s0_native')]
        paths += [Path(root, f'nat_{fold}_s0') for fold in ('loc', 'ev')]
        paths += [Path(root, 'comparison_nat_quick_s0.json')]
        paths += [dest, Path('tr_nat.jsonl.refilter.json')]
        for path in paths:
            if path.exists():
                target = archive / path
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
        print('Archived stale native artifacts:', archive, flush=True)
    Path(tmp).replace(dest)
    manifest = dict(env='env_nat2.json', env_md5=report['env_md5'], parent_md5=report['parent_md5'],
                    fmt_md5=report['fmt_md5'], data_md5=md5(dest))
    Path(str(dest) + '.refilter.json').write_text(json.dumps(manifest, indent=2))


def check():
    report = validate()
    manifest = json.loads(Path('tr_nat.jsonl.refilter.json').read_text())
    assert manifest['env_md5'] == report['env_md5'], 'rebuild tr_nat from env_nat2 first'
    assert manifest['data_md5'] == md5('tr_nat.jsonl'), 'tr_nat changed since rebuild'
    assert manifest['fmt_md5'] == md5(fmt.__file__), 'training format changed'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('validate', 'publish', 'check'))
    ap.add_argument('--tmp', default='tr_nat.rebuild.jsonl')
    ap.add_argument('--root', default='s23/all_rr8_mr4_2e437ddad635/lobo')
    a = ap.parse_args()
    if a.cmd == 'publish':
        publish(a.tmp, a.root)
    else:
        (validate if a.cmd == 'validate' else check)()


if __name__ == '__main__':
    main()
