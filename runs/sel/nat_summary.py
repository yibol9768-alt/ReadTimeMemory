"""Native versus existing LOBO arms; same paired qid bootstrap as lobo-summary."""
import argparse
import hashlib
import json
from pathlib import Path
from budget_sample import paired_bootstrap, BENCH, LOBO_VERSION


def compare(root, folds=('loc', 'ev'), arm='nat'):
    if arm not in ('nat', 'natj'): raise ValueError('unknown native arm')
    root = Path(root)
    result = dict(single_seed=True, seed=0, interval_scope='paired qid uncertainty only; excludes training randomness',
                  contrasts={})
    for baseline in ('jia', 'yi', 'jiasmall'):
        per_fold, left, right = {}, {}, {}
        for fold in folds:
            paths = {name: root / f'{name}_{fold}_s0/results.json' for name in (arm, baseline)}
            data = {arm: json.loads(p.read_text()) for arm, p in paths.items()}
            nat, base = data[arm], data[baseline]
            if (any(r.get('version') != LOBO_VERSION or r.get('arm') != arm or r.get('fold') != fold
                    or r.get('seed') != 0 for arm, r in data.items())
                    or nat['plan']['sample'] != base['plan']['sample']
                    or nat['inference'] != base['inference']
                    or nat['plan']['data_freeze_md5'] != base['plan']['data_freeze_md5']):
                raise ValueError(f'incompatible LOBO pair: {arm}-{baseline}/{fold}')
            if abs(nat['plan']['ratio'] - 1) > .05 + 1e-12:
                raise ValueError('native FLOPs outside yi tolerance')
            left[fold], right[fold] = nat['inner_per_question'][fold], base['inner_per_question'][fold]
            per = {b: paired_bootstrap({b: nat['inner_per_question'][b]}, {b: base['inner_per_question'][b]}, strata=(b,)) for b in BENCH}
            per_fold[fold] = dict(contrast=arm + '-' + baseline, fold=fold,
                                  nat_area=nat['heldout_area'], baseline_area=base['heldout_area'],
                                  per_bench=per, full_inner=paired_bootstrap(nat['inner_per_question'], base['inner_per_question']),
                                  planned_flops_ratio_nat_baseline=nat['planned_flops'] / base['planned_flops'],
                                  actual_flops_ratio_nat_baseline=nat['actual_flops'] / base['actual_flops'],
                                  native=nat['plan']['native'],
                                  result_md5={arm: hashlib.md5(p.read_bytes()).hexdigest() for arm, p in paths.items()},
                                  **per[fold])
        result['contrasts'][arm + '-' + baseline] = dict(folds=per_fold,
            heldout_two_fold=paired_bootstrap(left, right, strata=tuple(folds)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='s23/all_rr8_mr4_2e437ddad635/lobo')
    ap.add_argument('--arm', choices=('nat', 'natj'), default='nat')
    a = ap.parse_args()
    r = compare(a.root, arm=a.arm)
    path = Path(a.root) / f'comparison_{a.arm}_quick_s0.json'
    path.write_text(json.dumps(r, ensure_ascii=False, indent=2))
    for contrast, entry in r['contrasts'].items():
        for fold, c in entry['folds'].items():
            print(f'{contrast:14s} {fold}: {c["nat_area"]:.3f} vs {c["baseline_area"]:.3f}; delta={c["difference"]:+.3f} CI95={c["ci95"]}')
    print(path)


if __name__ == '__main__':
    main()
