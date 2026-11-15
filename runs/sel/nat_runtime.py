"""SGT deadline estimates from the historical yi folds; also guards each torchrun."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import time

SGT = dt.timezone(dt.timedelta(hours=8))


def deadline(now=None):
    now = now or dt.datetime.now(SGT)
    # Never roll an expired window forward to tomorrow.
    return now.replace(hour=6, minute=45, second=0, microsecond=0).timestamp()


def log_durations(text):
    starts, durations = {}, []
    stamp, day, previous = None, 0, None
    for line in text.splitlines():
        m = re.search(r'(?<!\d)(\d{2}):(\d{2})(?::(\d{2}))?(?!\d)', line)
        if m:
            seconds = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3] or 0)
            if previous is not None and seconds < previous - 12 * 3600:
                day += 86400
            previous, stamp = seconds, seconds + day
        if stamp is None or not re.search(r'\byi\b|yi_', line, re.I):
            continue
        f = re.search(r'(?:yi[_ /-]|--fold\s+)(loc|ev)(?=[_\W]|$)', line, re.I)
        if not f:
            f = re.search(r'\b(loc|ev)\b', line, re.I)
        if not f or 'smoke' in line.lower():
            continue
        fold = f[1].lower()
        if re.search(r'done|finish|completed|exit.?0|rc=0', line, re.I):
            if fold in starts and stamp > starts[fold]:
                durations.append(stamp - starts.pop(fold))
        elif re.search(r'start|begin|launch|\brun\b|python3.*lobo', line, re.I):
            starts.setdefault(fold, stamp)
    return durations


def estimate(root, log_path):
    root, log_path = Path(root), Path(log_path)
    if not log_path.exists():
        raise ValueError(f'missing historical yi log: {log_path}')
    durations = log_durations(log_path.read_text(errors='replace'))
    source = str(log_path)
    # Some launcher logs redirect child output and record only bare start lines.
    # PROGRESS has the same run's UTC start/completion times, without ambiguity.
    if not durations and (root / 'PROGRESS.jsonl').exists():
        starts = {}
        for line in (root / 'PROGRESS.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r.get('arm') != 'yi' or r.get('fold') not in ('loc', 'ev'):
                continue
            t = dt.datetime.fromisoformat(r['time_utc'].replace('Z', '+00:00')).timestamp()
            if r['status'] == 'started': starts[r['fold']] = t
            if r['status'] == 'completed' and r['fold'] in starts:
                durations.append(t - starts[r['fold']])
        source += ' (timestamps recovered from matching LOBO PROGRESS.jsonl)'
    if not durations:
        raise ValueError('cannot identify completed yi loc/ev durations in log or PROGRESS; refusing training')
    duration = max(durations)
    rates, flops23 = [], []
    for fold in ('loc', 'ev'):
        r = json.loads((root / f'yi_{fold}_s0/results.json').read_text())
        c = r['plan']['planned_stage_flops']
        cost = c['stage2'] + c['stage3']
        rates.append(cost / duration)
        flops23.append(cost)
    result = dict(source=source, yi_fold_durations_seconds=durations,
                  yi_fold_seconds=duration, conservative_flops_per_second=min(rates) / 1.3,
                  yi_s23_flops=min(flops23), deadline_epoch=deadline(),
                  estimate_note='slowest yi fold incl evaluation; 30% margin + 5min per new job')
    return result


def require_time(seconds, label, config):
    end = min(config['deadline_epoch'], deadline())
    remaining = end - time.time()
    if not math.isfinite(seconds) or seconds <= 0 or seconds > remaining:
        raise ValueError(f'REFUSE {label}: estimate={seconds:.0f}s remaining={remaining:.0f}s before 06:45 SGT')
    print(f'[{dt.datetime.now(SGT):%F %T SGT}] deadline OK {label}: estimate={seconds:.0f}s remaining={remaining:.0f}s', flush=True)


def phase_seconds(config, phase, root, arm='nat'):
    root = Path(root)
    rate = config['conservative_flops_per_second']
    s1 = 0
    if arm == 'nat':
        stage1 = json.loads(Path('s23/nat_s1_s0/recipe.json').read_text())
        s1 = sum(s['planned_flops'] * (4 / s['world']) for s in stage1['stages']) / rate + 600
    folds = []
    for f in ('loc', 'ev'):
        p = json.loads((root / f'{arm}_{f}_s0/plan.json').read_text())
        cost = p['planned_stage_flops']
        if arm == 'natj' and cost['stage1'] != 0:
            raise ValueError('natj must have zero stage-1 FLOPs')
        folds.append((cost['stage2'] + cost['stage3']) / rate + 300)
    full = max(folds)
    smoke = max(900, .25 * full)  # 50 steps/stage + model startup and fixed inner subset
    return {'all': s1 + smoke + full, 's1': s1 + smoke + full,
            'smoke': smoke + full, 'full': full}[phase]


def guard_stage(out, args, world):
    config = json.loads(Path(os.environ.get('NAT_RUNTIME_CONFIG', 'nat_runtime.json')).read_text())
    if 's23_s1_nat_' in out:
        recipe = json.loads(Path('s23/nat_s1_s0/recipe.json').read_text())
        data = args[args.index('--data') + 1].removeprefix('/w/')
        st = next(s for s in recipe['stages'] if s['data'] == data)
        cost = st['planned_flops'] * 4 / world
    else:
        p = json.loads((Path(out).parent / 'plan.json').read_text())
        stage = 2 if Path(out).name == 'st2' else 3
        cost = p['planned_stage_flops'][f'stage{stage}']
        if '--max-steps' in args:
            cost *= min(1, int(args[args.index('--max-steps') + 1]) / (p[f's{stage}_rows'] // 16))
    require_time(cost / config['conservative_flops_per_second'] + 300, out, config)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('estimate', 'check', 'remaining'))
    ap.add_argument('--root', default='s23/all_rr8_mr4_2e437ddad635/lobo')
    ap.add_argument('--arm', choices=('nat', 'natj'), default='nat')
    ap.add_argument('--log', default='s23/run_lobo1107.log')
    ap.add_argument('--config', default='nat_runtime.json')
    ap.add_argument('--phase', default='all', choices=('all', 's1', 'smoke', 'full'))
    a = ap.parse_args()
    if a.cmd == 'estimate':
        c = estimate(a.root, a.log)
        Path(a.config).write_text(json.dumps(c, indent=2))
        print(json.dumps(c))
    elif a.cmd == 'remaining':
        print(max(0, int(deadline() - time.time())))
    else:
        c = json.loads(Path(a.config).read_text())
        require_time(phase_seconds(c, a.phase, a.root, a.arm), a.phase, c)


if __name__ == '__main__':
    main()
