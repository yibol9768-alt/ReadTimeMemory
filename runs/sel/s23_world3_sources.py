"""Optional s23 source routing, activated only by the reviewed s23_v3_patch.md.

No GPU, subprocess or import of s23. All three environment variables are required
as a bundle, so stage reservations cannot accidentally fall back to the old pool.
"""
import json
import os
from pathlib import Path
import random

ENV = ('S23_WORLD3_S1', 'S23_WORLD3_S2', 'S23_WORLD3_S3')
ROOT = Path(__file__).resolve().parent


def enabled():
    present = [bool(os.environ.get(k)) for k in ENV]
    if any(present) and not all(present):
        raise ValueError('set all S23_WORLD3_S1/S2/S3 together')
    return all(present)


def sources():
    if not enabled(): return None
    paths = [Path(os.environ[k]).resolve() for k in ENV]
    if len(set(paths)) != 3: raise ValueError('stage sources must be distinct')
    # s23 mounts ROOT at /w, including preflight and FLOPs subprocesses.
    if any(not p.is_relative_to(ROOT) for p in paths):
        raise ValueError('world3 sources must be inside the s23 /w mount')
    return [str(p.relative_to(ROOT)) for p in paths]


def stage1():
    return sources()[0] if enabled() else 'tr_w20.jsonl'


def container_stage1():
    return '/w/' + stage1()


def stage1_count(default=17784):
    if not enabled(): return default
    return len(_validated_chunks()[0])


def provenance_paths(default):
    return sources() if enabled() else default


def _validated_chunks():
    """Validate the entire reserved bundle before stage 1 or stage 2/3."""
    paths = sources()
    if paths is None: raise ValueError('world3 source override is disabled')
    chunks = [(ROOT / p).read_text().splitlines(keepends=True) for p in paths]
    ledgers=[]; identities=[]
    for chunk in chunks:
        rows=[json.loads(r) for r in chunk]
        if not rows or any(r.get('split')!='train' or not r.get('ledger_id') for r in rows):
            raise ValueError('world3 requires nonempty train ledgers')
        if any(r.get('mock') for r in rows) and os.environ.get('S23_WORLD3_ALLOW_MOCK') != '1':
            raise ValueError('mock sources require explicit S23_WORLD3_ALLOW_MOCK=1 for CPU tests only')
        ledgers.append({r['ledger_id'] for r in rows})
        ids=[json.dumps([r['qid'],r['query'],sorted(r['docs'])],sort_keys=True) for r in rows]
        if len(set(ids)) != len(ids): raise ValueError('duplicate world3 group identities')
        identities.append(set(ids))
    for i in range(3):
        for j in range(i):
            if ledgers[i]&ledgers[j] or identities[i]&identities[j]:
                raise ValueError('world3 stage reservation overlap')
    return chunks


def reserved(n2, n3, seed):
    """Return separate, disjoint seeded blocks; never repeat or slice a combined pool."""
    chunks = _validated_chunks()
    if n2 > len(chunks[1]) or n3 > len(chunks[2]): raise ValueError('insufficient reserved world3 groups')
    random.Random(seed).shuffle(chunks[1]);random.Random(seed+1).shuffle(chunks[2])
    return chunks[1][:n2],chunks[2][:n3]


def tag():
    import hashlib
    if not enabled(): return ''
    return '_world3_' + hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in sources())).hexdigest()[:12]


def checkpoint(seed):
    return f'ckpt/s23_s1{tag()}_s{seed}'
