"""v4.1b teacher storage and T=1 targets; stdlib only (also used by train.py)."""
import json
import math


def logits(values, scale="probability"):
    out = []
    for v in values:
        if v is None:
            out.append(None); continue
        if not isinstance(v, (int, float)) or not math.isfinite(v): raise ValueError("nonfinite teacher score")
        if scale == "probability":
            if not 0 <= v <= 1: raise ValueError("teacher probability outside [0,1]")
            # Uniform clipping preserves order, including saturated endpoints.
            p = min(1 - 1e-6, max(1e-6, v))
            out.append(math.log(p / (1 - p)))
        elif scale == "score10":
            if not 0 <= v <= 10: raise ValueError("teacher score outside [0,10]")
            out.append((v - 5) / 1.5)
        elif scale == "logit": out.append(float(v))
        else: raise ValueError(f"unknown teacher scale: {scale}")
    return out


def ensemble_logits(members):
    if not members or len({len(m) for m in members}) != 1: raise ValueError("unaligned ensemble")
    standardized = []
    for z in members:
        if any(v is None for v in z): raise ValueError("missing ensemble score")
        mean = sum(z) / len(z); sd = math.sqrt(sum((v - mean) ** 2 for v in z) / len(z))
        standardized.append([(v - mean) / sd if sd else 0.0 for v in z])
    return [sum(v) / len(members) for v in zip(*standardized)]


def target(values, scale="probability"):
    z = logits(values, scale)
    if any(v is None for v in z): raise ValueError("target needs scored candidates")
    peak = max(z); e = [math.exp(v - peak) for v in z]
    return [v / sum(e) for v in e]


def teacher_maps(pool, qid, name="rr8", config=""):
    """Config: {name: {key: pool_key OR file: 'scores/name/{qid}.json', scale: ...}}.
    External files contain a round-index -> score object. Ensembles use name1+name2.
    No scores are ever written into the immutable retrieval pools.
    """
    cfg = json.load(open(config)) if config else {}
    result = []
    for member in name.split("+"):
        entry = cfg.get(member, {"key": member, "scale": "probability"})
        raw = json.load(open(entry["file"].replace("{qid}", qid))) if "file" in entry else pool[entry.get("key", member)]
        keys = list(raw); z = logits([raw[k] for k in keys], entry.get("scale", "probability"))
        result.append({int(k): v for k, v in zip(keys, z) if v is not None})
    return result


def group_logits(maps, indices):
    values = [[m[i] for i in indices] for m in maps]
    return values[0] if len(values) == 1 else ensemble_logits(values)
