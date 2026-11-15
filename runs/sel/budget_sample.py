"""Frozen, benchmark-stratified nested question samples for SPEC_BUDGET v1.1.

python budget_sample.py --frac 0.1 --out-dir s23/all_rr8/budget
Sampling seed 1234 is independent of training seed. Existing freezes fail closed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

BENCH = ("beam", "lme", "loc", "ev", "ep")
FRACTIONS = (0.1, 0.25, 1.0)
VERSION = "budget-2026-11-06-v1.1"
LOBO_VERSION = "lobo-2026-11-07-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.md5(canonical(value).encode()).hexdigest()


def make_sample(questions, frozen, frac, inputs=None):
    if frac not in FRACTIONS:
        raise ValueError("budget fraction must be 0.1, 0.25 or 1.0")
    per = {}
    for bench in BENCH:
        ids = [q["question_id"] for q in questions[bench]]
        if any(not isinstance(qid, str) or not qid for qid in ids):
            raise ValueError(f"{bench}: invalid question identity")
        train, inner = set(frozen[bench]["train"]), set(frozen[bench]["inner"])
        if (not ids or len(ids) != len(set(ids)) or set(ids) != train or train & inner):
            raise ValueError(f"{bench}: training identities differ from freeze")
        order = sorted(ids)
        random.Random(1234).shuffle(order)
        target = math.ceil(frac * len(ids))
        selected = sorted(order[:target])
        per[bench] = {"qids": selected, "qid_md5": digest(selected),
                      "qid_order": order,
                      "n_train": len(ids), "n_selected": len(selected),
                      "realized_fraction": len(selected) / len(ids)}
    return {"version": VERSION, "sampling_unit": "question", "sampling_seed": 1234, "frac": frac,
            "input_md5": inputs or {}, "bench": per,
            "qid_md5": digest({b: per[b]["qids"] for b in BENCH})}


def freeze_json(path, value):
    """Atomic create-only publication; concurrent identical writers are harmless."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen budget artifact changed: {path}")
        return value
    import os
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=1)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError(f"frozen budget artifact changed: {path}")
    finally:
        os.unlink(temporary)
    return value


def make_lobo_sample(questions, frozen, fold, inputs=None):
    """The full question set in four benchmarks, and no supervision in fold."""
    if fold not in BENCH:
        raise ValueError("invalid LOBO fold")
    sample = make_sample(questions, frozen, 1.0, inputs)
    held = sample["bench"][fold]
    held.update(qids=[], qid_md5=digest([]), n_selected=0, realized_fraction=0.0)
    sample.update(version=LOBO_VERSION, fold=fold,
                  qid_md5=digest({b: sample["bench"][b]["qids"] for b in BENCH}))
    return sample


def freeze_sample(frac, out_dir, question_ids=None, *, fold=None):
    questions, inputs = {}, {}
    paths = ["DATA_FREEZE_STAGE23.json"] + [f"q_train_{b}.json" for b in BENCH]
    for path in paths:
        checksum = hashlib.md5()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        inputs[path] = checksum.hexdigest()
    frozen = json.loads(Path(paths[0]).read_text())["bench"]
    for bench in BENCH:
        if question_ids is not None:
            questions[bench] = [{"question_id": qid} for qid in question_ids[bench]]
        else:
            # Retain identities only; never keep five multi-GB histories alive.
            with open(f"q_train_{bench}.json") as stream:
                questions[bench] = [{"question_id": q["question_id"]} for q in json.load(stream)]
    sample = (make_sample(questions, frozen, frac, inputs) if fold is None else
              make_lobo_sample(questions, frozen, fold, inputs))
    name = f"budget_f{frac}.json" if fold is None else f"lobo_{fold}.json"
    return freeze_json(Path(out_dir) / name, sample)


def distinct_rows(rows, identity):
    seen, result = set(), []
    for row in rows:
        key = identity(row)
        if key not in seen:
            seen.add(key); result.append(row)
    return result


def supervision_counts(stages, real_qids, identity):
    """Count consumed real teacher scoring inputs, including scores on gold rows.

    qid+formatted query+formatted document identifies a scoring input; teacher
    name separates ensemble supervision from RR8 gold scores in the breakdown.
    Synthetic qids never enter these counts.
    """
    pairs, by_teacher = Counter(), defaultdict(Counter)
    real_questions, gold_questions = set(), set()
    groups = {"teacher": Counter(), "gold": Counter(), "crucible": Counter()}
    for rows in stages:
        for line in rows:
            row = json.loads(line)
            real = row["qid"] in real_qids
            kind = ("teacher" if row.get("nolabel") else "gold") if real else "crucible"
            groups[kind][identity(row)] += 1
            if not real:
                continue
            real_questions.add(row["qid"])
            if kind == "gold": gold_questions.add(row["qid"])
            name = row.get("teacher_name", "rr8")
            scores = row.get("teacher", [])
            if len(scores) != len(row["docs"]):
                raise ValueError("unaligned teacher scores")
            for doc, score in zip(row["docs"], scores):
                if score is not None:
                    key = (row["qid"], row["query"], doc)
                    pairs[key] += 1; by_teacher[name][key] += 1
    def counts(counter):
        return {"unique": len(counter), "exposures": sum(counter.values()),
                "repeats": sum(counter.values()) - len(counter)}
    return {"unique_teacher_pairs": len(pairs), "teacher_pairs": counts(pairs),
            "unique_real_questions": len(real_questions), "unique_gold_questions": len(gold_questions),
            "teacher_pairs_by_teacher": {t: counts(c) for t, c in sorted(by_teacher.items())},
            "groups": {k: counts(c) for k, c in groups.items()}}


def paired_bootstrap(left, right, iterations=4000, *, strata=BENCH):
    """Stratified paired qid bootstrap, benchmark means weighted equally (points)."""
    if not strata or set(left) != set(strata) or set(right) != set(strata):
        raise ValueError("paired bootstrap requires all requested benchmarks")
    differences = []
    for bench in strata:
        if not left[bench] or set(left[bench]) != set(right[bench]):
            raise ValueError("paired bootstrap qid mismatch")
        differences.append([left[bench][q] - right[bench][q] for q in sorted(left[bench])])
    rng = random.Random(1234)
    samples = sorted(sum(sum(rng.choices(d, k=len(d))) / len(d) for d in differences) / len(strata)
                     for _ in range(iterations))
    def percentile(p):
        index = (len(samples) - 1) * p
        lo = int(index); hi = math.ceil(index)
        return samples[lo] + (samples[hi] - samples[lo]) * (index - lo)
    return {"difference": sum(sum(d) / len(d) for d in differences) / len(strata),
            "ci95": [percentile(.025), percentile(.975)], "iterations": iterations,
            "seed": 1234, "unit": "paired qid within benchmark", "units": "area points"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frac", type=float, choices=FRACTIONS, required=True)
    parser.add_argument("--out-dir", default="s23/all_rr8/budget")
    args = parser.parse_args()
    result = freeze_sample(args.frac, args.out_dir)
    print(json.dumps({"qid_md5": result["qid_md5"],
                      "realized_fractions": {b: v["realized_fraction"] for b, v in result["bench"].items()}}))


if __name__ == "__main__":
    main()
