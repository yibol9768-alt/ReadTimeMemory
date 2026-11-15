"""Freeze D4 training IDs, real C-recipe groups, spare order and H4 world hashes.

CPU only. Run in d4dev; sources may be symlinks. Output paths are relative to the
freeze file, so the file can be copied together with its source layout. No new
questions or extra steps are invented if the actual C recipe is too small.
"""
import argparse
import collections
import hashlib
import math
from pathlib import Path
import random
import tempfile

from d4data import C_RECIPE, read_json, stratified_take, write_json

TAGS = ("ws4", "ws6f", "ws6v", "ws6e")
EXCLUSION_DIRS = ("ws1", "ws3", "ws3t", "ws4", "ws4_bak", "ws5p", "ws6", "ws6e", "ws7", "ws7e", "ws7x")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_id(qid):
    return int(qid.split("_")[0][1:])


def is_group(q):
    return any("group" in str(c).lower() for c in q.get("carriers", [q.get("carrier", "")]))


def names(w):
    return {w["name"]} | set(w.get("team", []))


def h4_metadata(directory, last, audit_dirs):
    entries, worlds, missing = [], {}, []
    for wid in range(2000, last + 1):
        path = directory / f"w{wid:05d}.json"
        if not path.is_file():
            missing.append(wid)
            continue
        w = read_json(path)
        if w["world"] != wid: raise ValueError(f"world/file ID mismatch: {path}")
        worlds[wid] = w
        entries.append(dict(world=wid, file=str(path), sha256=sha256(path),
                            split="H4s" if wid % 2 == 0 else "H4j",
                            family=w.get("family", w.get("render_family")),
                            project=w.get("project"), names=sorted(names(w)), knobs=w.get("knobs")))
    split_names = {half: set().union(*(names(w) for wid, w in worlds.items() if wid % 2 == parity))
                   for half, parity in (("H4s", 0), ("H4j", 1))}
    overlaps = []
    existing = []
    missing_dirs = []
    for directory_ in audit_dirs:
        files = sorted(directory_.glob("w0*.json"))
        if not files: missing_dirs.append(str(directory_))
        for path in files:
            old = read_json(path)
            existing.append((str(path), old))
    # Tiny world 1999 must be excluded too.
    tiny = directory / "w01999.json"
    if audit_dirs and tiny.is_file(): existing.append((str(tiny), read_json(tiny)))
    for wid, w in worlds.items():
        for path, old in existing:
            # world 1999 (e2e-tiny) draws from the same 30-name table as H4 by design (§12.1): only its project is checked
            common = set() if path.endswith("w01999.json") else names(w) & names(old)
            same_project = str(w.get("project", "")).strip().casefold() == str(old.get("project", "")).strip().casefold()
            if common or same_project:
                overlaps.append(dict(world=wid, existing=path, names=sorted(common), project=same_project))
    repeated_projects = []
    for wid, w in worlds.items():
        for wid2, other in worlds.items():
            if wid2 > wid and str(w.get("project", "")).strip().casefold() == str(other.get("project", "")).strip().casefold():
                repeated_projects.append([wid, wid2])
    return dict(worlds=entries, missing_worlds=missing,
                H4s=[wid for wid in worlds if wid % 2 == 0], H4j=[wid for wid in worlds if wid % 2],
                H4s_H4j_name_overlap=sorted(split_names["H4s"] & split_names["H4j"]),
                H4s_H4j_name_overlap_count=len(split_names["H4s"] & split_names["H4j"]),
                isolation=dict(status="not_checked" if not audit_dirs else
                               "fail" if overlaps or repeated_projects else
                               "incomplete" if missing_dirs or not tiny.is_file() else "pass",
                               checked_world_files=len(existing), missing_dirs=missing_dirs,
                               tiny_world_checked=bool(audit_dirs and tiny.is_file()),
                               overlaps=overlaps, repeated_H4_projects=repeated_projects),
                family_note="null means the source world did not record a generator family; never inferred")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="D4_FREEZE.json")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--c-seed", type=int, default=0, help="legacy mkdata C seed; default matches the unmodified C command")
    ap.add_argument("--groups", type=int, default=9600, help="production default 9600; smaller values for CPU smoke tests")
    ap.add_argument("--worlds", default=",".join(TAGS), help="training subsets only; default all four")
    ap.add_argument("--ids", default="", help="smoke-test restriction, intersected with ws4_tr.txt")
    ap.add_argument("--h4-dir", default="")
    ap.add_argument("--h4-last", type=int, default=2011, help="include any mandated additional worlds in numeric order")
    ap.add_argument("--audit-dirs", default="", help="comma-separated existing-world directories for exact project/name isolation audit")
    a = ap.parse_args(argv)
    if a.groups <= 0 or a.h4_last < 2011: ap.error("groups must be positive; H4 must include 2000–2011")
    tags = a.worlds.split(",")
    if len(set(tags)) != len(tags) or set(tags) - set(TAGS): ap.error("training worlds must be unique ws4,ws6f,ws6v,ws6e")
    root = Path(a.data_dir).absolute()
    out = Path(a.out).absolute()
    extra_ids = set(Path(a.ids).read_text().split()) if a.ids else None
    train_ids = set((root / "ws4_tr.txt").read_text().split())
    sources, metadata = [], []
    import os
    relative = lambda path: os.path.relpath(path, out.parent)
    for tag in tags:
        path, pooldir = root / f"env_{tag}.json", root / f"pool_{tag}"
        qs = read_json(path)
        source = dict(tag=tag, qfile=relative(path), pooldir=relative(pooldir), sha256=sha256(path))
        if tag == "ws4": source.update(ids=relative(root / "ws4_tr.txt"), ids_sha256=sha256(root / "ws4_tr.txt"))
        sources.append(source)
        for q in qs:
            qid = q["question_id"]
            if tag == "ws4" and qid not in train_ids: continue
            if extra_ids is not None and qid not in extra_ids: continue
            metadata.append(dict(qid=qid, source=tag, world=world_id(qid), type=q["question_type"],
                                 scope=q.get("scope"), group_chat=is_group(q)))
        del qs
    if len({q["qid"] for q in metadata}) != len(metadata): raise ValueError("overlapping training question IDs")
    metadata.sort(key=lambda q: q["qid"])
    nondated = [q for q in metadata if q["scope"] != "date"]
    dated = [q for q in metadata if q["scope"] == "date"]
    # d/(n+d) <= .15 iff 17*d <= 3*n. Exact integer arithmetic.
    date_cap = min(len(dated), 3 * len(nondated) // 17)
    selected = nondated + stratified_take(dated, date_cap, random.Random(a.seed), key=lambda q: q["type"])
    selected.sort(key=lambda q: q["qid"])
    kept = {q["qid"] for q in selected}
    available = []
    from mkdata import main as mkdata_main
    # Use the actual legacy C loop, including its positive choices and random draws.
    # group_sink observes completed groups, and skips only expensive text rendering.
    with tempfile.TemporaryDirectory(prefix="d4freeze-", dir=out.parent) as temp:
        idsfile = Path(temp) / "ids.txt"
        idsfile.write_text("\n".join(sorted(kept)) + "\n")
        for source in sources:
            def capture(row):
                if row.pop("nneg") == 15: available.append(row)
            mkdata_main([str(out.parent / source["qfile"]), str(out.parent / source["pooldir"]),
                         str(Path(temp) / "unused.jsonl"), "--ids", str(idsfile), "--seed", str(a.c_seed)] + C_RECIPE,
                        group_sink=capture)
    spare_size = math.ceil(a.groups / 10)
    rng = random.Random(a.seed + 2)
    groups = stratified_take(available, a.groups, rng, key=lambda r: r["type"])
    chosen = {(r["qid"], r["positive"], r["neg_seed"]) for r in groups}
    rest = [r for r in available if (r["qid"], r["positive"], r["neg_seed"]) not in chosen]
    spare = stratified_take(rest, spare_size, rng, key=lambda r: r["type"])
    seeds_by_q = collections.defaultdict(list)
    for row in groups + spare: seeds_by_q[row["qid"]].append(row["neg_seed"])
    for q in selected: q["neg_seeds"] = seeds_by_q[q["qid"]]
    h4 = h4_metadata(Path(a.h4_dir).absolute() if a.h4_dir else root / "ws8", a.h4_last,
                     [Path(p).expanduser().absolute() for p in a.audit_dirs.split(",") if p])
    problems = []
    if len(groups) < a.groups: problems.append(f"C recipe yields only {len(available)} groups; need {a.groups} + {spare_size} spare")
    elif len(spare) < spare_size: problems.append(f"only {len(spare)} spare groups; need {spare_size}")
    if h4["missing_worlds"]: problems.append(f"missing H4 worlds: {h4['missing_worlds']}")
    if h4["isolation"]["status"] in ("fail", "incomplete"): problems.append("H4 isolation audit failed or incomplete")
    result = dict(schema="d4-freeze-v1", spec="D4_SPEC_2026-11-02 v2.4", seed=a.seed,
                  status="insufficient" if problems else "ready",
                  test_only=bool(a.ids or a.groups != 9600 or tuple(tags) != TAGS),
                  problems=problems, sources=sources, questions=selected,
                  question_ids_by_world={str(w): [q["qid"] for q in selected if q["world"] == w] for w in sorted({q["world"] for q in selected})},
                  recipe=dict(no_set=True, ignore_kinds="repeat,fu", hard_skip=10, boost=3, steps=8,
                              policy="rr8", p_mixed=0.3, P=600, K=15, seed=a.c_seed),
                  group_seed_rule="SHA256 of legacy RNG state after C sampling; first 64 bits, no extra RNG draw",
                  group_sampling="proportional by question_type, largest remainders, without replacement; main then spare",
                  target_groups=a.groups, groups=groups, spare=spare,
                  counts=dict(eligible_questions=len(metadata), questions=len(selected), date_questions=date_cap,
                              date_fraction=date_cap / len(selected) if selected else 0,
                              group_chat_questions=sum(q["group_chat"] for q in selected),
                              by_type=dict(collections.Counter(q["type"] for q in selected)),
                              by_source=dict(collections.Counter(q["source"] for q in selected)),
                              group_chat_by_source=dict(collections.Counter(q["source"] for q in selected if q["group_chat"])),
                              questions_with_main_groups=len({r["qid"] for r in groups}),
                              questions_with_main_or_spare_groups=len({r["qid"] for r in groups + spare}),
                              C_available_groups=len(available), groups=len(groups), spare=len(spare),
                              groups_by_type=dict(collections.Counter(r["type"] for r in groups)),
                              spare_by_type=dict(collections.Counter(r["type"] for r in spare))),
                  group_chat_definition="any carrier in q['carriers'] (or q['carrier']) containing 'group', case insensitive",
                  H4=h4, E0_training_exposure=["ws4", "ws6e (972-975)"],
                  provenance_note="env_* exports are treated as passed-gate training questions; no gate recomputation",
                  pending=dict(filtering_scale="not measured/frozen here", H4_headroom="not measured; no evaluation invoked"))
    write_json(out, result)
    print(f"questions {len(selected)} date {date_cap} groups {len(groups)} spare {len(spare)} available {len(available)} H4 {len(h4['worlds'])}")
    for problem in problems: print("INSUFFICIENT:", problem)
    if problems: return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
