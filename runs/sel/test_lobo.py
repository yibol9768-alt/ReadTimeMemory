"""LOBO audit on synthetic data; subprocesses, SSH and GPUs are forbidden."""
import json
import hashlib
from pathlib import Path
import sys

import pytest

import budget_sample as bs
import s23
from test_budget import args as budget_args, environment, row


def args(arm="yi", fold="beam", **extra):
    return s23.clone(budget_args(arm, 1.0), cmd="lobo", fold=fold, **extra)


def sample(a):
    return bs.freeze_sample(1.0, s23.all_namespace(a) + "/lobo", fold=a.fold)


@pytest.mark.parametrize("fold", bs.BENCH)
def test_all_arms_zero_heldout_qids_bounded_fill_compute_and_real_md5(environment, monkeypatch, fold):
    original = s23.source
    def source(kind, bench, *rest):
        assert bench != fold, "heldout teacher/gold source must not even be prepared"
        return original(kind, bench, *rest)
    monkeypatch.setattr(s23, "source", source)
    plans = {arm: s23.plan_budget(args(arm, fold), sample(args(arm, fold)))
             for arm in ("yi", "jia", "jiasmall")}
    yi, jia, small = (plans[a] for a in ("yi", "jia", "jiasmall"))
    assert yi["sample"]["bench"][fold]["qids"] == []
    assert yi["sample"]["bench"][fold]["realized_fraction"] == 0
    full_ids = {b: set(s23._question_ids(f"q_train_{b}.json")) for b in bs.BENCH}
    heldout = full_ids[fold] | {f"inner_{fold}"}
    for arm, p in plans.items():
        assert p["fold"] == fold and p["heldout_groups"] == 0 and p["heldout_zero_asserted"]
        assert not any(p["per_bench_groups"][fold].values())
        assert not any(p["base"]["base_counts"][kind][fold] for kind in ("teacher2", "teacher3", "gold3"))
        assert s23.budget_dir(args(arm, fold)) == f"s23/all_rr8/lobo/{arm}_{fold}_s0"
        for b in bs.BENCH:
            if b != fold: assert set(p["sample"]["bench"][b]["qids"]) == full_ids[b]
        for stage in (2, 3):
            rows = list(map(json.loads, s23.lines(p[f"s{stage}"])))
            assert not heldout & {r["qid"] for r in rows}
            for r in rows:
                if not r["qid"].startswith("syn"):
                    assert r["qid"] in set.union(*(v for b, v in full_ids.items() if b != fold))
        s23.assert_teacher_disjoint(s23.lines(p["s2"]), s23.lines(p["s3"]))
    # Full per-benchmark allocations survive removal of just one benchmark.
    for b in bs.BENCH:
        if b == fold: continue
        assert yi["per_bench_groups"][b] == {"teacher2": 800, "teacher3": 0, "gold3": 800, "fill2": 0}
        assert jia["base"]["base_counts"]["teacher2"][b] == 800
        assert jia["base"]["base_counts"]["teacher3"][b] == 0
        assert jia["per_bench_groups"][b]["fill2"] == jia["fill_teacher_per_bench"]
    assert yi["ratio"] == 1
    assert yi["s2_rows"] == yi["s3_rows"] == 4 * 800 + 4000
    assert .95 <= jia["ratio"] <= 1.05
    assert jia["fill_teacher_per_bench"] > 0
    assert small["fill_teacher_per_bench"] == 0 and not small["compute_matched"]
    assert small["planned_stage_flops"]["stage1"] == 0
    assert yi["planned_stage_flops"]["stage1"] == 10000
    for key in ("real_teacher_s2_md5", "real_gold_s3_md5"):
        assert yi["base"][key] == small["base"][key] == jia["base"][key]
    assert sum(json.loads(r)["qid"].startswith("syn") for r in s23.lines(yi["s2"])) == 4000
    assert sum(json.loads(r)["qid"].startswith("syn") for r in s23.lines(yi["s3"])) == 4000
    for p in (jia, small): assert p["supervision"]["groups"]["crucible"]["exposures"] == 0
    base = s23.lines(s23.budget_dir(args("jia", fold)) + "/real_teacher_s2.jsonl")
    assert set(map(s23.group_identity, base)) == set(map(s23.group_identity, s23.lines(jia["s2"])))
    assert jia["supervision"]["groups"]["teacher"]["repeats"] > 0
    assert s23.plan_budget(args("jia", fold), sample(args("jia", fold))) == jia


@pytest.mark.parametrize("fold", bs.BENCH)
def test_remaining_blocks_equal_full_grid_row_for_row_md5(environment, fold):
    a = args(fold=fold)
    full_args = s23.clone(a, cmd="grid", fold="", arm="yi", s2_groups=8000, r=.5, s3_groups=8000)
    full_paths = s23.build_all(full_args)
    lobo = s23.plan_budget(a, sample(a))
    for b in bs.BENCH:
        if b == fold: continue
        qids = set(s23._question_ids(f"q_train_{b}.json"))
        for stage, full_path in zip((2, 3), full_paths):
            # Sort bytes only to undo the final cross-benchmark shuffle, retaining
            # every repeated row. No identity deduplication can hide a mismatch.
            full = sorted(r for r in s23.lines(full_path) if json.loads(r)["qid"] in qids)
            retained = sorted(r for r in s23.lines(lobo[f"s{stage}"]) if json.loads(r)["qid"] in qids)
            assert len(full) == len(retained) == 800
            assert list(map(lambda r: hashlib.md5(r.encode()).hexdigest(), full)) == [
                hashlib.md5(r.encode()).hexdigest() for r in retained]
            assert hashlib.md5("".join(full).encode()).hexdigest() == hashlib.md5("".join(retained).encode()).hexdigest()
    assert lobo["s3_rows"] == 4 * 800 + 4000


def test_lobo_cycles_original_source_bytes_and_order_without_deduplication(environment):
    for b in bs.BENCH:
        for kind in ("teacher", "gold"):
            path = Path(f"{kind}_{b}.jsonl")
            rows = s23.lines(path)[:7]
            # Duplicate identities and different JSON formatting alter the original
            # file's MD5, which is itself part of build_all's seeded selection.
            path.write_text("".join(rows + rows[:3]) + json.dumps(json.loads(rows[0]), indent=None, separators=(",", ":")) + "\n")
    a = args()
    data = s23.prepare_budget(a, sample(a))
    for b in bs.BENCH:
        if b == a.fold: continue
        expected2, _ = s23.teacher_blocks(f"teacher_{b}.jsonl", 800, 0, 0)
        expected3 = s23.take(f"gold_{b}.jsonl", 800, seed=0)
        for actual, expected in ((data["teacher2"][b], expected2), (data["gold3"][b], expected3)):
            assert len(actual) == 800 and actual == expected
            assert hashlib.md5("".join(actual).encode()).hexdigest() == hashlib.md5("".join(expected).encode()).hexdigest()


def test_samples_freeze_fail_closed_and_reject_nonempty_heldout(environment):
    a = args(); frozen = sample(a)
    assert sample(a) == frozen
    broken = json.loads(json.dumps(frozen)); broken["bench"][a.fold]["qids"] = ["c0_probe_0"]
    with pytest.raises(ValueError, match="LOBO real question sets"):
        s23.plan_budget(a, broken)
    Path("s23/all_rr8/lobo/lobo_beam.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="frozen budget artifact changed"):
        sample(a)


def test_final_qid_audit_catches_heldout_row_injection(environment, monkeypatch):
    original = s23.build_budget
    def build(a, fill=0):
        p2, p3 = original(a, fill)
        with open(p2, "a") as f: f.write(row("c0_probe_0", 0, True))
        return p2, p3
    monkeypatch.setattr(s23, "build_budget", build)
    a = args()
    with pytest.raises(ValueError, match="heldout qid leakage"):
        s23.plan_budget(a, sample(a))
    assert not Path(s23.budget_dir(a), "plan.json").exists()


@pytest.mark.parametrize("stage,delta", [(2, -1), (2, 1), (3, -1), (3, 1)])
def test_final_real_count_mismatch_refuses_freeze(environment, monkeypatch, stage, delta):
    original = s23.build_budget
    def build(a, fill=0):
        paths = original(a, fill)
        path = Path(paths[stage - 2]); rows = s23.lines(path)
        index = next(i for i, r in enumerate(rows) if not json.loads(r)["qid"].startswith("syn"))
        if delta == -1: rows.pop(index)
        else: rows.append(rows[index])
        path.write_text("".join(rows))
        return paths
    monkeypatch.setattr(s23, "build_budget", build)
    a = args()
    with pytest.raises(ValueError, match="LOBO final counts differ"):
        s23.plan_budget(a, sample(a))
    assert not Path(s23.budget_dir(a), "plan.json").exists()


@pytest.mark.parametrize("qid", ["c0_probe_0", "inner_beam"])
def test_crucible_cannot_hide_real_heldout_qids(environment, qid):
    path = Path("tr_w20.jsonl")
    rows = s23.lines(path); rows[0] = row(qid, 0)
    path.write_text("".join(rows))
    with pytest.raises(ValueError, match="Crucible/real qid collision"):
        s23.plan_budget(args(), sample(args()))


@pytest.mark.parametrize("arm", ["yi", "jia", "jiasmall"])
def test_dry_run_prepares_only_cpu_artifacts(environment, monkeypatch, capsys, arm):
    def forbidden(*a, **kw): pytest.fail("dry-run attempted execution/evaluation")
    for name in ("preflight", "budget_stage1_actual", "stage23_gpus", "train_all_stage", "inner_score", "budget_comparison"):
        monkeypatch.setattr(s23, name, forbidden)
    a = args(arm, dry_run=True, gpus="")
    p = s23.lobo(a)
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["heldout_groups"] == 0
    assert report["per_bench_groups"] == p["per_bench_groups"]
    assert report["plan"] == s23.budget_dir(a) + "/plan.json"
    assert not list(Path("s23").rglob("results.json"))
    assert not list(Path("s23").rglob("PROGRESS.jsonl"))
    assert s23.plan_budget(s23.clone(a, dry_run=False), p["sample"]) == p


def fake_execution(monkeypatch, scores):
    calls = []
    def train(a, p, stage, start, out):
        calls.append((a.arm, stage, start))
        steps = s23.candidate_steps(p[f"s{stage}_rows"])
        return [{"path": f"{out}/s{step}", "step": step,
                 "actual_flops": p["planned_stage_flops"][f"stage{stage}"] * step / steps[-1]} for step in steps]
    def score(a, ckpt, tag, sel=True, options=None):
        assert sel is False
        assert "/st3/s" in ckpt
        for b in bs.BENCH:
            Path(f"ctx_{tag}_inner_{b}.jsonl").write_text("".join(json.dumps({
                "qid": f"inner_{b}", "budget": budget, "gold_in": scores[a.arm] / 100,
                "gold_total": 1, "keep": [0] if scores[a.arm] else []}) + "\n"
                for budget in (1000, 2000, 4000, 8000, 16000)))
        # EverMem uses session recall, so its score is 100 whenever keep=[0].
        areas = {b: (100 if scores[a.arm] else 0) if b == "ev" else scores[a.arm] for b in bs.BENCH}
        return {"tag": tag, "area": sum(areas.values()) / 5, "per_bench": areas}
    monkeypatch.setattr(s23, "train_all_stage", train)
    monkeypatch.setattr(s23, "inner_score", score)
    return calls


def test_frozen_choice_never_scores_candidates_or_uses_heldout_for_selection(environment, monkeypatch):
    calls = fake_execution(monkeypatch, {"yi": 0, "jia": 100, "jiasmall": 0})
    def forbidden(*a, **kw): pytest.fail("LOBO attempted model/option selection")
    for name in ("select", "score_candidates", "select_options"):
        monkeypatch.setattr(s23, name, forbidden)
    # Distinguish the two frozen switches: each arm must follow its own choice,
    # and jia-small must follow yi, regardless of heldout evaluation performance.
    jia_path = Path(s23.choice_file(args("jia")))
    choice = json.loads(jia_path.read_text()); choice["stage2_checkpoint"] = "end"
    jia_path.write_text(json.dumps(choice))
    choice_hashes = {arm: s23.md5(s23.choice_file(args(arm))) for arm in ("yi", "jia")}
    for arm in ("yi", "jia", "jiasmall"):
        result = s23.lobo(args(arm))
        expected_switch = "end" if arm == "jia" else "mid"
        assert result["stage2_checkpoint"] == expected_switch and result["stage3_checkpoint"] == "end"
        assert result["plan"]["stage2_checkpoint"] == expected_switch
        assert result["plan"]["selected_supervision"] == result["selected_supervision"]
        assert result["plan"]["selected_unique_supervision"] == result["selected_unique_supervision"]
        expected = result["stage2_candidates"][-1 if arm == "jia" else 0]["path"]
        assert calls[-1][2] == "/w/" + expected
        assert result["heldout_area"] == result["per_bench_area"]["beam"]
        assert set(result["per_bench_area"]) == set(bs.BENCH)
        assert result["actual_flops"] == pytest.approx(result["planned_flops"])
        assert all(Path(ctx["path"]).is_file() for ctx in result["inner_ctx"].values())
    assert choice_hashes == {arm: s23.md5(s23.choice_file(args(arm))) for arm in ("yi", "jia")}
    before = len(calls)
    s23.lobo(args("yi"))
    assert len(calls) == before
    entries = [json.loads(l) for l in Path("s23/all_rr8/lobo/PROGRESS.jsonl").read_text().splitlines()]
    assert entries[-1]["status"] == "reused" and all(e["fold"] == "beam" for e in entries)
    assert not list(Path("s23").rglob("comparison*.json"))
    summary = s23.lobo_summary(args())
    audit = summary["selected_supervision_by_fold"]["beam"]
    assert audit["available_arms"] == ["jia", "jiasmall", "yi"] and not audit["missing_arms"]
    assert audit["equal"] is False and audit["differences_from_yi"]
    assert audit["stage2_choices"] == {"yi": "mid", "jia": "end", "jiasmall": "mid"}


def test_failure_ledger_records_fold(environment, monkeypatch):
    def fail(a): raise RuntimeError("fixture failure")
    monkeypatch.setattr(s23, "preflight", fail)
    with pytest.raises(RuntimeError, match="fixture failure"): s23.lobo(args())
    entries = [json.loads(l) for l in Path("s23/all_rr8/lobo/PROGRESS.jsonl").read_text().splitlines()]
    assert [e["status"] for e in entries] == ["started", "failed"]
    assert entries[-1]["fold"] == "beam"


def comparison_results(fold, arm, value):
    stages = [[row("real", 0, True)], [row("real", 1)]]
    signature = s23.lobo_unique_supervision(stages, {"real"})
    supervision = bs.supervision_counts(stages, {"real"}, s23.group_identity)
    p = {"sample": {"fold": fold}, "data_freeze_md5": "freeze", "stage2_checkpoint": "mid",
         "selected_unique_supervision": signature, "selected_supervision": supervision}
    return {"version": bs.LOBO_VERSION, "seed": 0, "fold": fold, "arm": arm,
            "plan": p, "inference": {}, "planned_flops": 10, "actual_flops": 10,
            "stage2_checkpoint": "mid", "selected_unique_supervision": signature, "selected_supervision": supervision,
            "inner_per_question": {b: {f"q{i}": value for i in range(1 if b == "beam" else 20)} for b in bs.BENCH}}


def test_both_contrasts_fold_pairing_and_five_fold_stratification(environment):
    for fold in bs.BENCH:
        for arm, value in (("yi", 12), ("jia", 10), ("jiasmall", 11)):
            s23.atomic_json(s23.budget_dir(args(arm, fold)) + "/results.json", comparison_results(fold, arm, value))
        summary = s23.lobo_summary(args(fold=fold))
        assert summary["single_seed"] is True and summary["known_recipe_seed_sd_points"] == .25
        for contrast in ("yi-jia", "yi-jiasmall"):
            entry = summary["contrasts"][contrast]
            assert entry["complete"] == (fold == "ep")
            assert entry["folds"][fold]["iterations"] == 4000
        assert summary["selected_supervision_by_fold"][fold]["equal"] is True
    for contrast, difference in (("yi-jia", 2), ("yi-jiasmall", 1)):
        aggregate = summary["contrasts"][contrast]["five_fold"]
        assert aggregate["difference"] == difference
        assert aggregate["ci95"] == [difference, difference]
        assert aggregate["iterations"] == 4000
        assert "five folds weighted equally" in aggregate["unit"]
    assert summary["contrasts"]["yi-jia"]["five_fold"]["success"] is True
    # This mutates only the heldout pair and must fail before publication.
    path = s23.budget_dir(args("jia")) + "/results.json"
    result = json.loads(Path(path).read_text()); result["inner_per_question"]["beam"]["unpaired"] = 10
    Path(path).write_text(json.dumps(result))
    with pytest.raises(ValueError, match="qid mismatch"): s23.lobo_comparison(args())


def test_stratified_bootstrap_is_qid_paired_and_equal_fold_weight():
    left = {b: {f"q{i}": i * 10 for i in range(1 if b == "beam" else 20)} for b in bs.BENCH}
    right = {b: {q: v - (5 if b == "beam" else 0) for q, v in values.items()} for b, values in left.items()}
    result = bs.paired_bootstrap(left, right)
    assert result["difference"] == 1 and result["ci95"] == [1, 1]
    single = bs.paired_bootstrap({"beam": left["beam"]}, {"beam": right["beam"]}, strata=("beam",))
    assert single["difference"] == 5 and single["ci95"] == [5, 5]


@pytest.mark.parametrize("arm", ["yi", "jia", "jiasmall"])
@pytest.mark.parametrize("fold", bs.BENCH)
def test_cli_dispatches_frozen_lobo_defaults(monkeypatch, arm, fold):
    monkeypatch.setattr(sys, "argv", ["s23.py", "lobo", "--arm", arm, "--fold", fold, "--seed", "0", "--gpus", "4,5,6,7", "--dry-run"])
    monkeypatch.setattr(s23, "lobo", lambda a: a)
    a = s23.main()
    assert (a.arm, a.fold, a.seed, a.gpus, a.dry_run) == (arm, fold, 0, "4,5,6,7", True)
    assert (a.teacher, a.teacher_config) == ("rr8+mr4", "s23/teachsel/teacher_mr4.json")


@pytest.mark.parametrize("extra", [[], ["--fold", "bad"], ["--fold", "ep", "--seed", "1"],
                                  ["--fold", "ep", "--frac", "1.0"], ["--fold", "ep", "--smoke"]])
def test_cli_rejects_unregistered_variants(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["s23.py", "lobo", "--arm", "yi"] + extra)
    with pytest.raises(SystemExit) as error: s23.main()
    assert error.value.code == 2


def test_cli_dispatches_lobo_smoke_and_summary(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["s23.py", "lobo", "--arm", "yi", "--fold", "beam", "--smoke-steps", "50"])
    monkeypatch.setattr(s23, "lobo", lambda a: a)
    assert s23.main().smoke_steps == 50
    monkeypatch.setattr(sys, "argv", ["s23.py", "lobo-summary"])
    monkeypatch.setattr(s23, "lobo_summary", lambda a: a)
    a = s23.main()
    assert a.cmd == "lobo-summary" and a.seed == 0
    assert (a.teacher, a.teacher_config) == ("rr8+mr4", "s23/teachsel/teacher_mr4.json")


def test_summary_excludes_smoke_and_serializes_index_publication(environment):
    import fcntl
    a = args()
    smoke = s23.budget_dir(s23.clone(a, smoke_steps=50))
    s23.atomic_json(smoke + "/results.json", {"deliberately": "invalid smoke result"})
    summary = s23.lobo_summary(a)
    assert all(not entry["folds"] for entry in summary["contrasts"].values())
    root = Path("s23/all_rr8/lobo")
    before = (root / "comparison_s0.json").read_bytes()
    with open(root / ".summary.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="run summaries serially"):
            s23.lobo_summary(a)
    assert (root / "comparison_s0.json").read_bytes() == before


def test_comparison_records_different_sets_even_when_unique_counts_match(environment):
    results = {arm: comparison_results("beam", arm, 10) for arm in ("yi", "jia", "jiasmall")}
    # Same counts do not imply the same identities.
    results["jia"]["selected_unique_supervision"]["teacher_pairs"]["md5"] = "different-set"
    report = s23.lobo_supervision_comparison(results)
    assert not report["equal"]
    assert report["differences_from_yi"]["jia"]["teacher_pairs"]["jia"]["md5"] == "different-set"
    assert set(report["differences_from_yi"]) == {"jia"}
    results["jia"]["plan"] = json.loads(json.dumps(results["jia"]["plan"]))
    results["jia"]["plan"]["selected_unique_supervision"]["teacher_pairs"]["md5"] = "tampered-plan"
    with pytest.raises(ValueError, match="disagrees with frozen plan"):
        s23.lobo_supervision_comparison(results)


@pytest.mark.parametrize("arm", ["yi", "jia", "jiasmall"])
def test_lobo_smoke_runs_50_steps_fixed_ten_innersel_and_isolates_all_outputs(environment, monkeypatch, arm):
    import inner_eval
    import inner_rank
    frozen = json.loads(Path("DATA_FREEZE_STAGE23.json").read_text())
    selection = {}
    for b in bs.BENCH:
        template = json.loads(Path(f"q_inner_{b}.json").read_text())[0]
        questions = [{**template, "question_id": f"inner_{b}_{i}"} for i in range(20)]
        selection[b] = [q["question_id"] for q in questions]
        frozen["bench"][b]["inner"] = selection[b]
        for name in ("inner", "innersel"):
            Path(f"q_{name}_{b}.json").write_text(json.dumps(questions))
    Path("DATA_FREEZE_STAGE23.json").write_text(json.dumps(frozen))
    Path("INNERSEL_STAGE23.json").write_text(json.dumps(selection))
    monkeypatch.setattr(inner_eval, "INNERSEL_IDENTITY_SHA256", inner_eval.selection_hash(selection))
    for name in ("yi", "jia"):
        path = Path(s23.choice_file(args(name)))
        choice = json.loads(path.read_text()); choice["data_freeze_md5"] = s23.md5("DATA_FREEZE_STAGE23.json")
        path.write_text(json.dumps(choice))
    formal = Path(s23.budget_dir(args(arm)))
    sentinels = [formal / "results.json", formal.parent / "comparison_s0.json", formal.parent / "PROGRESS.jsonl"]
    for path in sentinels:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"existing formal output\n")
    launches, evaluations = [], []
    def launch(gpus, out, command, log, master_port=None):
        assert command[command.index("--max-steps") + 1] == "50"
        assert command[command.index("--save-at") + 1] == "25,50"
        assert "/smoke50/" in out
        launches.append(out)
        Path(out, "compute_steps.jsonl").write_text("".join(json.dumps({"est_flops": i * 10}) + "\n" for i in range(1, 51)))
    def rank(tag, benches, sel, port, smoke=False, **options):
        assert sel and smoke and tag.endswith("_smoke50")
        evaluations.append(tag)
        for b in benches:
            questions = inner_eval.inner_questions(b, {b: {"inner": selection[b]}}, selmode=sel, smoke=smoke)
            assert {q["question_id"] for q in questions} == inner_eval.smoke_ids(selection[b], b)
            assert len(questions) == 10
            Path(f"ctx_{tag}_inner_{b}.jsonl").write_text("".join(json.dumps({
                "qid": q["question_id"], "budget": budget, "gold_in": 1, "gold_total": 1, "keep": [0]}) + "\n"
                for q in questions for budget in inner_eval.B))
    monkeypatch.setattr(s23, "torchrun", launch)
    monkeypatch.setattr(s23, "checkpoint_candidates", lambda out, steps: [{"path": f"{out}/s{step}", "step": step} for step in steps])
    monkeypatch.setattr(s23, "serve", lambda *a: None)
    monkeypatch.setattr(inner_rank, "rank_inner", rank)
    def cleanup(command, **kwargs):
        assert command[:3] == ["docker", "rm", "-f"]  # simulated cleanup; no subprocess is executed
    monkeypatch.setattr(s23.subprocess, "run", cleanup)
    def forbidden(*a, **kw): pytest.fail("smoke attempted formal reporting or model selection")
    for name in ("budget_comparison", "budget_question_areas", "lobo_comparison", "select", "score_candidates", "select_options"):
        monkeypatch.setattr(s23, name, forbidden)
    a = args(arm, smoke_steps=50)
    report = s23.lobo(a)
    assert len(launches) == 2 and len(evaluations) == 1
    assert report["stage2_selected"]["step"] == 25 and report["stage3_selected"]["step"] == 50
    assert report["plan"]["selected_steps"] == {"stage2": 25, "stage3": 50}
    assert report["inner_report"]["area"] == 100
    assert Path(s23.budget_dir(a), "smoke_report.json").is_file()
    for path in sentinels: assert path.read_bytes() == b"existing formal output\n"
    assert not list(Path(s23.budget_dir(a)).rglob("results.json"))
    assert not list(Path(s23.budget_dir(a)).rglob("comparison*.json"))
    assert not list(Path(s23.budget_dir(a)).rglob("PROGRESS*"))


def test_failed_smoke_never_writes_formal_progress(environment, monkeypatch):
    def fail(a): raise RuntimeError("smoke fixture failure")
    monkeypatch.setattr(s23, "preflight", fail)
    with pytest.raises(RuntimeError, match="smoke fixture failure"):
        s23.lobo(args(smoke_steps=50))
    assert not list(Path("s23").rglob("PROGRESS*"))


def test_stage_resume_and_fixed_port(environment, monkeypatch):
    a = args(gpus="4,5,6,7")
    p = s23.plan_budget(a, sample(a))
    out = s23.budget_dir(a) + "/st2"
    steps = s23.candidate_steps(p["s2_rows"])
    candidates = [{"path": f"{out}/s{step}", "step": step} for step in steps]
    calls = []
    def launch(gpus, target, command, log, master_port=None):
        calls.append(master_port)
        assert gpus == "4,5,6,7" and master_port == 29504
        Path(out, "compute_steps.jsonl").write_text("".join(json.dumps({"step": i, "est_flops": i * 10}) + "\n"
                                                                for i in range(1, steps[-1] + 1)))
    monkeypatch.setattr(s23, "torchrun", launch)
    monkeypatch.setattr(s23, "checkpoint_candidates", lambda *a: [dict(c) for c in candidates])
    first = s23.train_all_stage(a, p, 2, p["start"], out)
    assert s23.train_all_stage(a, p, 2, p["start"], out) == first
    assert calls == [29504]
