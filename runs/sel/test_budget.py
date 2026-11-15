"""Synthetic CPU-only budget audit; external processes and GPUs are forbidden."""
import argparse
import json
import math
from pathlib import Path
import random
import sys

import pytest

import budget_sample as bs
import s23


def question_ids():
    return {"beam": [f"c{i // 2}_probe_{i}" for i in range(20)],
            "lme": [f"q{i // 2}" + ("_abs" if i % 2 else "") for i in range(20)],
            "loc": [f"loc{i // 5}_{i}" for i in range(20)],
            "ev": [f"e01_F_MH_{i}" for i in range(20)],
            "ep": [f"p200_entities_{i}" for i in range(20)]}


def sample_inputs():
    ids = question_ids()
    questions = {b: [{"question_id": q} for q in v] for b, v in ids.items()}
    frozen = {b: {"train": v, "inner": [f"inner_{b}"]} for b, v in ids.items()}
    return questions, frozen


def row(qid, i, teacher=False):
    return json.dumps({"qid": qid, "query": "question " + qid,
                       "docs": [f"{qid}_{i}_doc{k}" for k in range(5)],
                       "teacher": [0.5] * 5, "nolabel": teacher,
                       **({"teacher_name": "rr8", "teacher_format": "logit"} if teacher else
                          {"label": 0, "nsel": 0})}) + "\n"


def args(arm="yi", frac=.1, seed=0, **extra):
    return argparse.Namespace(**{**dict(cmd="budget", arm=arm, frac=frac, seed=seed, all_data=True,
                                      teacher="rr8", teacher_config="", fold="", smoke=False,
                                      smoke_steps=0, dry_run=False, gpus="0,1,2,3", gpus2=None, r=.5,
                                      s2_groups=8000, s3_groups=8000), **extra})


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s23, "_QUESTION_IDS_CACHE", {})
    questions, frozen = sample_inputs()
    Path("DATA_FREEZE_STAGE23.json").write_text(json.dumps({"bench": frozen}))
    for b in bs.BENCH:
        Path(f"q_train_{b}.json").write_text(json.dumps(questions[b]))
        for kind, n in (("teacher", 120), ("gold", 8)):
            Path(f"{kind}_{b}.jsonl").write_text("".join(row(q["question_id"], i, kind == "teacher")
                                                               for q in questions[b] for i in range(n)))
        inner = [{"question_id": f"inner_{b}", "haystack_sessions": [[{"role": "user", "content": "gold", "has_answer": True}]],
                  "haystack_dates": ["2026-01-01"]}]
        Path(f"q_inner_{b}.json").write_text(json.dumps(inner))
    Path("tr_w20.jsonl").write_text("".join(row(f"syn{i}", i) for i in range(8000)))
    Path("tr_d3_e.jsonl").write_text(row("stage1", 0))
    for arm, level in (("yi", 8000), ("jia", 16000)):
        s23.atomic_json(s23.choice_file(args(arm)), {
            "version": "4.1b", "protocol": s23.PROTOCOL, "arm": arm, "seed": 0,
            "teacher": "rr8", "teacher_config_md5": None,
            "data_freeze_md5": s23.md5("DATA_FREEZE_STAGE23.json"), "smoke_steps": 0,
            "s2_groups": level, "s3_groups": 8000, "r": .5,
            "stage2_checkpoint": "mid", "stage3_checkpoint": "end",
            "inference": {"options": {"neighbour_context": True, "rrf_weight": .25}}})
    monkeypatch.setattr(s23, "source", lambda kind, b, *a: f"{kind}_{b}.jsonl")
    monkeypatch.setattr(s23, "s1_flops", lambda arm, seed: 10000.0 if arm == "yi" else 0.0)
    monkeypatch.setattr(s23, "preflight", lambda a: {})
    monkeypatch.setattr(s23, "budget_stage1_actual", lambda a: 10000.0 if a.arm == "yi" else 0.0)
    def flops(path, world, accum, seed, epochs=1, w_base=0):
        assert (world, accum, epochs) == (4, 4, 1)
        return len(s23.consumed_rows(path, seed, len(s23.lines(path)) // 16)) * (1 + bool(w_base) / 3)
    monkeypatch.setattr(s23, "flops", flops)
    def forbidden(*a, **kw):
        raise AssertionError("external process/GPU attempted")
    monkeypatch.setattr(s23.subprocess, "run", forbidden)
    return tmp_path


@pytest.mark.parametrize("n", [1, 20, 23])
def test_question_samples_are_exact_seeded_nested_prefixes_per_benchmark(n):
    questions, frozen = sample_inputs()
    for b in bs.BENCH:
        ids = frozen[b]["train"][:n] + [f"opaque_{b}_{i}" for i in range(20, n)]
        frozen[b]["train"] = ids
        questions[b] = [{"question_id": qid} for qid in ids]
    samples = [bs.make_sample(questions, frozen, f) for f in bs.FRACTIONS]
    for b in bs.BENCH:
        chosen = [set(s["bench"][b]["qids"]) for s in samples]
        assert chosen[0] <= chosen[1] <= chosen[2] == set(frozen[b]["train"])
        order = sorted(frozen[b]["train"])
        random.Random(1234).shuffle(order)
        for f, sample, selected in zip(bs.FRACTIONS, samples, chosen):
            assert sample["sampling_unit"] == "question" and sample["sampling_seed"] == 1234
            assert len(selected) == math.ceil(f * n)
            assert selected == set(order[:math.ceil(f * n)])
            assert sample["bench"][b]["qid_order"] == order
            assert sample["bench"][b]["realized_fraction"] == len(selected) / n
            assert sample["bench"][b]["qid_md5"] == bs.digest(sorted(selected))
            assert not any("cluster" in key for key in sample["bench"][b])
    if n == 20:
        # A single history no longer forces EverMem/EpBench to 100% at low budgets.
        for b in ("ev", "ep"):
            assert samples[0]["bench"][b]["realized_fraction"] == .1
            assert samples[1]["bench"][b]["realized_fraction"] == .25
    # Input order and training seed cannot affect the one frozen ordering.
    assert samples[0] == bs.make_sample({b: list(reversed(v)) for b, v in questions.items()}, frozen, .1)


def test_sample_freeze_rejects_tampering_and_training_drift(environment):
    root = "s23/all_rr8/budget"
    sample = bs.freeze_sample(.1, root)
    assert bs.freeze_sample(.1, root) == sample
    path = Path(root) / "budget_f0.1.json"
    modified = json.loads(path.read_text()); modified["bench"]["beam"]["qids"].pop()
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="frozen budget"):
        bs.freeze_sample(.1, root)
    questions, frozen = sample_inputs()
    questions["beam"][0]["question_id"] = "inner_beam"
    with pytest.raises(ValueError, match="training identities"):
        bs.make_sample(questions, frozen, .1)


@pytest.mark.parametrize("qid", ["", None, 1, []])
def test_sampling_rejects_invalid_question_identities(qid):
    questions, frozen = sample_inputs()
    questions["beam"][0]["question_id"] = qid
    with pytest.raises(ValueError, match="question identity"):
        bs.make_sample(questions, frozen, .1)


def test_sample_freeze_rejects_old_revision(environment):
    root = "s23/all_rr8/budget"
    sample = bs.freeze_sample(.1, root)
    sample["version"] = "budget-2026-11-06-v1"
    Path(root, "budget_f0.1.json").write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="frozen budget artifact changed"):
        bs.freeze_sample(.1, root)


@pytest.mark.parametrize("frac", bs.FRACTIONS)
def test_plans_budget_only_repeat_fill_compute_match_and_identical_small_real_bytes(environment, frac):
    sample = bs.freeze_sample(frac, "s23/all_rr8/budget")
    plans = {arm: s23.plan_budget(args(arm, frac), sample) for arm in ("yi", "jia", "jiasmall")}
    yi, jia, small = (plans[a] for a in ("yi", "jia", "jiasmall"))
    assert .95 <= jia["ratio"] <= 1.05
    assert jia["fill_teacher_per_bench"] > 0
    assert yi["ratio"] == 1
    assert small["fill_teacher_per_bench"] == 0 and not small["compute_matched"]
    assert small["planned_stage_flops"]["stage1"] == 0
    for key in ("real_teacher_s2_md5", "real_gold_s3_md5"):
        assert yi["base"][key] == small["base"][key]
    assert yi["supervision"]["groups"]["crucible"]["exposures"] == 8000
    assert small["supervision"]["groups"]["crucible"]["exposures"] == 0
    assert jia["supervision"]["groups"]["crucible"]["exposures"] == 0
    assert jia["supervision"]["groups"]["teacher"]["repeats"] > 0
    for arm, p in plans.items():
        assert p["start"] == ("/w/ckpt/d3_e" if arm == "yi" else s23.A0)
        for stage in (2, 3):
            for r in map(json.loads, s23.lines(p[f"s{stage}"])):
                if not r["qid"].startswith("syn"):
                    b = next(b for b in bs.BENCH if r["qid"] in sample["bench"][b]["qids"])
                    assert r["qid"] in sample["bench"][b]["qids"]
        s23.assert_teacher_disjoint(s23.lines(p["s2"]), s23.lines(p["s3"]))
    # Filled jia has precisely the same stage-2 identity/pair set as its frozen base.
    base = s23.lines(f"{s23.budget_dir(args('jia', frac))}/real_teacher_s2.jsonl")
    assert set(map(s23.group_identity, s23.lines(jia["s2"]))) == set(map(s23.group_identity, base))
    base_counts = bs.supervision_counts([base, s23.lines(jia["s3"])],
                                      set().union(*(set(v["qids"]) for v in sample["bench"].values())), s23.group_identity)
    assert jia["supervision"]["unique_teacher_pairs"] == base_counts["unique_teacher_pairs"]
    # Full-data gold shortages are capped; they never increase the Crucible allocation.
    if frac == 1:
        assert yi["s3_rows"] == 4000 + 20 * 8 * 5
        assert sum(yi["base"]["base_counts"]["teacher2"].values()) == 4000
        assert sum(jia["base"]["base_counts"]["teacher2"].values()) == 8000
    assert s23.plan_budget(args("jia", frac), sample) == jia


def test_frozen_plan_does_not_accept_changed_group_bytes(environment):
    a = args(); sample = bs.freeze_sample(a.frac, "s23/all_rr8/budget")
    p = s23.plan_budget(a, sample)
    Path(p["s2"]).write_text("corrupt\n")
    with pytest.raises(ValueError, match="frozen budget groups changed"):
        s23.plan_budget(a, sample)


def test_compute_matching_rejects_infeasible_plan(environment, monkeypatch):
    sample = bs.freeze_sample(.1, "s23/all_rr8/budget")
    # Jia's baseline is already over budget: additive repeat fill cannot fix it.
    original = s23.flops
    monkeypatch.setattr(s23, "flops", lambda path, *a, **kw:
                        original(path, *a, **kw) * (100 if "/jia_" in path else 1))
    with pytest.raises(ValueError, match="cannot match"):
        s23.plan_budget(args("jia"), sample)
    assert not Path(s23.budget_dir(args("jia")) + "/plan.json").exists()


def test_teacher_pair_accounting_includes_gold_and_deduplicates_candidates():
    teacher = row("real", 0, True)
    gold = json.loads(row("real", 0)); gold["docs"][-1] = "new gold teacher pair"
    syn = row("syn", 0)
    count = bs.supervision_counts([[teacher, teacher], [json.dumps(gold), syn]], {"real"}, s23.group_identity)
    assert count["unique_teacher_pairs"] == 6
    assert count["teacher_pairs"] == {"unique": 6, "exposures": 15, "repeats": 9}
    assert count["groups"]["teacher"] == {"unique": 1, "exposures": 2, "repeats": 1}
    assert count["groups"]["gold"]["unique"] == 1
    assert count["groups"]["crucible"]["exposures"] == 1


@pytest.mark.parametrize("arm,frac", [("yi", .1), ("jia", .25), ("jiasmall", 1.0)])
def test_dry_run_only_builds_plans_and_prints_counts(environment, monkeypatch, capsys, arm, frac):
    def forbidden(*a, **kw):
        pytest.fail("dry-run attempted checkpoint validation, training, evaluation or comparison")
    for name in ("preflight", "budget_stage1_actual", "stage23_gpus", "train_all_stage", "inner_score", "budget_comparison"):
        monkeypatch.setattr(s23, name, forbidden)
    a = args(arm, frac, dry_run=True, gpus="")
    p = s23.budget(a)
    output = capsys.readouterr().out
    assert "CPU-only" in output and "Docker tokenization" in output
    report = json.loads(output.splitlines()[-1])
    assert report["dry_run"] is True
    assert report["planned_group_counts"] == {
        "stage2": p["s2_rows"], "stage3": p["s3_rows"],
        "by_kind": {k: v["exposures"] for k, v in p["supervision"]["groups"].items()}}
    assert report["unique_scored_pairs"] == p["supervision"]["unique_teacher_pairs"]
    assert report["repeat_counts"] == {
        "scored_pairs": p["supervision"]["teacher_pairs"]["repeats"],
        "groups": {k: v["repeats"] for k, v in p["supervision"]["groups"].items()}}
    assert report["flops_ratio"] == p["ratio"]
    assert report["planned_flops"] == p["planned_flops"]
    assert report["planned_consumed_supervision"] == p["planned_consumed_supervision"]
    assert Path(report["plan"]).is_file()
    assert Path("s23/all_rr8/budget", f"budget_f{frac}.json").is_file()
    assert not list(Path("s23").rglob("results.json"))
    assert not list(Path("s23").rglob("PROGRESS.jsonl"))
    assert not list(Path("s23").rglob("st2"))
    assert not list(Path("s23").rglob("st3"))
    # Dry-run freezes the same plan that a subsequent execution will consume.
    assert s23.plan_budget(s23.clone(a, dry_run=False), p["sample"]) == p


def test_uncached_flops_uses_cpu_only_docker_and_reuses_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s23, "H", str(tmp_path))
    Path("groups.jsonl").write_text(row("real", 0))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[:3] == ["docker", "run", "--rm"]
        assert "--gpus" not in command
        assert command[command.index("-e") + 1] == "CUDA_VISIBLE_DEVICES="
        assert "plan_flops.py" in command and "train.py" not in command
        assert command[command.index("--params") + 1] == "595776512"
        return SimpleNamespace(returncode=0, stdout=json.dumps({"planned_flops": 1234}), stderr="")
    monkeypatch.setattr(s23.subprocess, "run", run)
    assert s23.flops("groups.jsonl", 4, 4, 0) == 1234
    assert s23.flops("groups.jsonl", 4, 4, 0) == 1234
    assert len(calls) == 1


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_execution_frozen_switches_full_inner_actual_costs_and_comparison(environment, monkeypatch, seed):
    calls = []
    def train(a, p, stage, start, out):
        calls.append((a.arm, stage, start))
        steps = s23.candidate_steps(p[f"s{stage}_rows"])
        cost = p["planned_stage_flops"][f"stage{stage}"]
        return [{"path": f"{out}/s{step}", "step": step, "actual_flops": cost * step / steps[-1]}
                for step in steps]
    def score(a, ckpt, tag, sel=True, options=None):
        assert sel is False
        assert options == {"neighbour_context": True, "rrf_weight": .25}
        assert ckpt.endswith(str(s23.candidate_steps(len(s23.lines(s23.budget_dir(a) + "/s3.jsonl")))[-1]))
        for b in bs.BENCH:
            Path(f"ctx_{tag}_inner_{b}.jsonl").write_text("".join(
                json.dumps({"qid": f"inner_{b}", "budget": budget, "gold_in": 1, "gold_total": 1, "keep": [0]}) + "\n"
                for budget in (1000, 2000, 4000, 8000, 16000)))
        return {"tag": tag, "area": 100, "per_bench": {b: 100 for b in bs.BENCH}}
    monkeypatch.setattr(s23, "train_all_stage", train)
    monkeypatch.setattr(s23, "inner_score", score)
    yi = s23.budget(args("yi", seed=seed))
    jia = s23.budget(args("jia", seed=seed))
    start = f"/w/ckpt/s23_s1_yi_s{seed}" if seed else "/w/ckpt/d3_e"
    assert calls[0] == ("yi", 2, start)
    assert yi["plan"]["start"] == start
    assert "/st2/s" in calls[1][2]
    assert calls[2] == ("jia", 2, s23.A0)
    assert yi["stage2_checkpoint"] == "mid" and yi["stage3_checkpoint"] == "end"
    assert yi["full_inner_area"] == 100
    assert all(Path(c["path"]).is_file() and "/budget/" in c["path"] for c in yi["inner_ctx"].values())
    assert yi["actual_flops"] == pytest.approx(yi["planned_flops"])
    assert yi["selected_actual_flops"] < yi["actual_flops"]
    assert yi["actual_supervision"] == yi["plan"]["planned_consumed_supervision"]
    assert jia["actual_supervision"]["groups"]["teacher"]["repeats"] > 0
    path = Path(f"s23/all_rr8/budget/comparison_f0.1_s{seed}.json")
    comparison = json.loads(path.read_text())
    assert comparison["iterations"] == 4000 and comparison["difference"] == 0
    assert comparison["ci95"] == [0, 0]
    assert s23.budget(args("yi", seed=seed)) == yi
    assert len(calls) == 4  # completed runs are validated and reused
    statuses = [json.loads(l)["status"] for l in Path("s23/all_rr8/budget/PROGRESS.jsonl").read_text().splitlines()]
    assert statuses == ["started", "completed", "started", "completed", "started", "reused"]


def test_failures_are_recorded_before_reraising(environment, monkeypatch):
    def fail(a): raise RuntimeError("fixture missing checkpoint")
    monkeypatch.setattr(s23, "preflight", fail)
    with pytest.raises(RuntimeError, match="missing checkpoint"):
        s23.budget(args())
    entries = [json.loads(l) for l in Path("s23/all_rr8/budget/PROGRESS.jsonl").read_text().splitlines()]
    assert [v["status"] for v in entries] == ["started", "failed"]
    assert entries[-1]["error_type"] == "RuntimeError"


def test_incompatible_choices_are_rejected(environment):
    path = Path(s23.choice_file(args()))
    choice = json.loads(path.read_text()); choice["s2_groups"] = 16000
    path.write_text(json.dumps(choice))
    with pytest.raises(ValueError, match="SPEC_BUDGET"):
        s23.budget_choices(args())


def test_qid_bootstrap_preserves_pairing_and_benchmark_weights():
    left = {b: {f"q{i}": 10 for i in range(1 if b == "beam" else 20)} for b in bs.BENCH}
    right = {b: {qid: 5 if b == "beam" else 10 for qid in v} for b, v in left.items()}
    result = bs.paired_bootstrap(left, right)
    assert result["difference"] == 1 and result["ci95"] == [1, 1]
    right["ep"].pop("q0")
    with pytest.raises(ValueError, match="qid mismatch"):
        bs.paired_bootstrap(left, right)


@pytest.mark.parametrize("command", ["grid", "plan", "run", "s1"])
@pytest.mark.parametrize("seed", [3, 4])
def test_other_subcommands_still_reject_extra_seeds(monkeypatch, command, seed):
    monkeypatch.setattr(sys, "argv", ["s23.py", command, "--arm", "yi", "--seed", str(seed)])
    with pytest.raises(SystemExit) as error:
        s23.main()
    assert error.value.code == 2


@pytest.mark.parametrize("frac", ["0.1", "0.25", "1.0"])
@pytest.mark.parametrize("arm", ["yi", "jia", "jiasmall"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_budget_cli_dispatches_all_arms_and_fractions(monkeypatch, frac, arm, dry_run):
    monkeypatch.setattr(sys, "argv", ["s23.py", "budget", "--arm", arm, "--frac", frac, "--seed", "4"]
                        + (["--dry-run"] if dry_run else []))
    monkeypatch.setattr(s23, "budget", lambda a: a)
    a = s23.main()
    assert a.arm == arm and a.frac == float(frac) and a.seed == 4
    assert a.dry_run is dry_run


@pytest.mark.parametrize("command", ["grid", "plan", "run", "s1"])
def test_dry_run_is_budget_only(monkeypatch, command):
    monkeypatch.setattr(sys, "argv", ["s23.py", command, "--arm", "yi", "--dry-run"])
    with pytest.raises(SystemExit) as error:
        s23.main()
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [[], ["--frac", "0.5"], ["--frac", "0.1", "--seed", "5"],
                                  ["--frac", "0.1", "--fold", "ep"], ["--frac", "0.1", "--smoke"]])
def test_budget_cli_rejects_invalid_requests(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["s23.py", "budget", "--arm", "yi"] + extra)
    with pytest.raises(SystemExit) as error:
        s23.main()
    assert error.value.code == 2


def test_budget_rejects_invalid_gpus_before_any_process(monkeypatch):
    monkeypatch.setattr(s23, "budget_choices", lambda a: pytest.fail("read choices before GPU validation"))
    with pytest.raises(ValueError, match="four distinct"):
        s23.budget(args(gpus="0,1"))


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_budget_stage1_provenance_and_actual_logs_follow_all_start(tmp_path, monkeypatch, seed):
    import preflight_s23
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s23, "H", str(tmp_path))
    calls = []
    monkeypatch.setattr(preflight_s23, "checkpoint_recipe", lambda *a: calls.append(a))
    monkeypatch.setattr(preflight_s23, "training_rows", lambda data, count: count)
    end = f"ckpt/s23_s1_yi_s{seed}" if seed else "ckpt/d3_e"
    parent = end + "_w20" if seed else "ckpt/w20"
    for name, world, count, epochs in ((parent, 4, 17784, 2), (end, 2, 1500, 1)):
        path = Path(name); path.mkdir(parents=True)
        (path / "compute.json").write_text(json.dumps({"world": world, "accum": 4,
                                                      "steps": (count // world * epochs) // 4, "est_flops": 1234}))
    assert s23.budget_stage1_actual(args(seed=seed)) == 2468
    assert calls == [(str(tmp_path / parent), seed, 4, 2, "/w/tr_w20.jsonl", s23.A0, 17784),
                     (str(tmp_path / end), seed, 2, 1, "/w/tr_d3_e.jsonl", "/w/" + parent, 1500)]
    # A complete checkpoint without measured compute must not report planned cost as actual.
    Path(end, "compute.json").unlink()
    with pytest.raises(FileNotFoundError):
        s23.budget_stage1_actual(args(seed=seed))


def test_full_inner_areas_follow_evermem_session_recall(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frozen = {b: {"inner": [f"inner_{b}"]} for b in bs.BENCH}
    Path("DATA_FREEZE_STAGE23.json").write_text(json.dumps({"bench": frozen}))
    for b in bs.BENCH:
        # EverMem: two evidence rounds in one session. Retrieving one gets full
        # session recall, whereas other benchmarks get turn recall of 1/2.
        q = {"question_id": f"inner_{b}", "haystack_sessions": [[
             {"role": "user", "content": "a", "has_answer": True},
             {"role": "user", "content": "b", "has_answer": True}]], "haystack_dates": ["date"]}
        Path(f"q_inner_{b}.json").write_text(json.dumps([q]))
        Path(f"ctx_test_inner_{b}.jsonl").write_text("".join(json.dumps({
            "qid": f"inner_{b}", "budget": budget, "gold_in": 1, "gold_total": 2, "keep": [0]}) + "\n"
            for budget in (1000, 2000, 4000, 8000, 16000)))
    result = s23.budget_question_areas({"tag": "test", "area": 60})
    assert result["ev"]["inner_ev"] == 100
    assert result["ep"]["inner_ep"] == 50
    with pytest.raises(ValueError, match="disagrees"):
        s23.budget_question_areas({"tag": "test", "area": 59})
