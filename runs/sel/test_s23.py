"""CPU-only, in-memory audit regressions: python3 -B -m unittest -v test_s23.

For original copies: python3 -B test_s23.py --baseline-dir ORIGINAL_DIR AuditRegressions.
The same regression assertions must fail on the old runner and inner_eval.
No docker, SSH, filesystem writes, model imports, or GPUs are used by these tests.
"""
import contextlib
import ast
import io
import json
import os
import shlex
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch, Mock

BASELINE = "--baseline-dir" in sys.argv
if BASELINE:
    index = sys.argv.index("--baseline-dir")
    baseline_dir = Path(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    runner = types.ModuleType("s23_before")
    runner.__file__ = str(baseline_dir / "s23.py")
    exec(compile(Path(runner.__file__).read_text(), runner.__file__, "exec"), runner.__dict__)
    INNER_CODE = (baseline_dir / "inner_eval.py").read_text()
else:
    import s23 as runner
    INNER_CODE = Path(__file__).with_name("inner_eval.py").read_text()
import inner_eval
TRAIN_AST = ast.parse(Path(__file__).with_name("train.py").read_text())


def row(qid, teacher=True, docs=None):
    docs = docs or [f"{qid}:doc{i}" for i in range(16)]
    return json.dumps({"qid": qid, "query": f"query {qid}", "docs": docs,
                       "teacher": [0.5] * len(docs), "nolabel": teacher,
                       **({} if teacher else {"label": 0, "nsel": 0})}) + "\n"


class MemoryFiles:
    """Small virtual filesystem; every write stays in a Python dict."""
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.counter = 0

    def open(self, path, mode="r", *args, **kwargs):
        path = os.fspath(path)
        if "w" in mode: self.files[path] = ""
        if path not in self.files: raise FileNotFoundError(path)
        if "b" in mode: return io.BytesIO(self.files[path].encode())
        owner = self

        class Buffer(io.StringIO):
            def write(self, value):
                result = super().write(value)
                if "w" in mode or "a" in mode: owner.files[path] = self.getvalue()
                return result

            def writelines(self, values):
                for value in values: self.write(value)

        handle = Buffer(self.files[path])
        if "a" in mode: handle.seek(0, 2)
        return handle

    def replace(self, src, dst): self.files[dst] = self.files.pop(src)

    def mkstemp(self, prefix, suffix, dir):
        self.counter += 1
        path = f"{dir}/{prefix}{self.counter}{suffix}"
        self.files[path] = ""
        return 99, path

    @contextlib.contextmanager
    def active(self):
        with contextlib.ExitStack() as stack:
            for target, replacement in [
                ("builtins.open", self.open), ("os.path.exists", lambda p: os.fspath(p) in self.files),
                ("os.path.isfile", lambda p: os.fspath(p) in self.files),
                ("os.path.isdir", lambda p: any(k.startswith(os.fspath(p) + "/") for k in self.files)),
                ("os.path.getsize", lambda p: len(self.files[os.fspath(p)].encode())),
                ("os.makedirs", lambda *a, **k: None), ("os.replace", self.replace),
                ("os.remove", lambda p: self.files.pop(p)), ("os.close", lambda fd: None),
                ("tempfile.mkstemp", self.mkstemp),
            ]:
                stack.enter_context(patch(target, replacement))
            # Any unmocked external process is a test failure.
            stack.enter_context(patch("subprocess.run", side_effect=AssertionError("external process attempted")))
            yield self


def arguments(arm="jia", **kw):
    return types.SimpleNamespace(arm=arm, fold="ep", seed=0, r=0.25, gpus="0,1,2,3", **kw)


def source_files():
    return {p: "input" for p in ["q_train_beam.json", "POOL_S23_MD5.json", "mkdata.py", "fmt.py", "k1.py", "d4data.py", "teacher_s23.py"]}


def checkpoint_files(end=5, root_end=False):
    files = {"ck/compute_steps.jsonl": "".join(json.dumps({"step": s}) + "\n" for s in range(1, end + 1))}
    for s in (end // 2, end):
        path = "ck" if root_end and s == end else f"ck/s{s}"
        files[f"{path}/train_args.json"] = json.dumps({"checkpoint_step": s, "steps_total": end, "world": 4, "accum": 4, "epochs": 1})
        files[f"{path}/config.json"] = "{}"
        files[f"{path}/model.safetensors"] = "weights"
    return files


def inner_fixture(ids, inner=None, duplicate_questions=False):
    qids = sorted(set(ids))
    qs = [{"question_id": q, "haystack_sessions": [[{"role": "user", "content": "gold", "has_answer": True}]],
           "haystack_dates": ["2026-01-01"]} for q in qids]
    if duplicate_questions: qs += qs[:1]
    return {"DATA_FREEZE_STAGE23.json": json.dumps({"bench": {"beam": {"inner": inner or ["inner"]}}}),
            "INNERSEL_STAGE23.json": json.dumps({"beam": ids}), "q_innersel_beam.json": json.dumps(qs),
            "ctx_test_inner_beam.jsonl": "".join(json.dumps({"qid": q, "budget": b, "gold_total": 1, "gold_in": 1}) + "\n"
                                                       for q in qids for b in inner_eval.B)}


def inner_cli(files):
    with files.active(), patch.object(sys, "argv", ["inner_eval.py", "test", "beam", "--sel"]):
        exec(compile(INNER_CODE, "inner_eval.py", "exec"), {"__name__": "__main__"})


class AuditRegressions(unittest.TestCase):
    def test_p1_1_topup_cycles_only_remainder_and_records_repeats(self):
        fs = MemoryFiles({"teacher": "".join(row(f"t{i}") for i in range(8)), "gold": row("g", False)})
        a = arguments()
        with fs.active(), patch.object(runner, "PER_STAGE", 16), patch.object(runner, "source", side_effect=lambda k, b, s: k):
            s2, s3 = runner.build(a, fill_teacher_per_bench=20)
            t2 = {json.loads(l)["qid"] for l in fs.files[s2].splitlines() if json.loads(l).get("nolabel")}
            t3 = {json.loads(l)["qid"] for l in fs.files[s3].splitlines() if json.loads(l).get("nolabel")}
            self.assertFalse(t2 & t3, "old offset scheme repeats source identities in stage 3")
            exposure = json.loads(fs.files[f"{runner.wd(a)}/exposures.json"])
            self.assertEqual(exposure["teacher_overlap"], 0)
            counts = exposure["group_exposures"]["teacher"]["beam"]
            self.assertEqual(sum(v["stage2"] for v in counts.values()), 24)
            self.assertEqual(sum(v["stage3"] for v in counts.values()), 3)
            self.assertTrue(any(v["repeat_count"] > 0 for v in counts.values()))

    def test_p1_2_missing_midpoint_fails_before_any_ranking(self):
        fs = MemoryFiles({k: v for k, v in checkpoint_files().items() if not k.startswith("ck/s2/")})
        with fs.active(), patch.object(runner, "rank") as rank, patch.object(
                runner.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout='{"area":42}', stderr="")):
            with self.assertRaisesRegex(ValueError, "missing midpoint"):
                runner.select(arguments(), 2, "ck", [2, 5], "0,1,2,3", None)
            rank.assert_not_called()

    def test_p1_3_unmanifested_partial_cache_is_not_reused(self):
        fs = MemoryFiles({**source_files(), "s23/src/teacher_beam_s0.jsonl": row("partial")})
        with fs.active(), patch.object(runner, "sh", side_effect=RuntimeError("pool verification invoked")):
            with self.assertRaisesRegex(RuntimeError, "pool verification invoked"):
                runner.source("teacher", "beam", 0)

    def test_p1_3_interrupted_generation_never_publishes_partial_source(self):
        fs = MemoryFiles(source_files())
        def interrupt(cmd, log=None):
            if cmd[1] == "mkdata.py":
                fs.files[cmd[4]] = row("partial")
                raise RuntimeError("generator interrupted")
        with fs.active(), patch.object(runner, "sh", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "generator interrupted"):
                runner.source("teacher", "beam", 0)
            self.assertNotIn("s23/src/teacher_beam_s0.jsonl", fs.files)
            self.assertFalse(any(k.endswith(".tmp") for k in fs.files))

    def test_p1_3_empty_cache_fails_immediately(self):
        fs = MemoryFiles({"s23/src/teacher_beam_s0.jsonl": ""})
        with fs.active(), patch.object(runner, "sh") as sh:
            with self.assertRaisesRegex(ValueError, "empty source"):
                runner.source("teacher", "beam", 0)
            sh.assert_not_called()

    def test_p1_3_empty_take_cannot_spin(self):
        with patch.object(runner, "lines", return_value=[]), patch.object(runner, "md5", return_value="x"), \
             patch.object(runner.random, "Random", side_effect=[Mock(), Mock(), RuntimeError("old infinite-loop guard")]):
            try:
                with self.assertRaisesRegex(ValueError, "empty"):
                    runner.take("empty", 1)
            except RuntimeError as exc:
                self.fail(str(exc))

    def test_p1_4_two_gpus_rejected_before_plan_or_launch(self):
        a = arguments(); a.gpus = "0,1"
        fs = MemoryFiles({"s2": row("teacher2"), "s3": row("teacher3")})
        with fs.active():
            fs.files[f"{runner.wd(a)}/plan.json"] = json.dumps({"s2": "s2", "s3": "s3", "s2_rows": 8000,
                "s3_rows": 4000, "s2_md5": runner.md5("s2"), "s3_md5": runner.md5("s3")})
        with fs.active(), patch.object(runner, "torchrun", side_effect=AssertionError("two-GPU training launched")) as launch:
            with self.assertRaisesRegex(ValueError, "four distinct"):
                runner.run(a)
            launch.assert_not_called()

    def test_p2_5_training_question_cannot_replace_inner_subset(self):
        with self.assertRaises(SystemExit) as exc:
            inner_cli(MemoryFiles(inner_fixture(["train"])))
        self.assertEqual(exc.exception.code, 2)

    def test_p2_5_duplicate_subset_fails(self):
        with self.assertRaises(SystemExit) as exc:
            inner_cli(MemoryFiles(inner_fixture(["inner", "inner"])))
        self.assertEqual(exc.exception.code, 2)

    def test_p2_5_per_benchmark_cap_enforced(self):
        ids = [f"q{i}" for i in range(101)]
        with self.assertRaises(SystemExit) as exc:
            inner_cli(MemoryFiles(inner_fixture(ids, ids)))
        self.assertEqual(exc.exception.code, 2)

    def test_p2_5_changed_in_subset_identity_hash_fails(self):
        with self.assertRaises(SystemExit) as exc:
            inner_cli(MemoryFiles(inner_fixture(["inner"], ["inner", "other"])))
        self.assertEqual(exc.exception.code, 2)


@unittest.skipIf(BASELINE, "new helper checks; baseline runs AuditRegressions only")
class IntegrityChecks(unittest.TestCase):
    def test_train_save_records_actual_step_after_weights(self):
        main = next(n for n in TRAIN_AST.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        save = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "save")
        fs = MemoryFiles(); model, tok = Mock(), Mock()
        model.state_dict.return_value = {}
        def save_model(path, **kwargs):
            fs.files[f"{path}/config.json"] = "{}"
            fs.files[f"{path}/model.safetensors"] = "weights"
            self.assertNotIn(f"{path}/train_args.json", fs.files)
        model.save_pretrained.side_effect = save_model
        ns = {"os": os, "json": json, "torch": Mock(), "model": model, "tok": tok,
              "a": types.SimpleNamespace(accum=4, epochs=1), "world": 4, "steps_total": 50}
        exec(compile(ast.Module(body=[save], type_ignores=[]), "train.save", "exec"), ns)
        with fs.active():
            ns["save"]("ck/s25", 25)
            ns["save"]("ck", 50)
        for path, step in (("ck/s25", 25), ("ck", 50)):
            args = json.loads(fs.files[f"{path}/train_args.json"])
            self.assertEqual((args["checkpoint_step"], args["steps_total"], args["world"], args["accum"]), (step, 50, 4, 4))

    def test_run_saves_two_explicit_candidates_exposures_and_isolated_smoke(self):
        for smoke in (0, 50):
            a = arguments(smoke_steps=smoke)
            fs = MemoryFiles({"s2": "".join(row(f"t{i}") for i in range(801)),
                              "s3": "".join(row(f"g{i}", False) for i in range(801))})
            d = runner.wd(a)
            with fs.active():
                fs.files[f"{d}/plan.json"] = json.dumps({"s2": "s2", "s3": "s3", "s2_rows": 801, "s3_rows": 801,
                    "s2_md5": runner.md5("s2"), "s3_md5": runner.md5("s3"), "teacher_overlap": 0, "group_exposures": {}})
            def select(a, stage, ckdir, steps, gpus, log):
                self.assertEqual(steps, [25, 50])
                return f"{ckdir}/s25", [{"step": 25, "root_as_end": False}]
            with fs.active(), patch.object(runner, "torchrun") as launch, patch.object(runner, "select", side_effect=select), \
                    patch.object(runner, "rank"), contextlib.redirect_stdout(io.StringIO()):
                runner.run(a)
                self.assertEqual(launch.call_count, 2)
                for call in launch.call_args_list:
                    args = call.args[2]
                    self.assertEqual(args[args.index("--save-at") + 1], "25,50")
                    self.assertEqual("--max-steps" in args, bool(smoke))
                result_dir = d + ("/smoke50" if smoke else "")
                result = json.loads(fs.files[f"{result_dir}/results.json"])
                self.assertEqual(result["smoke_steps"], smoke)
                self.assertEqual(sum(v["stage2"] for v in result["training_group_exposures"]["teacher"].values()), 800)
                self.assertEqual(sum(v["stage3"] for v in result["training_group_exposures"]["labelled"].values()), 800)
                self.assertEqual("_smoke50" in result["heldout_ctx"], bool(smoke))

    def test_identity_ignores_json_and_document_order(self):
        r = json.loads(row("same")); r2 = {**r, "docs": list(reversed(r["docs"]))}
        self.assertEqual(runner.group_identity(r), runner.group_identity(r2))
        with self.assertRaisesRegex(ValueError, "identity overlap"):
            runner.assert_teacher_disjoint([json.dumps(r)], [json.dumps(r2, indent=2)])

    def test_reservation_is_first_seeded_block_and_independent_of_fill(self):
        fs = MemoryFiles({"teacher": "".join(row(str(i)) for i in range(20))})
        with fs.active():
            for n in (4, 20, 200):
                s2, s3 = runner.teacher_blocks("teacher", n, 3, 7)
                self.assertEqual(s3, runner.take("teacher", 3, seed=7))
                remainder = runner.take("teacher", 20, seed=7)[3:]
                self.assertEqual(s2, [remainder[i % len(remainder)] for i in range(n)])
                self.assertEqual(len(s2), n)
                runner.assert_teacher_disjoint(s2, s3)

    def test_duplicate_content_across_reserved_boundary_is_rejected(self):
        fs = MemoryFiles({"teacher": row("same") * 4})
        with fs.active(), self.assertRaisesRegex(ValueError, "overlap"):
            runner.teacher_blocks("teacher", 5, 1, 0)

    def test_no_reservation_for_synthetic_arms(self):
        fs = MemoryFiles({"teacher": "".join(row(str(i)) for i in range(4))})
        with fs.active():
            s2, s3 = runner.teacher_blocks("teacher", 20, 0, 0)
            self.assertEqual((len(s2), len(s3)), (20, 0))

    def test_empty_or_exhausted_remainder_fails(self):
        with MemoryFiles({"teacher": row("one")}).active():
            with self.assertRaisesRegex(ValueError, "empty source/remainder"):
                runner.teacher_blocks("teacher", 10, 1, 0)
            with self.assertRaisesRegex(ValueError, "reservation exceeds"):
                runner.teacher_blocks("teacher", 10, 2, 0)

    def test_gpu_policy_and_candidate_steps(self):
        for invalid in ("0,1", "0,1,2,3,4,5,6,7", "0,0,1,2", "0,01,1,2", "0,1,2,", "0,1,2,x"):
            with self.subTest(gpus=invalid), self.assertRaises(ValueError): runner.stage23_gpus(invalid)
        self.assertEqual(runner.stage23_gpus("4,5,6,7"), ["4", "5", "6", "7"])
        self.assertEqual(runner.candidate_steps(8000), [250, 500])
        self.assertEqual(runner.candidate_steps(4000), [125, 250])
        self.assertEqual(runner.candidate_steps(80), [2, 5])
        with self.assertRaises(ValueError): runner.candidate_steps(16)

    def test_explicit_checkpoint_steps_and_root_end_record(self):
        for root in (False, True):
            with self.subTest(root=root), MemoryFiles(checkpoint_files(root_end=root)).active():
                candidates = runner.checkpoint_candidates("ck", [2, 5])
                self.assertEqual([c["recorded_step"] for c in candidates], [2, 5])
                self.assertEqual(candidates[1]["root_as_end"], root)

    def test_mislabeled_checkpoint_incomplete_log_and_missing_weights_fail(self):
        for kind in ("step", "log", "weights", "world", "accum"):
            f = checkpoint_files()
            if kind in ("step", "world", "accum"):
                a = json.loads(f["ck/s2/train_args.json"])
                a[{"step": "checkpoint_step", "world": "world", "accum": "accum"}[kind]] = 99
                f["ck/s2/train_args.json"] = json.dumps(a)
            elif kind == "log": f["ck/compute_steps.jsonl"] = '{"step":5}\n'
            else: del f["ck/s2/model.safetensors"]
            with self.subTest(kind=kind), MemoryFiles(f).active(), self.assertRaises(ValueError):
                runner.checkpoint_candidates("ck", [2, 5])

    def test_selection_ties_choose_earlier_checkpoint(self):
        with MemoryFiles(checkpoint_files(root_end=True)).active(), patch.object(runner, "rank"), \
             patch.object(runner.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout='{"area":42}', stderr="")):
            chosen, records = runner.select(arguments(), 2, "ck", [2, 5], "0,1,2,3", None)
            self.assertEqual(chosen, "ck/s2")
            self.assertTrue(records[1]["root_as_end"])

    def test_complete_manifest_reused_and_changed_inputs_rebuilt(self):
        fs = MemoryFiles(source_files()); calls = []
        def generate(cmd, log=None):
            calls.append(cmd)
            if cmd[1] == "mkdata.py":
                self.assertTrue(cmd[4].endswith(".tmp"))
                fs.files[cmd[4]] = row("gold", False)
        with fs.active(), patch.object(runner, "sh", side_effect=generate):
            p = runner.source("gold", "beam", 0)
            manifest = json.loads(fs.files[p + ".manifest.json"])
            self.assertEqual(manifest["line_count"], 1)
            self.assertEqual(manifest["source_md5"], runner.md5(p))
            runner.source("gold", "beam", 0)
            self.assertEqual(sum(c[1] == "mkdata.py" for c in calls), 1)
            fs.files["q_train_beam.json"] = "changed"
            runner.source("gold", "beam", 0)
            self.assertEqual(sum(c[1] == "mkdata.py" for c in calls), 2)
            # Pool verification is performed on the cache hit as well.
            self.assertEqual(sum(c[1] == "verify_pools.py" for c in calls), 5)

    def test_manifest_hash_count_parameters_and_fmt_must_match(self):
        for field in ("source_md5", "line_count", "parameters", "invalid_json", "FMT_INST"):
            fs = MemoryFiles(source_files()); generated = []
            def generate(cmd, log=None):
                if cmd[1] == "mkdata.py":
                    generated.append(cmd); fs.files[cmd[4]] = row("gold", False)
            with self.subTest(field=field), fs.active(), patch.object(runner, "sh", side_effect=generate):
                p = runner.source("gold", "beam", 0)
                m = json.loads(fs.files[p + ".manifest.json"])
                if field == "invalid_json": fs.files[p] = "partial json\n"
                elif field != "FMT_INST": m[field] = "wrong"; fs.files[p + ".manifest.json"] = json.dumps(m)
                with patch.dict(os.environ, {"FMT_INST": "2" if field == "FMT_INST" else "1"}):
                    runner.source("gold", "beam", 0)
                self.assertEqual(len(generated), 2)

    def test_empty_generated_source_and_partial_teacher_fail_before_publish(self):
        for content in ("", row("partial")):
            fs = MemoryFiles(source_files())
            def generate(cmd, log=None):
                if cmd[1] == "mkdata.py": fs.files[cmd[4]] = content
            with fs.active(), patch.object(runner, "sh", side_effect=generate), self.assertRaises(ValueError):
                runner.source("teacher", "beam", 0)
            self.assertNotIn("s23/src/teacher_beam_s0.jsonl", fs.files)

    def test_pool_mismatch_prevents_cache_reuse(self):
        fs = MemoryFiles(source_files())
        def generate(cmd, log=None):
            if cmd[1] == "mkdata.py": fs.files[cmd[4]] = row("gold", False)
        with fs.active(), patch.object(runner, "sh", side_effect=generate): runner.source("gold", "beam", 0)
        with fs.active(), patch.object(runner, "sh", side_effect=SystemExit("pool hash mismatch")), self.assertRaises(SystemExit):
            runner.source("gold", "beam", 0)

    def test_source_parameters_and_teacher_schema(self):
        p = runner.source_parameters("teacher", "beam", 0)
        fs = MemoryFiles({"source": row("valid") * 20000})
        with fs.active(): self.assertEqual(runner.validate_source("source", p), 20000)
        for r in ({**json.loads(row("q")), "nolabel": False}, {**json.loads(row("q")), "docs": ["only one"]}):
            with MemoryFiles({"source": (json.dumps(r) + "\n") * 20000}).active(), self.assertRaises(ValueError):
                runner.validate_source("source", p)

    def test_valid_selection_and_existing_ctx_integrity(self):
        f = inner_fixture(["inner"])
        digest = inner_eval.selection_hash({"beam": ["inner"]})
        with patch.object(inner_eval, "INNERSEL_IDENTITY_SHA256", digest):
            with MemoryFiles(f).active():
                self.assertEqual(inner_eval.evaluate("test", ["beam"], True)["area"], 100)
            for kind in ("duplicate", "missing", "unexpected", "budget", "duplicate_question"):
                broken = dict(f); key = "ctx_test_inner_beam.jsonl"
                if kind == "duplicate": broken[key] += broken[key].splitlines()[0] + "\n"
                elif kind == "missing": broken[key] = "\n".join(broken[key].splitlines()[:-1]) + "\n"
                elif kind == "duplicate_question": broken = inner_fixture(["inner"], duplicate_questions=True)
                else:
                    r = json.loads(broken[key].splitlines()[0]); r["qid" if kind == "unexpected" else "budget"] = "bad"
                    broken[key] += json.dumps(r) + "\n"
                with self.subTest(kind=kind), MemoryFiles(broken).active(), self.assertRaises((SystemExit, ValueError)):
                    inner_eval.evaluate("test", ["beam"], True)

    def test_selection_hash_independent_of_order_and_all_benchmarks_validated(self):
        self.assertEqual(inner_eval.selection_hash({"beam": ["b", "a"]}), inner_eval.selection_hash({"beam": ["a", "b"]}))
        with self.assertRaisesRegex(ValueError, "benchmark identities"):
            inner_eval.validate_selection({"beam": {"inner": ["a"]}}, {})





@unittest.skipIf(BASELINE, "v4.1b checks")
class V41bChecks(unittest.TestCase):
    def all_args(self, arm="yi", **kw):
        values = dict(arm=arm, all_data=True, fold="", seed=0, r=.25, s2_groups=8000,
                      s3_groups=4000, teacher="rr8", teacher_config="", smoke_steps=0, gpus="0,1,2,3")
        values.update(kw)
        return types.SimpleNamespace(**values)

    def frozen_files(self):
        fr = {b: {"train": [b + "_train"], "inner": [b + "_inner"]} for b in runner.BENCH}
        return {"DATA_FREEZE_STAGE23.json": json.dumps({"bench": fr}),
                **{f"q_train_{b}.json": json.dumps([{"question_id": b + "_train"}]) for b in runner.BENCH}}

    def test_all_data_exact_id_guard_and_inner_exclusion(self):
        fs = MemoryFiles(self.frozen_files())
        with fs.active():
            self.assertEqual(set(runner.validate_train_files()), set(runner.BENCH))
            for ids in (["beam_inner"], ["beam_train", "beam_train"], ["beam_train_extra"]):
                fs.files["q_train_beam.json"] = json.dumps([{"question_id": q} for q in ids])
                with self.assertRaisesRegex(ValueError, "identities"):
                    runner.validate_train_files()

    def test_teacher_probability_score10_and_logit_targets(self):
        import math
        from teacher_s23 import target, logits
        # sigmoid(0), sigmoid(log(3)) -> exp(z) ratio 1:3 at T=1.
        self.assertEqual(target([.5, .75]), [.25, .75])
        expected = [1 / (1 + math.e), math.e / (1 + math.e)]
        for actual in (target([5, 6.5], "score10"), target([0, 1], "logit")):
            for x, y in zip(actual, expected): self.assertAlmostEqual(x, y)
        self.assertTrue(all(math.isfinite(v) for v in logits([0, 1])))
        self.assertAlmostEqual(logits([1e-12])[0], math.log(1e-6 / (1 - 1e-6)))
        for values, scale in (([11], "score10"), ([-.1], "probability"), ([float("nan")], "logit")):
            with self.assertRaises(ValueError): logits(values, scale)

    def test_teacher_ensemble_standardizes_within_sampled_group(self):
        from teacher_s23 import ensemble_logits, group_logits, target
        # Different offsets and scales produce equal standardized members.
        z = ensemble_logits([[0, 2, 4], [101, 102, 103]])
        self.assertAlmostEqual(sum(z), 0)
        self.assertAlmostEqual(sum(v * v for v in z) / len(z), 1)
        self.assertEqual(ensemble_logits([[1, 1], [2, 2]]), [0, 0])
        maps = [{0: -100, 1: 0, 2: 2}, {0: 900, 1: 101, 2: 103}]
        self.assertEqual(group_logits(maps, [1, 2]), [-1, 1])
        self.assertAlmostEqual(target(z, "logit")[2], 0.7245482753, places=8)

    def test_per_teacher_key_and_file_do_not_modify_rr8(self):
        from teacher_s23 import teacher_maps, group_logits
        pool = {"rr8": {"0": .5, "1": .75}, "mr4": {"0": .75, "1": .5}}
        original = json.dumps(pool)
        fs = MemoryFiles({"cfg.json": json.dumps({"judge": {"file": "judge/{qid}.json", "scale": "score10"}}),
                          "judge/q.json": '{"0":5,"1":6.5}'})
        with fs.active():
            self.assertEqual(group_logits(teacher_maps(pool, "q", "judge", "cfg.json"), [0, 1]), [0, 1])
            self.assertEqual(group_logits(teacher_maps(pool, "q", "rr8+mr4"), [0, 1]), [0, 0])
        self.assertEqual(json.dumps(pool), original)

    def test_teacher_generation_no_gold_and_group_logits_metadata(self):
        import mkdata, random
        q = {"question_id": "q", "question": "where?", "question_date": "today", "haystack_dates": ["today"],
             "haystack_sessions": [[{"role": "user", "content": f"round {i}"} for i in range(40)]]}
        pool = {"fused": list(range(40)), "rr8": {str(i): (i + 1) / 42 for i in range(40)}}
        a = types.SimpleNamespace(pooldir="pool", P=600, n_groups=3, out="groups", teacher="rr8", teacher_config="")
        fs = MemoryFiles({"pool/q.json": json.dumps(pool)})
        with fs.active(), contextlib.redirect_stdout(io.StringIO()):
            mkdata.teacher_only(a, [q], random.Random(7)); before = fs.files["groups"]
            pool.update(gold=[True] * 40, stage=["fake"] * 40)
            for turn in q["haystack_sessions"][0]: turn["has_answer"] = True
            fs.files["pool/q.json"] = json.dumps(pool)
            mkdata.teacher_only(a, [q], random.Random(7))
            self.assertEqual(before, fs.files["groups"])
        for r in map(json.loads, before.splitlines()):
            self.assertTrue(r["nolabel"]); self.assertEqual(r["teacher_format"], "logit")
            self.assertEqual(len(r["docs"]), 16); self.assertEqual(len(set(r["docs"])), 16)

    def test_train_kl_uses_declared_format_and_preserves_nolabel_early_return(self):
        main = next(n for n in TRAIN_AST.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        kl = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "kl_teacher")
        calls = []
        class Tensor:
            def __init__(self, v, **kw): self.v = v
            def tolist(self): return self.v
        from teacher_s23 import logits
        ns = {"torch": types.SimpleNamespace(tensor=Tensor), "dev": "cpu", "teacher_logits": logits,
              "F": types.SimpleNamespace(log_softmax=lambda t, axis: t,
                  kl_div=lambda s, t, **kw: calls.append(t.v))}
        exec(compile(ast.Module(body=[kl], type_ignores=[]), "kl", "exec"), ns)
        class Student:
            def __getitem__(self, index): return self
        ns["kl_teacher"](Student(), [None, 5, 6.5], "score10")
        ns["kl_teacher"](Student(), [0, 1], "logit")
        self.assertEqual(calls, [[0, 1], [0, 1]])
        group = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "group_loss")
        guard = next(n for n in group.body if isinstance(n, ast.If) and "nolabel" in ast.unparse(n.test))
        self.assertIsInstance(guard.body[-1], ast.Return)
        self.assertIn("r.get('teacher_format', 'probability')", ast.unparse(guard))

    def test_all_data_build_counts_equal_benchmarks_and_no_crucible_jia(self):
        fs = MemoryFiles(self.frozen_files())
        fs.files["tr_w20.jsonl"] = "".join(row(f"syn{i}", False) for i in range(14000))
        for b in runner.BENCH:
            fs.files[f"t_{b}"] = "".join(row(f"{b}t{i}") for i in range(1210))
            fs.files[f"g_{b}"] = row(b + "gold", False)
        def source(kind, b, *args): return f"{'t' if kind == 'teacher' else 'g'}_{b}"
        with fs.active(), patch.object(runner, "source", side_effect=source):
            a = self.all_args()
            p2, p3 = runner.build_all(a)
            r2, r3 = ([json.loads(l) for l in fs.files[p].splitlines()] for p in (p2, p3))
            self.assertEqual((len(r2), len(r3)), (8000, 4000))
            for b in runner.BENCH:
                self.assertEqual(sum(r["qid"].startswith(b + "t") for r in r2), 800)
                self.assertEqual(sum(r["qid"] == b + "gold" for r in r3), 200)
            hashes = []
            for ratio, count in runner.S3_CELLS:
                j = self.all_args("jia", r=ratio, s3_groups=count)
                p2, p3 = runner.build_all(j, 10)
                self.assertNotIn("syn", fs.files[p2] + fs.files[p3])
                self.assertEqual(len(fs.files[p3].splitlines()), count)
                hashes.append(runner.md5(p2))
                runner.assert_teacher_disjoint(runner.lines(p2), runner.lines(p3))
            self.assertEqual(len(set(hashes)), 1)

    def test_plan_matches_each_yi_cell_including_stage1_and_reference_cost(self):
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": "freeze"})
        def build(a, fill=0):
            d = runner.wd(a); p2, p3 = d + "/s2", d + "/s3"
            fs.files[p2] = "x\n" * (a.s2_groups if a.arm == "yi" else a.s2_groups // 2 + fill * 5)
            fs.files[p3] = "x\n" * a.s3_groups
            fs.files[d + "/exposures.json"] = '{"teacher_overlap":0,"group_exposures":{}}'
            return p2, p3
        def flops(path, world, accum, seed, epochs=1, w_base=0):
            return len(fs.files[path].splitlines()) * (8 if w_base else 6)
        with fs.active(), patch.object(runner, "build_all", side_effect=build), patch.object(runner, "flops", side_effect=flops), \
             patch.object(runner, "s1_flops", side_effect=lambda arm, seed: 120000 if arm == "yi" else 0):
            for level in runner.S2_LEVELS:
                a = self.all_args("jia", s2_groups=level)
                plans = runner.plan_level(a)
                self.assertEqual(len(plans), 4)
                self.assertEqual(len({p["s2_md5"] for p in plans}), 1)
                for p in plans:
                    self.assertLessEqual(abs(p["ratio"] - 1), .05)
                    self.assertEqual(p["budget_key"], [0, level, p["r"], p["s3_groups"]])
                    self.assertGreater(p["fill_teacher_per_bench"], 0)

    def test_unmatchable_common_budget_is_rejected(self):
        fs = MemoryFiles()
        def build(a, fill=0):
            p2, p3 = runner.wd(a) + "/s2", runner.wd(a) + "/s3"
            fs.files[p2] = str(fill); fs.files[p3] = "x\n"
            return p2, p3
        def costs(path, *args):
            if "/yi_" in path: return 100 if "g34000" in path else 10000
            return 1000 + int(fs.files[path]) if path.endswith("/s2") else 0
        with fs.active(), patch.object(runner, "build_all", side_effect=build), \
             patch.object(runner, "flops", side_effect=costs), patch.object(runner, "s1_flops", return_value=0):
            with self.assertRaisesRegex(ValueError, "no shared"):
                runner.plan_level(self.all_args("jia"))

    def test_neighbour_flag_off_is_byte_identical_and_never_crosses_session(self):
        import fmt
        from inner_rank import neighbours
        args = ("date", "hello", "answer")
        expected = '<Document>: (date) [follow-up to: prev]\nUser: hello\nAssistant: answer' + fmt.SFX
        self.assertEqual(fmt.doc_str(*args, prev="prev"), expected)
        self.assertEqual(fmt.doc_str(*args, prev="prev", neighbour_context=False, neighbours=[("x", "text")]), expected)
        self.assertIn("[Adjacent rounds]", fmt.doc_str(*args, neighbour_context=True, neighbours=[("x", "text")]))
        R = [(0, "d", "first", 0), (0, "d", "second", 0), (1, "e", "private", 0)]
        self.assertEqual(neighbours(R, 1), [("d", "first")])
        self.assertEqual(neighbours(R, 2), [])

    def test_rrf_weight_zero_and_one_and_missing_score(self):
        from inner_rank import fuse_order
        P, scores = [2, 1, 0], {0: .9, 1: .8, 2: .1}
        self.assertEqual(fuse_order(P, scores, 0), [0, 1, 2])
        self.assertEqual(fuse_order(P, scores, 1), P)
        for weight in (-1, 1.1):
            with self.assertRaises(ValueError): fuse_order(P, scores, weight)
        with self.assertRaises(ValueError): fuse_order(P, {0: float("nan")}, .5)

    def test_inner_options_apply_to_a0_rr8_mr4_and_rescore_neighbours(self):
        import inner_rank
        q = {"question_id": "q", "question": "which?", "question_date": "today", "haystack_dates": ["today"],
             "haystack_sessions": [[{"role": "user", "content": "one", "has_answer": True},
                                     {"role": "user", "content": "two"}]]}
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": json.dumps({"bench": {"beam": {"inner": ["q"]}}}),
                          "q_inner_beam.json": json.dumps([q]),
                          "pool_s23/q.json": '{"fused":[0,1],"rr8":{"0":0.2,"1":0.9}}'})
        calls = []
        def post(url, body):
            calls.append(body)
            return {"results": [{"index": 0, "relevance_score": .2}, {"index": 1, "relevance_score": .9}]}
        with fs.active(), patch.object(inner_rank, "_post", side_effect=post):
            inner_rank.rank_inner("cached", ["beam"], score_key="rr8", rrf_weight=.5)
            self.assertFalse(calls)
            for style, name in (("selector", "A0"), ("official", "RR8"), ("official", "MR4")):
                inner_rank.rank_inner(name, ["beam"], port="1", style=style, neighbour_context=True, rrf_weight=.5)
                self.assertTrue(all("[Adjacent rounds]" in d for d in calls[-1]["documents"]))
                self.assertEqual(inner_eval.evaluate(name, ["beam"])["area"], 100)
            with self.assertRaisesRegex(ValueError, "fresh model"):
                inner_rank.rank_inner("bad", ["beam"], score_key="rr8", neighbour_context=True)
            fs.files["q_inner_beam.json"] = json.dumps([{**q, "question_id": "train"}])
            with self.assertRaisesRegex(ValueError, "identity"):
                inner_rank.rank_inner("bad", ["beam"], port="1")

    def test_inference_options_threshold_and_inner_weight_search(self):
        scores = iter([50, 50.9, 50.5, 52, 51, 50])
        with patch.object(runner, "inner_score", side_effect=lambda *args: {"area": next(scores)}) as score:
            result = runner.select_options(self.all_args(), "ck")
        self.assertEqual(result["options"], {"neighbour_context": False, "rrf_weight": .5})
        self.assertTrue(all(call.args[3] is False for call in score.call_args_list))
        scores = iter([50, 51, 51.9, 51.9, 51.9, 51.9])
        with patch.object(runner, "inner_score", side_effect=lambda *args: {"area": next(scores)}):
            self.assertEqual(runner.select_options(self.all_args(), "ck")["options"],
                             {"neighbour_context": True, "rrf_weight": 0})

    def test_grid_symmetric_search_full_reselect_and_seed_reuse(self):
        fs = MemoryFiles(self.frozen_files())
        train_calls, score_calls = [], []
        def plan(a):
            return [{"s2_groups": a.s2_groups, "s3_groups": n, "r": r, "planned_flops": 123}
                    for r, n in runner.S3_CELLS]
        def train(a, p, stage, start, out):
            train_calls.append((a.arm, a.seed, stage, start, p))
            return [{"path": out + f"/s{step}", "step": step, "actual_flops": step} for step in (25, 50)]
        def score(a, ck, tag, sel=True, options=None):
            score_calls.append((a.arm, sel, ck))
            # Reverse top two on full inner; this must change the final chosen checkpoint.
            value = (10 if ck.endswith("s25") else 20) if sel else (30 if "r0.5_" in ck else 10)
            return {"area": value}
        opt = {"options": {"neighbour_context": True, "rrf_weight": .25}}
        with fs.active(), patch.object(runner, "plan_level", side_effect=plan), \
             patch.object(runner, "preflight", return_value={"selection": 3000, "full": 6000, "smoke": 3000}), \
             patch.object(runner, "train_all_stage", side_effect=train), patch.object(runner, "inner_score", side_effect=score), \
             patch.object(runner, "s1_flops", return_value=100), patch.object(runner, "select_options", return_value=opt) as options, \
             contextlib.redirect_stdout(io.StringIO()):
            for arm in ("yi", "jia"):
                a = self.all_args(arm)
                before = len(train_calls)
                result = runner.grid(a)
                self.assertEqual(len(train_calls) - before, 6)  # two stage2 levels + four stage3 cells
                if arm == "jia":
                    self.assertTrue(all(c[3] == runner.A0 for c in train_calls[before:] if c[2] == 2))
                self.assertEqual(result["inference"], opt)
                self.assertIn("with_options", result["inner_report"])
                self.assertEqual(len(result["stage2_candidates"]), 4)
                self.assertEqual(len(result["stage3_candidates"]), 8)
                self.assertEqual(len(result["full_inner_finalists"]), 2)
                self.assertNotEqual(result["stage3_selected"], result["stage3_candidates"][0]["path"])
                # Every stage3 starts at the same selected stage2 checkpoint.
                self.assertEqual(len({c[3] for c in train_calls[before:] if c[2] == 3}), 1)
                frozen_bytes = fs.files[runner.choice_file(a)]
                for seed in (1, 2):
                    prior_scores, prior_train = len(score_calls), len(train_calls)
                    reused = runner.grid(self.all_args(arm, seed=seed))
                    self.assertEqual(len(score_calls), prior_scores)
                    self.assertEqual(len(train_calls) - prior_train, 2)
                    self.assertEqual(reused["s2_groups"], result["s2_groups"])
                    self.assertEqual(reused["stage3_checkpoint"], result["stage3_checkpoint"])
                    self.assertTrue(reused["reused_seed0_choice"])
                self.assertEqual(fs.files[runner.choice_file(a)], frozen_bytes)
                with self.assertRaisesRegex(ValueError, "already frozen"): runner.grid(a)
            options.assert_called_once()  # yi inference switches reused verbatim by jia

    def test_smoke_train_uses_50_steps_cumulative_costs_and_separate_choice(self):
        a = self.all_args(smoke_steps=50)
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": "freeze", "s2": "".join(row(f"t{i}") for i in range(800)),
                          "s3": "".join(row(f"g{i}", False) for i in range(800))})
        with fs.active():
            p = {"s2": "s2", "s3": "s3", "s2_rows": 800, "s3_rows": 800,
                 "s2_md5": runner.md5("s2"), "s3_md5": runner.md5("s3"),
                 "data_freeze_md5": runner.md5("DATA_FREEZE_STAGE23.json")}
            self.assertIn("/smoke50/", runner.choice_file(a))
        out = "smoke/st2"
        def launch(gpus, target, args, log):
            self.assertEqual(args[args.index("--max-steps") + 1], "50")
            self.assertEqual(args[args.index("--save-at") + 1], "25,50")
            fs.files[out + "/compute_steps.jsonl"] = "".join(json.dumps({"step": i, "est_flops": i * 100}) + "\n" for i in range(1, 51))
        candidates = [{"path": out + "/s25", "step": 25}, {"path": out + "/s50", "step": 50}]
        with fs.active(), patch.object(runner, "torchrun", side_effect=launch), \
             patch.object(runner, "checkpoint_candidates", return_value=candidates):
            result = runner.train_all_stage(a, p, 2, "start", out)
            self.assertEqual([c["actual_flops"] for c in result], [2500, 5000])
            self.assertEqual(json.loads(fs.files[out + "/training.json"])["smoke_steps"], 50)
            fs.files["s2"] += row("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.train_all_stage(a, p, 2, "start", out)

    def test_choice_integrity_and_smoke_isolation(self):
        fs = MemoryFiles(self.frozen_files())
        a = self.all_args()
        with fs.active():
            value = {"version": "4.1b", "protocol": runner.PROTOCOL, "arm": "yi", "seed": 0, "teacher": "rr8", "teacher_config_md5": None,
                     "data_freeze_md5": runner.md5("DATA_FREEZE_STAGE23.json"), "smoke_steps": 0}
            fs.files[runner.choice_file(a)] = json.dumps(value)
            runner.load_choice(a)
            fs.files["DATA_FREEZE_STAGE23.json"] += " "
            with self.assertRaisesRegex(ValueError, "incompatible"): runner.load_choice(a)
            with self.assertRaises(FileNotFoundError): runner.load_choice(self.all_args(smoke_steps=50))

    def test_grid_requires_four_gpus_before_data_or_launch(self):
        with MemoryFiles().active(), patch.object(runner, "torchrun") as launch:
            with self.assertRaisesRegex(ValueError, "four distinct"):
                runner.grid(self.all_args(gpus="0,1"))
            launch.assert_not_called()


@unittest.skipIf(BASELINE, "opt-in parallel grid")
class ParallelGridChecks(unittest.TestCase):
    def run_grid(self, parallel=False, seed=0, frozen=None, arm="yi"):
        """Exercise real grid/train/scoring orchestration; replace only external work."""
        import inner_rank
        a = V41bChecks().all_args(arm, seed=seed, **({"gpus2": "4,5,6,7"} if parallel else {}))
        fs = MemoryFiles(V41bChecks().frozen_files())
        fs.files.update(s2=row("teacher") * 64, s3=row("gold", False) * 64)
        if frozen and seed:
            fs.files[runner.choice_file(a)] = frozen
        if arm == "jia":
            fs.files[runner.choice_file(runner.clone(a, arm="yi"))] = frozen
        events, commands, assignments, tags = [], [], [], set()
        lock = threading.Lock(); active = {}; peaks = {"train": 0, "score": 0}; barriers = {}

        def begin(name, gpus, port, kind):
            with lock:
                for used, other_port, other_kind in active.values():
                    self.assertTrue(set(gpus).isdisjoint(used))
                    self.assertNotEqual(port, other_port)
                    self.assertEqual(kind, other_kind, "training and scoring phases must not overlap")
                self.assertNotIn(name, active)
                active[name] = (set(gpus), port, kind)
                peaks[kind] = max(peaks[kind], len(active))

        def finish(name):
            with lock: del active[name]

        def rendezvous(key, n):
            if not parallel or n == 1: return
            with lock: barrier = barriers.setdefault(key, threading.Barrier(n))
            barrier.wait(timeout=5)  # proves actual overlap, without launching any process

        def plan(c):
            return [{"s2_groups": c.s2_groups, "s3_groups": n, "r": r, "planned_flops": 123,
                     "s2": "s2", "s3": "s3", "s2_md5": runner.md5("s2"), "s3_md5": runner.md5("s3"),
                     "s2_rows": 64, "s3_rows": 64, "data_freeze_md5": runner.md5("DATA_FREEZE_STAGE23.json")}
                    for r, n in runner.S3_CELLS]

        def launch(command, log=None):
            parts = shlex.split(command)
            self.assertIn("/w/train.py", parts)
            name = parts[parts.index("--name") + 1]
            gpus = parts[parts.index("--gpus") + 1].strip('"').removeprefix("device=")
            port = int(parts[parts.index("--master_port") + 1])
            out = parts[parts.index("--out") + 1].removeprefix("/w/")
            train_args = parts[parts.index("/w/train.py") + 1:]
            stage = out.rsplit("/", 1)[1]
            begin(name, gpus.split(","), port, "train")
            try:
                with lock:
                    commands.append((out, train_args, log))
                    assignments.append((out, gpus))
                    events.append(("train", out))
                rendezvous(stage, 2 if not seed else 1)
                fs.files[out + "/compute_steps.jsonl"] = "".join(
                    json.dumps({"step": s, "est_flops": s * 100}) + "\n" for s in range(1, 5))
            finally: finish(name)

        def candidates(out, steps):
            return [{"path": f"{out}/s{s}", "step": s, "recorded_step": s, "root_as_end": False} for s in steps]

        def serve(g, port, ckpt):
            self.assertEqual(port, 18480 + int(g))
            begin(f"rtm-s23ev-{g}", [g], port, "score")

        def rank(tag, benches, sel, port, **options):
            with lock:
                self.assertNotIn(tag, tags)
                tags.add(tag); events.append(("score", tag))
            if "_st2_" in tag: key, n = "score2", 4
            elif "_st3_" in tag: key, n = "score3", 8
            elif "_full_" in tag: key, n = "full", 2
            elif "_rrf" in tag: key, n = "rrf", 4
            elif "options_" in tag: key, n = "neighbours", 2
            else: key, n = "report", 1
            rendezvous(key, n)

        def evaluate(tag, benches, sel, **kw):
            # Ties exercise deterministic candidate order and option tie-breaking.
            area = 50
            if tag.endswith("neighbours"): area = 51
            if "_rrf" in tag: area = 53
            return {"tag": tag, "area": area}

        def cleanup(command, **kw):
            self.assertEqual(command[:3], ["docker", "rm", "-f"])
            finish(command[3])
            return types.SimpleNamespace(returncode=0)

        with fs.active(), patch.object(runner, "plan_level", side_effect=plan), \
             patch.object(runner, "preflight", return_value={"selection": 1, "full": 1}), \
             patch.object(runner, "sh", side_effect=launch), patch.object(runner, "checkpoint_candidates", side_effect=candidates), \
             patch.object(runner, "serve", side_effect=serve), patch.object(inner_rank, "rank_inner", side_effect=rank), \
             patch.object(inner_eval, "evaluate", side_effect=evaluate), patch.object(runner.subprocess, "run", side_effect=cleanup), \
             patch.object(runner.random, "randint", return_value=37) as randint, \
             patch.object(runner, "s1_flops", return_value=100), contextlib.redirect_stdout(io.StringIO()):
            result = runner.grid(a)
            self.assertEqual(randint.call_count, 0 if parallel else len(commands))
        self.assertFalse(active)
        return result, fs.files[runner.choice_file(a)], commands, assignments, events, peaks

    def test_sequential_call_order_and_parallel_choice_and_training_equivalence(self):
        sequential, choice, calls, _, events, peaks = self.run_grid()
        root = "s23/all_rr8/yi_s0_g28000"
        expected = [("train", f"s23/all_rr8/yi_s0_g2{level}/st2") for level in runner.S2_LEVELS]
        expected += [("score", f"ALL_rr8_grid_yi_st2_{i}") for i in range(4)]
        expected += [("train", f"{root}_r{r}_g3{n}/st3") for r, n in runner.S3_CELLS]
        expected += [("score", f"ALL_rr8_grid_yi_{phase}_{i}") for phase, count in (("st3", 8), ("full", 2)) for i in range(count)]
        expected += [("score", f"ALL_rr8_options_yi_{label}") for label in ("base", "neighbours", "rrf0.25", "rrf0.5", "rrf0.75", "rrf1.0")]
        expected += [("score", "ALL_rr8_report_yi_s0")]
        self.assertEqual(events, expected)
        self.assertEqual(peaks, {"train": 1, "score": 1})
        parallel, parallel_choice, parallel_calls, assigned, _, peaks = self.run_grid(True)
        self.assertEqual(parallel, sequential)
        self.assertEqual(parallel_choice, choice)  # byte-identical final choice JSON
        self.assertEqual(sorted(parallel_calls), sorted(calls))  # actual train.py argv, outputs, logs
        self.assertEqual(peaks, {"train": 2, "score": 8})
        self.assertEqual(dict(assigned), {out: ("0,1,2,3" if i % 2 == 0 else "4,5,6,7")
                                         for i, (_, out) in enumerate(e for e in expected if e[0] == "train")})
        self.assertEqual(parallel["inference"]["options"], {"neighbour_context": True, "rrf_weight": .25})
        # Later seeds train the frozen recipe once per stage and perform no scoring.
        for seed in (1, 2):
            seq = self.run_grid(seed=seed, frozen=choice)
            par = self.run_grid(True, seed=seed, frozen=choice)
            self.assertEqual(seq[:2], par[:2])
            self.assertEqual(seq[2], par[2])
            self.assertEqual(seq[4], par[4])
        # Jia keeps yi's options and follows the same parallel training/selection path.
        seq = self.run_grid(arm="jia", frozen=choice)
        par = self.run_grid(True, arm="jia", frozen=choice)
        self.assertEqual(seq[:2], par[:2])
        self.assertEqual(sorted(seq[2]), sorted(par[2]))

    def test_second_set_validation_before_preflight_or_launch(self):
        for devices in ("", "4,5", "4,5,6,6", "4,5,6,x", "4,5,6,７", "3,4,5,6", "00,4,5,6"):
            with self.subTest(devices=devices), patch.object(runner, "preflight") as check:
                with self.assertRaises(ValueError): runner.grid(V41bChecks().all_args(gpus2=devices))
                check.assert_not_called()
        with patch.object(sys, "argv", ["s23.py", "grid", "--all-data", "--arm", "yi", "--gpus2", "4,5,6,7"]), \
             patch.object(runner, "grid") as grid:
            runner.main()
            self.assertEqual(grid.call_args.args[0].gpus2, "4,5,6,7")

    def test_parallel_failure_propagates_after_workers_finish(self):
        barrier = threading.Barrier(2); finished = []
        def job(a, index):
            barrier.wait(timeout=5)
            if index == 0: raise ValueError("failed job")
            finished.append(index)
        with self.assertRaisesRegex(ValueError, "failed job"):
            runner.gpu_map(V41bChecks().all_args(gpus2="4,5,6,7"), job, [(0,), (1,), (2,)])
        self.assertEqual(finished, [1])  # no next batch, and no worker left running

    def test_scoring_keeps_stable_ties_despite_reverse_completion_and_partial_batch(self):
        candidates = [{"path": f"ck{i}", "step": 2, "s2_groups": 8000} for i in range(11)]
        def value(a, ck, tag, sel):
            return {"area": int(ck[2:]) % 3, "tag": tag}
        with patch.object(runner, "inner_score", side_effect=value):
            expected = runner.score_candidates(V41bChecks().all_args(), candidates, "ties")
        completed = [threading.Event() for _ in candidates]; order = []
        def reverse(a, ck, tag, sel):
            i = int(ck[2:]); end = min((i // 8 + 1) * 8, len(candidates)) - 1
            if i < end: self.assertTrue(completed[i + 1].wait(timeout=5))
            order.append(i); completed[i].set()
            return value(a, ck, tag, sel)
        with patch.object(runner, "inner_score", side_effect=reverse):
            result = runner.score_candidates(V41bChecks().all_args(gpus2="4,5,6,7"), candidates, "ties")
        self.assertEqual(result, expected)
        self.assertEqual(order, list(range(7, -1, -1)) + [10, 9, 8])

    def test_default_torchrun_command_is_unchanged(self):
        with patch.object(runner, "sh") as sh, patch.object(runner.random, "randint", return_value=37):
            runner.torchrun("0,1,2,3", "ck/st2", ["--seed", "0"], "run.log")
        sh.assert_called_once_with(
            "docker run --rm --name rtm-s23-st2-0123 --gpus '\"device=0,1,2,3\"' --ipc=host --network host "
            f"-v {runner.H}:/w -v {runner.MODELS}:/models:ro --entrypoint torchrun {runner.IMG} "
            "--nproc_per_node 4 --master_port 29537 /w/train.py --seed 0 --out /w/ck/st2", log="run.log")


@unittest.skipIf(BASELINE, "post-audit fixes")
class AuditFixChecks(unittest.TestCase):
    def args(self, **kw):
        values = dict(arm="yi", all_data=True, fold="", seed=0, r=.25, s2_groups=8000,
                      s3_groups=4000, teacher="rr8", teacher_config="", smoke=False,
                      smoke_steps=0, gpus="0,1,2,3")
        values.update(kw)
        return types.SimpleNamespace(**values)

    def question(self, qid, n=30, gold=0):
        return dict(question_id=qid, question="find target", question_date="today", question_type="test",
                    haystack_dates=["today"], haystack_sessions=[[
                        dict(role="user", content=("TARGET" if i == gold else f"round {i}") + " x" * 100,
                             has_answer=i == gold) for i in range(n)]])

    def pool(self, q, depth=None):
        from k1 import rounds_of
        rounds = rounds_of(q); n = min(600, len(rounds))
        return dict(fused=list(range(n)), rr8={str(i): .5 for i in range(n if depth is None else depth)},
                    gold=[r[3] for r in rounds], stage=[None] * len(rounds))

    def fixture(self):
        import hashlib
        files = {}; frozen = {}; selection = {}; manifest = {}
        for b in runner.BENCH:
            train = [self.question(b + "_train")]
            inner = [self.question(f"{b}_inner{i}") for i in range(15)]
            frozen[b] = dict(train=[q["question_id"] for q in train], inner=[q["question_id"] for q in inner])
            selection[b] = frozen[b]["inner"][:12]
            files[f"q_train_{b}.json"] = json.dumps(train)
            files[f"q_inner_{b}.json"] = json.dumps(inner)
            files[f"q_innersel_{b}.json"] = json.dumps(inner[:12])
            for q in train + inner:
                body = json.dumps(self.pool(q)); qid = q["question_id"]
                files[f"pool_s23/{qid}.json"] = body
                manifest[qid] = hashlib.md5(body.encode()).hexdigest()
        files["DATA_FREEZE_STAGE23.json"] = json.dumps(dict(bench=frozen))
        files["INNERSEL_STAGE23.json"] = json.dumps(selection)
        files["POOL_S23_MD5.json"] = json.dumps(manifest)
        return files, frozen, selection

    def assets(self, files, path):
        files[path + "/config.json"] = '{"model_type":"qwen3"}'
        files[path + "/model.safetensors"] = "weights"
        files[path + "/tokenizer_config.json"] = "{}"
        files[path + "/tokenizer.json"] = "{}"

    def recipe(self, files, path, seed, world, epochs, data, start, rows):
        self.assets(files, path)
        steps = int((rows // world) * epochs) // 4
        args = dict(seed=seed, world=world, epochs=epochs, accum=4, lr=1e-5, max_q=4000,
                    max_d=1100, group=16, tau=1., gamma=0, shuffle_docs=1, w_bce=.2,
                    w_kl=0., w_base=.5 if world == 2 else 0., checkpoint_step=steps,
                    steps_total=steps, data=data, model=start)
        files[path + "/train_args.json"] = json.dumps(args)
        return args

    @contextlib.contextmanager
    def frozen_hashes(self, frozen, selection):
        import preflight_s23 as pf
        with patch.object(pf, "SPLIT_IDENTITY_SHA256", pf.split_hash(frozen)), \
                patch.object(inner_eval, "INNERSEL_IDENTITY_SHA256", inner_eval.selection_hash(selection)):
            yield

    def test_probability_logits_and_targets_preserve_random_endpoint_order(self):
        import math, random
        from teacher_s23 import logits, target, teacher_maps
        rng = random.Random(20261103)
        p = sorted([0., 1e-12, 1e-6, .5, 1-1e-6, 1-2**-24, 1.] +
                   [rng.random() for _ in range(1000)] + [10**rng.uniform(-16, -1) for _ in range(1000)])
        z, t = logits(p), target(p)
        self.assertTrue(all(math.isfinite(v) for v in z + t))
        self.assertTrue(all(a <= b for a, b in zip(z, z[1:])))
        self.assertTrue(all(a <= b for a, b in zip(t, t[1:])))
        for pair in ([0., 1e-12], [1-2**-24, 1.]):
            self.assertEqual(target(pair), [.5, .5])
        # mkdata's rank ordering uses these same converted maps, not a separate endpoint transform.
        maps, = teacher_maps({"rr8": {str(i): v for i, v in enumerate(p)}}, "q")
        self.assertEqual(list(maps.values()), z)

    def test_301st_gold_scored_for_students_and_every_baseline(self):
        import inner_rank
        q = self.question("q", 601, 300)
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": json.dumps({"bench": {"beam": {"inner": ["q"]}}}),
                          "q_inner_beam.json": json.dumps([q]), "pool_s23/q.json": json.dumps(self.pool(q, 300))})
        calls = []
        def post(url, body):
            calls.extend(body["documents"])
            return {"results": [{"index": i, "relevance_score": float("TARGET" in d)}
                                for i, d in enumerate(body["documents"])]}
        with fs.active(), patch.object(inner_rank, "_post", side_effect=post):
            for name, style, key in (("student", "selector", None), ("A0", "selector", None),
                                     ("RR8", "official", "rr8"), ("MR4", "official", None)):
                calls.clear()
                inner_rank.rank_inner(name, ["beam"], port="mock", style=style, score_key=key)
                result = list(map(json.loads, fs.files[f"ctx_{name}_inner_beam.jsonl"].splitlines()))
                self.assertEqual(len(calls), 600)
                self.assertTrue(any("TARGET" in d for d in calls))
                self.assertEqual(result[0]["rank"][0], 300)
                self.assertEqual(result[0]["gold_in"], 1)
                self.assertNotIn(600, result[0]["rank"])

    def test_rr8_partial_cache_requires_service_and_full_cache_can_be_reused(self):
        import inner_rank
        q = self.question("q", 601)
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": json.dumps({"bench": {"beam": {"inner": ["q"]}}}),
                          "q_inner_beam.json": json.dumps([q]), "pool_s23/q.json": json.dumps(self.pool(q, 300))})
        with fs.active(), patch.object(inner_rank, "_post") as post:
            with self.assertRaisesRegex(ValueError, "top 600"):
                inner_rank.rank_inner("bad", ["beam"], score_key="rr8")
            post.assert_not_called()
            fs.files["pool_s23/q.json"] = json.dumps(self.pool(q))
            inner_rank.rank_inner("good", ["beam"], score_key="rr8")
            post.assert_not_called()
            self.assertEqual(len(json.loads(fs.files["ctx_good_inner_beam.jsonl"].splitlines()[0])["rank"]), 600)

    def test_teacher_group_support_remains_top_300(self):
        import mkdata, random, teacher_s23
        q = self.question("q", 601)
        fs = MemoryFiles({"pool/q.json": json.dumps(self.pool(q))})
        a = types.SimpleNamespace(pooldir="pool", P=600, n_groups=2, out="groups", teacher="rr8", teacher_config="")
        with fs.active(), patch.object(teacher_s23, "group_logits", wraps=teacher_s23.group_logits) as scores, \
                contextlib.redirect_stdout(io.StringIO()):
            mkdata.teacher_only(a, [q], random.Random(7))
            self.assertTrue(all(max(call.args[1]) < 300 for call in scores.call_args_list))

    def test_missing_or_bad_inner_inputs_fail_before_grid_and_s1_training(self):
        mutations = [lambda f: f.pop("INNERSEL_STAGE23.json"),
                     lambda f: f.pop("q_inner_ev.json"),
                     lambda f: f.pop("pool_s23/ep_inner14.json"),  # outside both small and normal INNERSEL
                     lambda f: f.update({"q_innersel_beam.json": "[]"}),
                     lambda f: f.update({"pool_s23/loc_inner14.json": "{}"})]
        for mutate in mutations:
            for stage1 in (False, True):
                files, frozen, selected = self.fixture(); mutate(files)
                with self.subTest(mutation=mutate, stage1=stage1), MemoryFiles(files).active(), \
                        self.frozen_hashes(frozen, selected), patch.object(runner, "torchrun") as launch, \
                        patch.object(runner, "plan_level") as plan, patch.object(runner, "train_all_stage") as train:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        (runner.s1 if stage1 else runner.grid)(self.args())
                    launch.assert_not_called(); plan.assert_not_called(); train.assert_not_called()

    def test_preflight_checks_split_identity_and_selection_contents(self):
        import preflight_s23 as pf
        files, frozen, selection = self.fixture()
        with MemoryFiles(files).active(), self.assertRaisesRegex(ValueError, "identity hash"):
            pf.data_preflight(self.args())
        changed = json.loads(files["q_innersel_beam.json"]); changed[0]["question"] = "changed"
        files["q_innersel_beam.json"] = json.dumps(changed)
        with MemoryFiles(files).active(), self.frozen_hashes(frozen, selection), \
                self.assertRaisesRegex(ValueError, "contents differ"):
            pf.data_preflight(self.args())

    def test_preflight_validates_all_pool_indices_even_with_matching_hash(self):
        import hashlib, preflight_s23 as pf
        files, frozen, selection = self.fixture(); path = "pool_s23/ev_inner14.json"
        pool = json.loads(files[path]); pool["fused"][-1] = 30000; files[path] = json.dumps(pool)
        manifest = json.loads(files["POOL_S23_MD5.json"])
        manifest["ev_inner14"] = hashlib.md5(files[path].encode()).hexdigest()
        files["POOL_S23_MD5.json"] = json.dumps(manifest)
        with MemoryFiles(files).active(), self.frozen_hashes(frozen, selection), \
                self.assertRaisesRegex(ValueError, "top-600"):
            pf.data_preflight(self.args())

    def test_valid_preflight_counts_small_subset_and_full_inner(self):
        import preflight_s23 as pf
        files, frozen, selection = self.fixture()
        with MemoryFiles(files).active(), self.frozen_hashes(frozen, selection):
            self.assertEqual(pf.data_preflight(self.args()), dict(full=2250, selection=1800, smoke=1500))

    def test_stage1_rejects_wrong_or_duplicate_gpu_ids_before_preflight(self):
        for devices in ("0,1", "0", "0,0,1,2", "0,1,2,3,4", "x,1,2,3"):
            with self.subTest(devices=devices), MemoryFiles().active(), \
                    patch.object(runner, "preflight") as check, patch.object(runner, "torchrun") as launch:
                with self.assertRaisesRegex(ValueError, "four distinct"):
                    runner.s1(self.args(seed=1, gpus=devices))
                check.assert_not_called(); launch.assert_not_called()

    def test_stage1_uses_four_then_two_gpus_and_checks_saved_recipes(self):
        import preflight_s23 as pf
        events = []
        def launch(gpus, out, args, log):
            events.append((gpus, args))
            self.assertEqual(args[args.index("--accum") + 1], "4")
        with MemoryFiles().active(), patch.object(runner, "preflight") as check, \
                patch.object(runner, "torchrun", side_effect=launch), patch.object(pf, "checkpoint_recipe") as recipe, \
                patch.object(pf, "training_rows", side_effect=[17784, 1500]), contextlib.redirect_stdout(io.StringIO()):
            runner.s1(self.args(seed=1, gpus="4,5,6,7"))
            check.assert_called_once()
            self.assertEqual([g for g, _ in events], ["4,5,6,7", "4,5"])
            self.assertEqual([c.args[2] for c in recipe.call_args_list], [4, 2])
            self.assertEqual([c.args[3] for c in recipe.call_args_list], [2, 1])

    def test_seed1_and_seed2_checkpoint_configs_are_strict(self):
        import preflight_s23 as pf
        for seed in (1, 2):
            path = f"ckpt/s23_s1_yi_s{seed}"
            files = {}; start = f"/w/ckpt/s23_s1_yi_s{seed}_w20"
            args = self.recipe(files, path, seed, 2, 1, "/w/tr_d3_e.jsonl", start, 1500)
            with MemoryFiles(files).active():
                pf.checkpoint_recipe(path, seed, 2, 1, "/w/tr_d3_e.jsonl", start, 1500)
            for field, value in (("world", 4), ("accum", 2), ("seed", 0), ("lr", 2e-5),
                                 ("max_d", 800), ("w_base", 0), ("checkpoint_step", 50),
                                 ("steps_total", 50), ("data", "/w/wrong"), ("model", runner.A0)):
                bad = {**files, path + "/train_args.json": json.dumps({**args, field: value})}
                with self.subTest(seed=seed, field=field), MemoryFiles(bad).active(), self.assertRaises(ValueError):
                    pf.checkpoint_recipe(path, seed, 2, 1, "/w/tr_d3_e.jsonl", start, 1500)
            missing = dict(args); missing.pop("world")
            files[path + "/train_args.json"] = json.dumps(missing)
            with MemoryFiles(files).active(), self.assertRaisesRegex(ValueError, "world=None"):
                pf.checkpoint_recipe(path, seed, 2, 1, "/w/tr_d3_e.jsonl", start, 1500)

    def test_grid_rejects_bad_start_config_before_planning_or_training(self):
        import preflight_s23 as pf
        files, frozen, selected = self.fixture()
        self.assets(files, pf.host_path(runner.A0, runner.H, runner.MODELS))
        for seed in (0, 1, 2):
            end = f"/w/ckpt/s23_s1_yi_s{seed}" if seed else "/w/ckpt/d3_e"
            start = end + "_w20" if seed else "/w/ckpt/w20"
            self.recipe(files, pf.host_path(start, runner.H, runner.MODELS), seed, 4, 2, "/w/tr_w20.jsonl", runner.A0, 17784)
            args = self.recipe(files, pf.host_path(end, runner.H, runner.MODELS), seed, 2, 1, "/w/tr_d3_e.jsonl", start, 1500)
            files[pf.host_path(end, runner.H, runner.MODELS) + "/train_args.json"] = json.dumps({**args, "world": 4})
            with self.subTest(seed=seed), MemoryFiles(files).active(), self.frozen_hashes(frozen, selected), \
                    patch.object(pf, "training_rows", side_effect=[17784, 1500]), \
                    patch.object(runner, "plan_level") as plan, patch.object(runner, "train_all_stage") as train:
                with self.assertRaisesRegex(ValueError, "world=4"):
                    runner.grid(self.args(seed=seed))
                plan.assert_not_called(); train.assert_not_called()

    def test_small_smoke_runs_one_cell_two_stages_four_subset_evaluations(self):
        a = self.args(smoke=True); launches = []; evaluations = []; printed = io.StringIO()
        def plan(c):
            self.assertTrue(c.smoke); self.assertEqual(c.smoke_steps, 50)
            return [dict(r=c.r, s2_groups=c.s2_groups, s3_groups=c.s3_groups)]
        def train(c, p, stage, start, out):
            self.assertIn("ESTIMATED_CANDIDATE_PAIRS 120000", printed.getvalue())
            launches.append((stage, start))
            return [dict(path=out + f"/s{s}", step=s, actual_flops=s) for s in (25, 50)]
        def score(c, ck, tag, sel=True, options=None):
            self.assertTrue(c.smoke); self.assertTrue(sel)
            evaluations.append(ck); return {"area": 1 if ck.endswith("s25") else 2}
        with MemoryFiles().active(), patch.object(runner, "validate_train_files"), \
                patch.object(runner, "preflight", return_value=dict(selection=1, full=1, smoke=30000)), \
                patch.object(runner, "plan_level", side_effect=plan) as plans, \
                patch.object(runner, "train_all_stage", side_effect=train), \
                patch.object(runner, "inner_score", side_effect=score), \
                patch.object(runner, "select_options") as options, contextlib.redirect_stdout(printed):
            for arm in ("yi", "jia"):
                result = runner.grid(self.args(smoke=True, arm=arm))
                self.assertEqual(result["full_inner_finalists"], [])
                self.assertFalse(result["production_choice"])
            self.assertEqual(plans.call_count, 2)
            self.assertEqual([s for s, _ in launches], [2, 3, 2, 3])
            self.assertEqual(len(evaluations), 8)
            options.assert_not_called()

    def test_smoke_rank_and_eval_share_exact_seeded_ten_question_subset(self):
        import inner_rank
        files, frozen, selected = self.fixture(); calls = []
        def post(url, body):
            calls.append(body)
            return {"results": [{"index": i, "relevance_score": float("TARGET" in d)} for i, d in enumerate(body["documents"])]}
        with MemoryFiles(files).active() as fs, self.frozen_hashes(frozen, selected), patch.object(inner_rank, "_post", side_effect=post):
            inner_rank.rank_inner("small", ["beam"], sel=True, port="mock", smoke=True)
            rows = [json.loads(l) for l in fs.files["ctx_small_inner_beam.jsonl"].splitlines()]
            self.assertEqual(len(calls), 10)
            self.assertEqual({r["qid"] for r in rows}, inner_eval.smoke_ids(selected["beam"], "beam"))
            self.assertEqual(len(rows), 50)
            self.assertEqual(inner_eval.evaluate("small", ["beam"], True, smoke=True)["area"], 100)
            with self.assertRaises(SystemExit): inner_eval.evaluate("small", ["beam"], True)
            with self.assertRaisesRegex(ValueError, "requires INNERSEL"):
                inner_rank.rank_inner("bad", ["beam"], port="mock", smoke=True)

    def test_old_protocol_choices_cannot_be_reused(self):
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": "freeze"}); a = self.args()
        with fs.active():
            fs.files[runner.choice_file(a)] = json.dumps(dict(version="4.1b", arm="yi", seed=0, teacher="rr8",
                teacher_config_md5=None, data_freeze_md5=runner.md5("DATA_FREEZE_STAGE23.json"), smoke_steps=0))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                runner.load_choice(a)

    def test_smoke_plans_only_one_cell_and_cannot_overwrite_production_files(self):
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": "freeze"}); built = []
        def build(a, fill=0):
            built.append((a.arm, a.r, a.s3_groups))
            path = runner.wd(a)
            fs.files[path + "/s2"] = "x\n" * a.s2_groups
            fs.files[path + "/s3"] = "x\n" * a.s3_groups
            fs.files[path + "/exposures.json"] = '{"teacher_overlap":0,"group_exposures":{}}'
            return path + "/s2", path + "/s3"
        a = self.args(smoke=True, r=.5, s3_groups=8000)
        with fs.active(), patch.object(runner, "build_all", side_effect=build), \
                patch.object(runner, "s1_flops", return_value=0), \
                patch.object(runner, "flops", side_effect=lambda p, *args: len(fs.files[p])):
            plans = runner.plan_level(a)
            self.assertEqual(len(plans), 1)
            self.assertTrue(all((r, n) == (.5, 8000) for _, r, n in built))
            self.assertTrue(all("/smoke50/" in p[k] for p in plans for k in ("s2", "s3")))
            self.assertNotEqual(runner.wd(a), runner.wd(self.args(r=.5, s3_groups=8000)))

    def test_preflight_accepts_complete_seed_chain_and_rejects_missing_assets(self):
        import preflight_s23 as pf
        files, frozen, selected = self.fixture()
        base = pf.host_path(runner.A0, runner.H, runner.MODELS)
        self.assets(files, base)
        start, end = "/w/ckpt/s23_s1_yi_s1_w20", "/w/ckpt/s23_s1_yi_s1"
        self.recipe(files, pf.host_path(start, runner.H, runner.MODELS), 1, 4, 2, "/w/tr_w20.jsonl", runner.A0, 17784)
        self.recipe(files, pf.host_path(end, runner.H, runner.MODELS), 1, 2, 1, "/w/tr_d3_e.jsonl", start, 1500)
        with MemoryFiles(files).active() as fs, self.frozen_hashes(frozen, selected), \
                patch.object(pf, "training_rows", side_effect=lambda p, count: count):
            self.assertEqual(runner.preflight(self.args(seed=1))["smoke"], 1500)
            fs.files.pop(base + "/tokenizer.json")
            with self.assertRaises(FileNotFoundError): runner.preflight(self.args(seed=1))

    def test_partial_rr8_fresh_scoring_keeps_official_input_by_default(self):
        import inner_rank
        q = self.question("q", 301)
        fs = MemoryFiles({"DATA_FREEZE_STAGE23.json": json.dumps({"bench": {"beam": {"inner": ["q"]}}}),
                          "q_inner_beam.json": json.dumps([q]), "pool_s23/q.json": json.dumps(self.pool(q, 300))})
        calls = []
        def post(url, body):
            calls.append(body)
            return {"results": [{"index": i, "relevance_score": .5} for i in range(len(body["documents"]))]}
        with fs.active(), patch.object(inner_rank, "_post", side_effect=post):
            inner_rank.rank_inner("rr8", ["beam"], score_key="rr8", port="mock")
            self.assertTrue(all(inner_rank.OFF_INST in c["query"] for c in calls))
            self.assertTrue(all("<Selected>" not in c["query"] for c in calls))

    def test_source_cache_fingerprints_teacher_transform_for_all_paths(self):
        fs = MemoryFiles({**source_files(), "DATA_FREEZE_STAGE23.json": "freeze"})
        with fs.active():
            for all_data in (False, True):
                params = runner.source_parameters("teacher", "beam", 0, all_data=all_data)
                before = runner.source_inputs(params)
                fs.files["teacher_s23.py"] += "changed"
                after = runner.source_inputs(params)
                self.assertNotEqual(before["teacher_s23.py"], after["teacher_s23.py"])


if __name__ == "__main__":
    unittest.main()


class TestStage1Provenance(unittest.TestCase):
    def _ckpt(self, d, name, extra):
        import json, os
        p = os.path.join(d, "ckpt", name); os.makedirs(p)
        args = dict(seed=0, epochs=1.0, accum=4, lr=1e-5, max_q=4000, max_d=1100, group=16, tau=1.0, gamma=0.0,
                    shuffle_docs=1, w_bce=0.2, w_kl=0.0, w_base=0.5, data="/w/tr_d3_e.jsonl", model="/w/ckpt/w20"); args.update(extra)
        json.dump(args, open(os.path.join(p, "train_args.json"), "w")); return p

    def test_sidecar_fills_and_conflicts(self):
        import json, os, tempfile
        from unittest import mock
        import preflight_s23 as P
        with tempfile.TemporaryDirectory() as d, mock.patch.object(P, "model_assets", lambda p: None):
            prov = os.path.join(d, "prov.json")
            json.dump({"ckpt/d3_e": {"world": 2, "steps_completed": 187}}, open(prov, "w"))
            p = self._ckpt(d, "d3_e", {})
            with mock.patch.dict(os.environ, {"S23_STAGE1_PROVENANCE": prov}):
                P.checkpoint_recipe(p, 0, 2, 1, "/w/tr_d3_e.jsonl", "/w/ckpt/w20", 1500)      # fills, passes
                json.dump({"ckpt/d3_e": {"world": 2, "steps_completed": 999}}, open(prov, "w"))
                with self.assertRaises(ValueError): P.checkpoint_recipe(p, 0, 2, 1, "/w/tr_d3_e.jsonl", "/w/ckpt/w20", 1500)
                json.dump({"ckpt/d3_e": {"world": 4, "steps_completed": 187}}, open(prov, "w"))
                with self.assertRaises(ValueError): P.checkpoint_recipe(p, 0, 2, 1, "/w/tr_d3_e.jsonl", "/w/ckpt/w20", 1500)
            with self.assertRaises(ValueError): P.checkpoint_recipe(p, 0, 2, 1, "/w/tr_d3_e.jsonl", "/w/ckpt/w20", 1500)  # no sidecar: missing fields fail
            q = self._ckpt(d, "x", {"world": 4}); json.dump({"ckpt/x": {"world": 2, "steps_completed": 187}}, open(prov, "w"))
            with mock.patch.dict(os.environ, {"S23_STAGE1_PROVENANCE": prov}):
                with self.assertRaises(ValueError): P.checkpoint_recipe(q, 0, 2, 1, "/w/tr_d3_e.jsonl", "/w/ckpt/w20", 1500)  # recorded vs sidecar conflict

    def test_sidecar_w_base_only_when_missing(self):
        import json, os, tempfile
        from unittest import mock
        import preflight_s23 as P
        with tempfile.TemporaryDirectory() as d, mock.patch.object(P, "model_assets", lambda p: None):
            prov = os.path.join(d, "prov.json"); p = self._ckpt(d, "w20", {"epochs": 2.0, "data": "/w/tr_w20.jsonl", "model": "/A0"}); a = json.load(open(p + "/train_args.json")); a.pop("w_base"); json.dump(a, open(p + "/train_args.json", "w"))
            with mock.patch.dict(os.environ, {"S23_STAGE1_PROVENANCE": prov}):
                json.dump({"ckpt/w20": {"world": 4, "steps_completed": 2223}}, open(prov, "w"))
                with self.assertRaises(ValueError): P.checkpoint_recipe(p, 0, 4, 2, "/w/tr_w20.jsonl", "/A0", 17784)   # w_base missing -> fail
                json.dump({"ckpt/w20": {"world": 4, "steps_completed": 2223, "w_base": 0.0}}, open(prov, "w"))
                P.checkpoint_recipe(p, 0, 4, 2, "/w/tr_w20.jsonl", "/A0", 17784)
                json.dump({"ckpt/w20": {"world": 4, "steps_completed": 2223, "w_base": 0.5}}, open(prov, "w"))
                with self.assertRaises(ValueError): P.checkpoint_recipe(p, 0, 4, 2, "/w/tr_w20.jsonl", "/A0", 17784)   # wrong w_base still fails equality

