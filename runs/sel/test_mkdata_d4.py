"""CPU D4 tests plus byte regression against the untouched server mkdata.py.

Run from d4dev:
 python3 test_mkdata_d4.py --baseline ../mkdata.py --qfile env_ws4.json --pooldir pool_ws4 --ids ws4_tr.txt
A five-question fixture is extracted once; every subprocess sees only those
questions. PYTHONHASHSEED is fixed because the original mixed-state recipe
iterates sets. --unit-only runs entirely on tiny synthetic fixtures.
"""
import argparse
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from d4data import read_json, write_json, sample_mix
import countfix

HERE = Path(__file__).resolve().parent
BASELINE_SHA256 = "c0a7f30e03cabb6ffe5f0044823346bbb637f3e1c2771e9d612c70876267f447"


def run_script(script, args, cwd, env=None):
    environment = dict(os.environ, PYTHONHASHSEED="0")
    environment.pop("FMT_INST", None)
    environment.pop("MASK_DATE", None)
    if env: environment.update(env)
    result = subprocess.run([sys.executable, str(script), *map(str, args)], cwd=cwd,
                            env=environment, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(f"command failed ({result.returncode}): {script} {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


class D4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-d4-", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pooldir = self.root / "pool"
        self.pooldir.mkdir()
        self.qs = []
        self.cands = {}
        self.verdict = {}
        for n, qid in enumerate(("w10_bad", "w10_good", "w10_spare")):
            turns = []
            for i in range(81):
                turns.extend([dict(role="user", content=f"round-{i}: evidence text", kind="ev", carrier="group"),
                              dict(role="assistant", content=f"reply-{i}")])
            self.qs.append(dict(question_id=qid, question_type="aggregation", question="List five changes.",
                                question_date="2026-01-01", gold_stages=["gold"], support_stages=[],
                                haystack_sessions=[turns], haystack_dates=["2025-12-01"], scope="none", carriers=["group"]))
            pool = dict(stage=["gold"] + [f"f{i}" for i in range(1, 81)], fused=list(range(81)),
                        rr8={str(i): 1 - i / 100 for i in range(81)}, gold=[True] + [False] * 80)
            write_json(self.pooldir / (qid + ".json"), pool)
            self.cands[qid] = {"struct": {str(i): "sibling" for i in range(1, 6)}, "8b": list(range(1, 31)),
                               "ret": list(range(46, 66)), "stu": list(range(66, 76)), "rand": list(range(31, 46))}
            self.verdict[qid] = dict(support=list(range(11, 46)) if n == 0 else [11], ignore=[] if n == 0 else [12])
        write_json(self.root / "q.json", self.qs)
        write_json(self.root / "cands.json", {"q": self.cands})
        write_json(self.root / "fn.json", {"q": self.verdict})
        write_json(self.root / "qtext.json", {q["question_id"]: "What changes occurred?" for q in self.qs})
        self.table = dict(target_groups=2, groups=[dict(qid=q["question_id"], positive=0, neg_seed=i + 10)
                                                  for i, q in enumerate(self.qs[:2])],
                          spare=[dict(qid=self.qs[2]["question_id"], positive=0, neg_seed=12)])
        write_json(self.root / "table.json", self.table)

    def mkdata(self, arm, extra=()):
        out = self.root / (arm + ".jsonl")
        args = [self.root / "q.json", self.pooldir, out, "--group-table", self.root / "table.json",
                "--arm", arm, "--cands", self.root / "cands.json",
                "--shared-fn-support" if arm in ("C", "I") else "--fn-support", self.root / "fn.json"]
        if arm == "P0P1": args += ["--neg-mix"]
        if arm in ("P0", "P0P1"): args += ["--qtext", self.root / "qtext.json"]
        run_script(HERE / "mkdata.py", args + list(extra), self.root)
        return [json.loads(line) for line in out.read_text().splitlines()], read_json(str(out) + ".drops.json")

    def test_shared_drop_replacement_own_sources_and_identity(self):
        outputs = {}
        for arm in ("C", "I", "P0", "P0P1"):
            rows, report = self.mkdata(arm)
            outputs[arm] = rows
            self.assertEqual([r["qid"] for r in rows], ["w10_good", "w10_spare"])
            self.assertEqual(len(report["dropped"]), 1)
            self.assertEqual(report["dropped"][0]["failing_arms"], ["P0"])
            self.assertEqual(len(report["replacements"]), 1)
            for row in rows:
                self.assertEqual(len(row["docs"]), 16)
                self.assertEqual(len(set(row["docs"])), 16)
                self.assertEqual("neg_src" in row, arm == "P0P1")
                if arm in ("P0", "P0P1"):
                    self.assertIn("What changes occurred?", row["query"])
                    self.assertTrue(all("User: round-11:" not in d and "User: round-12:" not in d for d in row["docs"][1:]))
                else:
                    self.assertIn("List five changes.", row["query"])
                if arm == "P0":
                    for doc in row["docs"][1:]:
                        i = int(doc.split("User: round-")[1].split(":")[0])
                        self.assertIn(i, set(range(13, 46)))
                if arm == "P0P1":
                    from collections import Counter
                    self.assertEqual(Counter(row["neg_src"]), dict(struct=5, **{"8b": 3}, ret=2, stu=2, rand=3))
        identity = lambda rows: [(r["qid"], r["positive"], r["neg_seed"], r["docs"][0]) for r in rows]
        for rows in outputs.values(): self.assertEqual(identity(rows), identity(outputs["C"]))
        self.assertEqual(outputs["C"][0]["docs"], outputs["I"][0]["docs"])
        self.assertNotEqual(outputs["C"][0]["query"], outputs["I"][0]["query"])

    def test_exhausted_spare_shrinks_all_arms(self):
        self.table["spare"] = []
        write_json(self.root / "table.json", self.table)
        for arm in ("C", "I", "P0", "P0P1"):
            rows, report = self.mkdata(arm)
            self.assertEqual(len(rows), 1)
            self.assertTrue(report["spare_exhausted"])
            self.assertEqual(report["steps"], 0)

    def test_mix_ownership_and_carry(self):
        import random
        e = {"struct": {str(i): "sibling" for i in range(2)}, "8b": list(range(12)),
             "ret": list(range(12, 20)), "stu": list(range(20, 30)), "rand": list(range(30, 40))}
        negs, srcs = sample_mix(e, set(range(40)), random.Random(0))
        self.assertEqual(srcs.count("struct"), 2)
        self.assertEqual(srcs.count("8b"), 6)
        self.assertEqual(len(set(negs)), 15)
        for i, src in zip(negs, srcs):
            if i < 2: self.assertEqual(src, "struct")
        # Exhaust later sources: §12.2 still requires using earlier leftovers.
        e = {"struct": {str(i): "sibling" for i in range(20)}}
        negs, srcs = sample_mix(e, set(range(20)), random.Random(0))
        self.assertEqual(len(negs), 15)
        self.assertEqual(set(srcs), {"struct"})

    def test_standalone_opt_in(self):
        out = self.root / "mix.jsonl"
        run_script(HERE / "mkdata.py", [self.root / "q.json", self.pooldir, out, "--no-set", "--steps", 1,
                   "--neg-mix", "--cands", self.root / "cands.json", "--fn-support", self.root / "fn.json"], self.root)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row["neg_src"]), 15)
            self.assertEqual(len(row["docs"]), 16)
            self.assertNotIn("User: round-11:", "\n".join(row["docs"][1:]))

    def test_instruction_environment_and_qtext(self):
        base = [self.root / "q.json", self.pooldir]
        cli, env = self.root / "cli.jsonl", self.root / "env.jsonl"
        run_script(HERE / "mkdata.py", base + [cli, "--no-set", "--steps", 1, "--inst", "v2"], self.root)
        run_script(HERE / "mkdata.py", base + [env, "--no-set", "--steps", 1], self.root, {"FMT_INST": "2"})
        self.assertEqual(cli.read_bytes(), env.read_bytes())
        rewritten = self.root / "rewritten.jsonl"
        run_script(HERE / "mkdata.py", base + [rewritten, "--no-set", "--steps", 1, "--qtext", self.root / "qtext.json"], self.root)
        rows = [json.loads(l) for l in rewritten.read_text().splitlines()]
        self.assertTrue(all("What changes occurred?" in r["query"] for r in rows))
        # Without a candidates file, verdicts still exclude support and ignore.
        filtered = self.root / "filtered.jsonl"
        run_script(HERE / "mkdata.py", base + [filtered, "--no-set", "--steps", 1, "--fn-support", self.root / "fn.json"], self.root)
        for row in map(json.loads, filtered.read_text().splitlines()):
            indices = {int(d.split("User: round-")[1].split(":")[0]) for d in row["docs"][1:]}
            blocked = set(self.verdict[row["qid"]]["support"] + self.verdict[row["qid"]]["ignore"])
            self.assertFalse(indices & blocked)

    def test_countfix_rejection_and_failure_preserves_original(self):
        for text in ("List 6 changes", "the 6 changes", "five items", "列出六个变化", "twenty-one items"):
            self.assertTrue(countfix.count_hints(text), text)
        self.assertFalse(countfix.count_hints("What changed after 2025-01-02?"))
        # Six questions -> exactly three selected, independent of input ordering.
        qs = [dict(q, question_id=f"w10_q{i}") for i, q in enumerate(self.qs * 2)]
        write_json(self.root / "six.json", qs)
        out = self.root / "rewrites.json"
        with mock.patch.object(countfix, "chat", side_effect=[json.dumps({"question": "What changed?"}),
                               json.dumps({"question": "List five changes."}), RuntimeError("offline")]) as call:
            countfix.main(["--qfile", str(self.root / "six.json"), "--port", "1,2", "--out", str(out)])
        report = read_json(str(out) + ".report.json")
        self.assertEqual(call.call_count, 3)
        self.assertEqual(report["failed"], 2)
        self.assertEqual(report["removed"], 1)
        self.assertEqual(list(read_json(out).values()).count("List five changes."), 2)
        with mock.patch.object(countfix, "chat", side_effect=AssertionError("must never call")), contextlib.redirect_stdout(io.StringIO()) as captured:
            countfix.main(["--qfile", str(self.root / "six.json"), "--dry-run"])
        self.assertEqual(captured.getvalue().count("Original question:"), 3)

    def test_freeze_date_limit_and_insufficiency(self):
        from d4freeze import main as freeze
        qs = [dict(q, question_id=f"w10_q{i}", scope="date" if i < 2 else "none")
              for i, q in enumerate(self.qs * 2)]
        for q in qs:
            write_json(self.pooldir / (q["question_id"] + ".json"), read_json(self.pooldir / "w10_good.json"))
        write_json(self.root / "env_ws4.json", qs)
        (self.root / "pool_ws4").symlink_to(self.pooldir, target_is_directory=True)
        (self.root / "ws4_tr.txt").write_text("\n".join(q["question_id"] for q in qs))
        h4 = self.root / "ws8"
        h4.mkdir()
        for wid in range(2000, 2012):
            write_json(h4 / f"w{wid:05d}.json", dict(world=wid, name="new", team=["team"], project=str(wid)))
        out = self.root / "freeze.json"
        status = freeze(["--data-dir", str(self.root), "--worlds", "ws4", "--groups", "8", "--out", str(out)])
        self.assertEqual(status, 0)
        result = read_json(out)
        self.assertLessEqual(result["counts"]["date_fraction"], .15)
        self.assertEqual(len(result["groups"]), 8)
        self.assertEqual(len(result["spare"]), 1)
        self.assertEqual(len(result["H4"]["H4s"]), 6)
        self.assertEqual(result["H4"]["worlds"][0]["sha256"], hashlib.sha256((h4 / "w02000.json").read_bytes()).hexdigest())
        self.assertEqual(result["counts"]["group_chat_questions"], len(result["questions"]))
        status = freeze(["--data-dir", str(self.root), "--worlds", "ws4", "--groups", "1000", "--out", str(out)])
        self.assertEqual(status, 3)
        self.assertEqual(read_json(out)["status"], "insufficient")


def regression(a):
    baseline = Path(a.baseline).resolve()
    digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    if digest != BASELINE_SHA256:
        raise AssertionError(f"baseline changed: {digest}; expected untouched server mkdata.py {BASELINE_SHA256}")
    root = Path(a.workdir).resolve()
    root.mkdir(exist_ok=True)
    keep = set(Path(a.ids).read_text().split()) if a.ids else None
    qs = read_json(a.qfile)
    eligible = [q for q in qs if (keep is None or q["question_id"] in keep)
                and (Path(a.pooldir) / (q["question_id"] + ".json")).is_file()]
    # One question per type where possible, in stable source order.
    picked, types = [], set()
    for q in eligible:
        if q["question_type"] not in types:
            picked.append(q)
            types.add(q["question_type"])
        if len(picked) == 5: break
    for q in eligible:
        if len(picked) == 5: break
        if q not in picked: picked.append(q)
    if len(picked) != 5: raise AssertionError("need five real questions")
    write_json(root / "five.json", picked)
    (root / "five_ids.txt").write_text("\n".join(q["question_id"] for q in picked) + "\n")
    print("REGRESSION IDs", ",".join(q["question_id"] for q in picked), flush=True)
    print("BASELINE sha256", digest, flush=True)
    recipes = {"C": ["--no-set", "--ignore-kinds", "repeat,fu", "--hard-skip", "10", "--boost", "3", "--steps", "8"],
               "mixed": ["--steps", "5", "--p-mixed", "1", "--ignore-kinds", "repeat,fu", "--seed", "7"]}
    for name, recipe in recipes.items():
        logs = []
        for tag, script in (("old", baseline), ("new", HERE / "mkdata.py")):
            args = [root / "five.json", Path(a.pooldir).resolve(), root / f"{name}.{tag}.jsonl", *recipe]
            logs.append(run_script(script, args, root))
        old = (root / f"{name}.old.jsonl").read_bytes()
        new = (root / f"{name}.new.jsonl").read_bytes()
        if not old or old != new: raise AssertionError(f"{name}: byte regression failed ({len(old)} / {len(new)} bytes)")
        print(f"PASS {name}: byte-identical old/new; groups={len(old.splitlines())} bytes={len(old)} sha256={hashlib.sha256(old).hexdigest()}", flush=True)
        print(f"  old: {logs[0]}\n  new: {logs[1]}", flush=True)
    print("REGRESSION PASS: both recipes, all new flags OFF", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--baseline", default="../mkdata.py")
    parser.add_argument("--qfile", default="env_ws4.json")
    parser.add_argument("--pooldir", default="pool_ws4")
    parser.add_argument("--ids", default="ws4_tr.txt")
    parser.add_argument("--workdir", default="test_d4_outputs")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(D4Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): sys.exit(1)
    if not args.unit_only: regression(args)
