"""CPU-only v3 contract tests. No network, GPU, SSH, or external dependencies."""
import collections
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import fmt
from k1 import rounds_of
from pool import round_stages
import world3 as v3
from world3_adapter import groups_for, mock_pool


def legacy_plan(*args, **kwargs):
    """Keep the v2 contracts as compatibility regressions; v3 has a separate suite."""
    return v3.make_plan(*args, **kwargs, revision=2)


class World3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = v3.ModelFamilies(mock=True)
        cls.world = v3.verify(v3.render(legacy_plan(n_questions=20), cls.models), cls.models)
        cls.questions = v3.build_questions(cls.world)

    def test_ledger_state_machine_and_snapshots(self):
        self.assertTrue(v3.validate_ledger(self.world["facts"]))
        case = self.world["cases"][0]
        first = next(f for f in self.world["facts"] if f["id"] == case["targets"][0])
        request = {k: first[k] for k in ("holder", "scope", "entity", "attr")}
        self.assertIsNone(v3.resolve(self.world["facts"], request, "2030-01-01", "2020-01-01"))
        illegal = copy.deepcopy(self.world["facts"])
        illegal[0]["status"] = "executed"
        with self.assertRaisesRegex(ValueError, "illegal transition"):
            v3.validate_ledger(illegal)
        illegal = copy.deepcopy(self.world["facts"])
        illegal[1]["holder"] = "Other"
        with self.assertRaisesRegex(ValueError, "changed slot"):
            v3.validate_ledger(illegal)
        illegal = copy.deepcopy(self.world["facts"])
        illegal[0]["record_time"] = "2020-01-01"
        with self.assertRaisesRegex(ValueError, "record precedes"):
            v3.validate_ledger(illegal)

    def test_veto_and_rollback_restore_old_version(self):
        w = dict(facts=[])
        case = dict(id="state", offset=30, record_time=v3.day(30))
        old = v3.add_version(w, case, "A", "personal", "budget", "limit", "100", when=0)[-1]
        new = v3.add_version(w, case, "A", "personal", "budget", "limit", "120", when=5,
                             statuses=("proposed", "approved", "executed", "rolled_back"), supersedes=old)
        v3.add_version(w, case, "A", "personal", "budget", "limit", "200", when=10,
                       statuses=("proposed", "vetoed"), supersedes=old)
        self.assertTrue(v3.validate_ledger(w["facts"]))
        req = dict(holder="A", scope="personal", entity="budget", attr="limit")
        self.assertEqual(v3.resolve(w["facts"], req, v3.day(5), v3.day(30))["id"], old)
        self.assertEqual(v3.resolve(w["facts"], req, v3.day(6), v3.day(30))["value"], "120")
        self.assertEqual(v3.resolve(w["facts"], req, v3.day(8), v3.day(30))["id"], old)
        self.assertEqual(v3.resolve(w["facts"], req, v3.day(20), v3.day(30))["id"], old)
        self.assertEqual(w["facts"][-1]["status"], "vetoed")
        self.assertNotEqual(old, new[-1])

    def test_holder_and_team_scope_are_independent(self):
        w = dict(facts=[])
        case = dict(id="scope", offset=30, record_time=v3.day(30))
        v3.add_version(w, case, "A", "personal", "report", "format", "plain text", when=0)
        v3.add_version(w, case, "Team", "team", "report", "format", "PDF", when=0)
        v3.add_version(w, case, "B", "personal", "report", "format", "slides", when=0)
        query = dict(operation="direct", entity="report", as_of=v3.day(20), known_at=v3.day(30),
                     requests=[dict(holder="A", scope="personal", entity="report", attr="format")])
        self.assertIn("plain text", v3.solve(w, query)["answer"])
        query["requests"][0]["holder"] = "B"
        self.assertIn("slides", v3.solve(w, query)["answer"])
        query["requests"][0].update(holder="A", scope="team")
        with self.assertRaisesRegex(ValueError, "unanswerable"):
            v3.solve(w, query)
        query["requests"][0].update(holder="Team", scope="team")
        self.assertIn("PDF", v3.solve(w, query)["answer"])

    def test_answers_and_necessary_atoms_are_programmatic(self):
        for q in self.questions:
            computed = v3.solve(self.world, q["query_spec"])
            self.assertEqual(computed["answer"], q["answer"])
            self.assertEqual(computed["atom_ids"], q["gold_stages"])
            self.assertEqual(len(q["answer_atoms"]), next(c["expected_evidence"] for c in self.world["cases"] if c["id"] == q["case_id"]))
            self.assertTrue(all(a["equivalent_rounds"] for a in q["answer_atoms"]))
            for atom in q["answer_atoms"]:
                for i in atom["equivalent_rounds"]:
                    self.assertIn(atom["statement"], v3.visible_documents(q)[i])
        for q in self.questions:
            if q["question_type"] == "multihop":
                rounds = rounds_of(q)
                route = {rounds[i][0] for i in q["answer_atoms"][0]["equivalent_rounds"]}
                location = {rounds[i][0] for i in q["answer_atoms"][-1]["equivalent_rounds"]}
                self.assertFalse(route & location)
        self.assertEqual({f["status"] for f in self.world["facts"]}, set(v3.TRANSITIONS) - {None})
        update = next(q for q in self.questions if q["question_type"] == "update")
        self.assertTrue(any("rolled_back" in a["statement"] for a in update["answer_atoms"]))
        self.assertTrue(any("Rollback" in q["answer"] for q in self.questions))

    def test_quotas_and_actual_evidence_histogram(self):
        for field, expected in (("carrier", v3.CARRIERS), ("question_type", v3.QTYPES)):
            counts = collections.Counter(q[field] for q in self.questions)
            for key, share in expected.items():
                self.assertLessEqual(abs(counts[key] / len(self.questions) - share), .05)
        bins = collections.Counter("one" if len(q["answer_atoms"]) == 1 else "two" if len(q["answer_atoms"]) == 2 else "three_five" if len(q["answer_atoms"]) <= 5 else "six_eight" for q in self.questions)
        self.assertEqual(dict(bins), v3.allocation(20, v3.EVIDENCE))
        ua = [q for q in self.questions if q["carrier"] == "assistant"]
        self.assertLessEqual(abs(sum(q["assistant_only"] for q in ua) / len(ua) - .25), 1 / len(ua))
        for carrier in ("group", "dyad"):
            counts = collections.Counter(q["perspective"] for q in self.questions if q["carrier"] == carrier)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        # More plans test stable rounding across seeds without allocating huge
        # duplicated JSON histories merely to count deterministic assignments.
        for seed in range(3):
            plan = legacy_plan(n_questions=100, seed=seed)
            counts = collections.Counter(c["qtype"] for c in plan["cases"])
            self.assertEqual(counts, v3.allocation(100, v3.QTYPES))
            actual = collections.Counter(len(v3.solve(plan, c["query"])["atom_ids"]) for c in plan["cases"])
            self.assertEqual(actual[1], 40)
            self.assertEqual(actual[2], 30)
            self.assertEqual(sum(v for k, v in actual.items() if 3 <= k <= 5), 25)
            self.assertEqual(sum(v for k, v in actual.items() if 6 <= k <= 8), 5)

    def test_carriers_and_continuous_interleaved_discussions(self):
        no_new = sum(not m["new_fact"] for m in self.world["rounds"]) / len(self.world["rounds"])
        self.assertTrue(.20 <= no_new <= .30)
        for case in self.world["cases"]:
            ms = [m for m in self.world["rounds"] if m["case"] == case["id"]]
            blocks = collections.Counter(m["local_block"] for m in ms)
            self.assertTrue(all(3 <= n <= 8 for n in blocks.values()))
            if case["carrier"] == "group":
                self.assertGreaterEqual(len({m["speaker"] for m in ms}), 3)
                channels = collections.defaultdict(set)
                for m in ms:
                    channels[m["speaker"]].add(m["channel"])
                self.assertTrue(any(len(ch) >= 2 for ch in channels.values()))
                self.assertNotEqual(ms[0]["thread"], ms[1]["thread"])
                self.assertEqual(ms[0]["thread"], ms[2]["thread"])
            if case["carrier"] == "record":
                self.assertTrue(all(f["event_time"] != f["record_time"] for f in self.world["facts"] if f["source"]["case"] == case["id"] and f["id"] in case["targets"]))
            if case["carrier"] == "dyad":
                self.assertEqual({m["speaker"] for m in ms}, set(self.world["team"][:2]))

    def test_no_list_n_date_windows_and_inside_distractors(self):
        self.assertEqual(sum(bool(q["date_window"]) for q in self.questions), 3)
        for q in self.questions:
            self.assertNotRegex(q["question"].lower(), r"\blist\s+\d+")
            if q["date_window"]:
                self.assertTrue(q["window_hard_negatives"])
                for i in q["window_hard_negatives"]:
                    date = v3.dt.datetime.strptime(rounds_of(q)[i][1], "%B-%d-%Y").date().isoformat()
                    self.assertTrue(q["date_window"][0] <= date <= q["date_window"][1])

    def test_exact_negative_quotas_no_false_negatives(self):
        for q in self.questions:
            neg, labels, deficit = v3.sample_negatives(q["negative_candidates"], random.Random(0))
            self.assertEqual(len(set(neg)), len(neg))
            self.assertFalse(any(deficit.values()), (q["question_id"], deficit))
            self.assertEqual(collections.Counter(labels), v3.NEG_QUOTAS)
            protected = {i for a in q["answer_atoms"] for i in a["equivalent_rounds"]} | set(q["support_rounds"])
            self.assertFalse(set(neg) & protected)
        empty = {k: [] for k in v3.NEG_QUOTAS}
        empty["random"] = list(range(100))
        neg, labels, deficit = v3.sample_negatives(empty, random.Random(0))
        self.assertEqual(labels, ["random", "random"])
        self.assertEqual(deficit["minimal_state_diff"], 3)
        self.assertEqual(deficit["random"], 0)

    def test_real_formatter_visibility_and_truncation(self):
        for q in self.questions:
            self.assertTrue(v3.check_visibility(self.world, q)["ok"])
            R = rounds_of(q)
            prev = fmt.round_prev(q)
            self.assertEqual(v3.visible_documents(q), [fmt.doc_str(r[1], *fmt.split_round(r[2]), prev=prev[i]) for i, r in enumerate(R)])
        q = self.questions[0]
        self.assertFalse(v3.check_visibility(self.world, q, max_chars=10)["ok"])
        with self.assertRaisesRegex(ValueError, "actual training tokenizer"):
            v3.visible_documents(q, token_limit=10)
        class CharacterTokenizer:
            def encode(self, text, **kwargs): return list(text)
            def decode(self, tokens): return "".join(tokens)
        self.assertFalse(v3.check_visibility(self.world, q, token_limit=20, tokenizer=CharacterTokenizer())["ok"])

    def test_short_ack_carrying_context_is_not_a_negative(self):
        q = copy.deepcopy(self.questions[0])
        atom = q["answer_atoms"][0]
        # A deliberately short preceding message makes its entire sufficient
        # clause visible in the real formatter's previous-user context.
        text = "Ari personal widget finish=cedar was executed."
        fid = atom["id"]
        w = copy.deepcopy(self.world)
        f = next(f for f in w["facts"] if f["id"] == fid)
        f["statement"] = text
        q["haystack_sessions"] = [[dict(role="user", content=text, stage=fid, speaker="Ari", atoms=[fid]),
                                    dict(role="user", content="Thanks!", stage="ack", speaker="Bela", atoms=[])]]
        q["haystack_dates"] = ["January-20-2025"]
        q["answer_atoms"] = [dict(id=fid, statement=text, equivalent_rounds=[0, 1])]
        q["support_rounds"] = []
        self.assertEqual(v3.evidence_map(w, q, [fid])[fid], [0, 1])
        categories, _, protected = v3.classify_negatives(w, q)
        self.assertNotIn(1, categories["contentless_ack"])
        self.assertIn(1, protected)

    def test_assistant_unique_fact_visible_on_assistant_side(self):
        qs = [q for q in self.questions if q["assistant_only"]]
        self.assertTrue(qs)
        for q in qs:
            for a in q["answer_atoms"]:
                for i in a["equivalent_rounds"]:
                    user, assistant = fmt.split_round(rounds_of(q)[i][2])
                    self.assertNotIn(a["statement"], user)
                    self.assertIn(a["statement"], assistant)

    def test_two_family_semantic_rejection_quarantines(self):
        class Reject(v3.MockPool):
            def chat(self, prompt, **kwargs):
                if '"task": "verify"' in prompt:
                    return json.dumps(dict(ok=False, no_extra_facts=False)), "reject"
                return super().chat(prompt, **kwargs)
        models = v3.ModelFamilies(pools=[Reject(), Reject(), Reject()], mock=True)
        rejected = v3.verify(self.world, models)
        self.assertEqual(v3.build_questions(rejected), [])
        self.assertEqual(v3.build_questions(self.world)[0]["split"], self.world["split"])
        self.assertEqual({i for i in range(3) if any(k.startswith(f"family{i}:render") for k in self.models.calls)}, {0, 1, 2})
        self.assertTrue(all(f["source"].get("message_ids") is not None for f in self.world["facts"]))

    def test_pool_and_mkdata_contract_and_adapter(self):
        for q in self.questions:
            self.assertEqual(len(round_stages(q)), len(rounds_of(q)))
            gs, report = groups_for(q, mock_pool(q), no_set=True, p=100000)
            self.assertEqual(len(gs), 1, report)
            self.assertEqual(len(gs[0]["docs"]), 16)
            self.assertEqual(collections.Counter(gs[0]["neg_src"]), v3.NEG_QUOTAS)
            self.assertEqual(gs[0]["nsel"], 0)
        # Execute the real stock mkdata on a tiny compatible pool. Offline,
        # standard library only, no GPU and no modification of mkdata.py.
        q = self.questions[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            v3.write(path / "q.json", [q])
            pool = mock_pool(q)
            pool.update(stage=round_stages(q), gold=[r[3] for r in rounds_of(q)])
            pool["rr8"] = {str(i): float(-i) for i in pool["fused"]}
            v3.write(path / (q["question_id"] + ".json"), pool)
            out = path / "groups.jsonl"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("mkdata.py")),
                                     str(path / "q.json"), str(path), str(out), "--no-set", "--steps", "1"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("groups 1", result.stdout)
            self.assertEqual(len(json.loads(out.read_text())["docs"]), 16)

    def test_gates_use_visible_context_and_delete_all_equivalents(self):
        gs = v3.gates(self.world, self.questions[:3], v3.ModelFamilies(mock=True))
        self.assertTrue(all(g["suff"] == 1 and all(g["atom_deletion"].values()) for g in gs))
        self.assertTrue(all(g["mock"] for g in gs))
        broken = copy.deepcopy(self.questions[0])
        for sess in broken["haystack_sessions"]:
            for t in sess:
                t["content"] = "nothing remains"
        broken["visibility"] = v3.check_visibility(self.world, broken)
        self.assertEqual(v3.gates(self.world, [broken], v3.ModelFamilies(mock=True))[0]["suff"], 0)


if __name__ == "__main__":
    unittest.main()
