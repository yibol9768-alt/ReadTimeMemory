"""CPU-only paired intervention contracts; all LLM calls use MockPool."""
import collections
import copy
import unittest

import pairs3
import world3 as v3


class Pairs3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = v3.ModelFamilies(mock=True)
        cls.base = v3.verify(v3.render(pairs3.diagnostic_plan(), cls.models), cls.models)
        cls.left = v3.build_questions(cls.base)[0]
        cls.questions, cls.pairs, cls.bases = pairs3.generate_pairs(140, models=cls.models)

    def test_pair_counts_eight_balanced_interventions_and_audit_split(self):
        self.assertEqual(len(self.pairs), 140)
        self.assertEqual(len(self.questions), 280)
        counts = collections.Counter(p["intervention"] for p in self.pairs)
        self.assertEqual(set(counts), set(pairs3.PRESERVE + pairs3.CHANGE))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(len(counts), 8)
        self.assertEqual(len(self.bases), 10)
        self.assertEqual({q["split"] for q in self.questions}, {"audit"})
        with self.assertRaises(ValueError):
            pairs3.generate_pairs(119, models=self.models)
        with self.assertRaises(ValueError):
            pairs3.generate_pairs(201, models=self.models)

    def test_preserving_pairs_keep_all_labels_and_equivalent_positions(self):
        byid = {q["question_id"]: q for q in self.questions}
        for pair in self.pairs:
            if pair["expected"] != "preserve":
                continue
            left, right = byid[pair["left"]], byid[pair["right"]]
            self.assertEqual(pair["left_atoms"], pair["right_atoms"])
            self.assertEqual(left["round_atoms"], right["round_atoms"])
            self.assertEqual(left["ledger_id"], right["ledger_id"])
            if pair["intervention"] != "rename":
                self.assertEqual(left["answer"], right["answer"])
            else:
                self.assertEqual(pairs3.rename_text(left["answer"], pair["identity_bijection"]), right["answer"])

    def test_changing_pairs_recompute_answer_and_gold(self):
        byid = {q["question_id"]: q for q in self.questions}
        byworld = {w["world"]: w for w in self.bases}
        for pair in self.pairs:
            if pair["expected"] != "change":
                continue
            spec = pair["intervention_spec"]
            variant = pairs3.intervene(byworld[spec["base_world"]], spec["kind"], spec["serial"])
            right = byid[pair["right"]]
            computed = v3.solve(variant, right["query_spec"])
            self.assertEqual(computed["answer"], right["answer"])
            self.assertEqual(set(computed["atom_ids"]), set(pair["right_atoms"]))
            self.assertNotEqual(pair["left_atoms"], pair["right_atoms"])
            self.assertNotEqual(byid[pair["left"]]["answer"], right["answer"])
            self.assertTrue(v3.validate_ledger(variant["facts"]))

    def test_reporter_exchange_preserves_holder_but_holder_exchange_changes(self):
        right = pairs3.intervene(self.base, "reporter")
        self.assertEqual(self.base["facts"], right["facts"])
        self.assertTrue(any(a["speaker"] != b["speaker"] for a, b in zip(self.base["rounds"], right["rounds"]) if a["atoms"]))
        self.assertEqual(v3.solve(self.base, self.base["cases"][0]["query"]),
                         v3.solve(right, right["cases"][0]["query"]))
        # A true holder swap, holding reporter and question fixed, also changes
        # the answer; the other holder's previously wrong-scope fact is now gold.
        swapped = copy.deepcopy(self.base)
        first = self.left["answer_atoms"][0]["id"]
        original = next(f for f in swapped["facts"] if f["id"] == first)
        holder = original["holder"]
        replacement = next(f["holder"] for f in swapped["facts"] if f["holder"] != holder
                           and f["scope"] == original["scope"] and f["entity"] == original["entity"]
                           and f["attr"] == original["attr"] and f["status"] == "executed")
        for f in swapped["facts"]:
            if f["holder"] == holder:
                f["holder"] = replacement
            elif f["holder"] == replacement:
                f["holder"] = holder
        before = v3.solve(self.base, self.base["cases"][0]["query"])
        after = v3.solve(swapped, swapped["cases"][0]["query"])
        self.assertNotEqual(before["atom_ids"], after["atom_ids"])
        self.assertNotEqual(before["answer"], after["answer"])

    def test_views_and_names_do_not_stealth_change_semantics(self):
        for kind in pairs3.PRESERVE:
            variant = pairs3.intervene(self.base, kind, 7)
            case = variant["cases"][0]
            left = v3.solve(self.base, self.base["cases"][0]["query"])
            right = v3.solve(variant, case["query"])
            self.assertEqual(left["atom_ids"], right["atom_ids"])
            self.assertTrue(v3.validate_ledger(variant["facts"]))
            self.assertEqual(case["query"]["as_of"], self.base["cases"][0]["query"]["as_of"])

    def test_every_pair_uses_visible_fmt_and_has_no_cardinality_leak(self):
        for q in self.questions:
            docs = v3.visible_documents(q)
            self.assertTrue(q["visibility"]["ok"])
            self.assertNotRegex(q["question"].lower(), r"\blist\s+\d+")
            protected = {i for a in q["answer_atoms"] for i in a["equivalent_rounds"]}
            negative = {i for xs in q["negative_candidates"].values() for i in xs}
            self.assertFalse(protected & negative)
            for atom in q["answer_atoms"]:
                self.assertEqual(atom["equivalent_rounds"], [i for i, doc in enumerate(docs) if atom["statement"] in doc])

    def test_status_edit_requires_new_verification(self):
        right = pairs3.intervene(self.base, "approval")
        self.assertFalse(right["verified"])
        with self.assertRaisesRegex(ValueError, "run verify"):
            v3.build_questions(right)
        right = v3.verify(right, self.models)
        self.assertEqual(len(v3.build_questions(right)), 1)


if __name__ == "__main__":
    unittest.main()
