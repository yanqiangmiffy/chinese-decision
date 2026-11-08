"""The decision training mix: case builders, deduplication against the suite, packing and splits."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from decision.data.mix import Decontaminator, build_mix, choice, noul, pack, score, shingles
from decision.data.mix_sources import Source
from decision.processing import examples_for, request_for

LEAKED = "the quick brown fox jumps over the lazy dog near the river bank at dawn today"
TEMPLATE = "answer with one option from the list below and give no further explanation of any kind at all"


def write_suite(directory):
    rows = [{"state": {"text": f"{LEAKED} item {i}" if i == 0 else f"{TEMPLATE} row {i}"},
             "questions": {"q": {"criteria": {"a": "x"}}}} for i in range(25)]
    with gzip.open(Path(directory) / "selected-rows.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


class BuilderTests(unittest.TestCase):
    def test_cases_read_back_through_the_training_reader(self):
        cases = [choice("s", 1, "state", "which?", [("a", "first"), ("b", None)], "b"),
                 choice("s", 2, "state", "which?", ["a", "b", "c"], None, probabilities=[1, 1, 2]),
                 noul("s", 3, "state", "true?", True), noul("s", 4, "state", "true?", 0.25),
                 score("s", 5, "state", "how?", ["lo", "mid", "hi"], 2)]
        targets = [list(examples_for(case, "d"))[0].target for case in cases]
        self.assertEqual(targets, [[0., 1.], [.25, .25, .5], [0., 1.], [.75, .25], [0., 0., 1.]])
        self.assertEqual(cases[0]["questions"]["q"]["criteria"]["b"], "b")     # a missing description is the key

    def test_bad_gold_is_rejected(self):
        with self.assertRaises(ValueError):
            choice("s", 1, "state", "which?", ["a", "b"], "c")
        with self.assertRaises(ValueError):
            choice("s", 1, "state", "which?", ["a", "a"], "a")


class DecontaminationTests(unittest.TestCase):
    def test_leaked_spans_are_found_and_template_spans_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            write_suite(directory)
            index = Decontaminator.from_suite(directory, log=lambda *_: None)
        self.assertGreater(index.hits("prefix words " + LEAKED + " suffix"), 0)
        self.assertEqual(index.hits(TEMPLATE + " in another dataset"), 0)       # in 20+ rows: boilerplate
        self.assertEqual(index.hits("a short unrelated sentence"), 0)

    def test_chinese_counts_one_character_per_unit(self):
        self.assertEqual(len(shingles("一二三四五六七八九十一二")), 0)
        self.assertEqual(len(shingles("一二三四五六七八九十一二三")), 1)


class PackTests(unittest.TestCase):
    def test_questions_about_one_state_share_a_line(self):
        cases = [noul("s", i, "same state", f"q{i}?", i % 2 == 0, group="g") for i in range(10)]
        cases.append(noul("s", 99, "other state", "q?", True, group="h"))
        packed = list(pack(cases, 8))
        self.assertEqual(sorted(len(c["questions"]) for c in packed), [1, 2, 8])
        request = request_for(max(packed, key=lambda c: len(c["questions"])), "d")
        self.assertEqual(len(request.examples), 8)
        self.assertEqual(request.kind, "joint")

    def test_unpacked_cases_stay_alone(self):
        cases = [choice("s", n, "same problem", "answer?", options, "1", group="g") for n, options in
                 enumerate((["1", "2", "3", "4"], ["1", "5", "6", "7", "8"]))]
        for c in cases:
            c["_meta"]["pack"] = False
        self.assertEqual([len(c["questions"]) for c in pack(cases, 8)], [1, 1])


class BuildTests(unittest.TestCase):
    def test_build_caps_splits_by_group_and_reports(self):
        def adapter(src, root):
            for i in range(500):
                yield noul(src.name, i, f"state {i // 4}", f"question {i % 4}?", i % 3 == 0, group=i // 4)
            yield {"bad": True}
        source = Source("fake", "fake", "toy", "en", 200, "MIT", adapter)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "catalog" / "fake").mkdir(parents=True)
            report = build_mix([source], Path(directory) / "out", Path(directory) / "catalog", validation_cap=1000,
                               log=lambda *_: None)
            stats = report["sources"]["fake"]
            self.assertEqual(stats["valid"], 500)
            self.assertEqual(stats["invalid"], 1)
            self.assertEqual(stats["train"] + stats.get("validation", 0), 200)
            groups = {}
            for split in ("train", "validation"):
                with gzip.open(Path(directory) / "out" / "toy" / f"{split}.cases.jsonl.gz", "rt", encoding="utf-8") as f:
                    for line in f:
                        groups.setdefault(json.loads(line)["_meta"]["group_key"], set()).add(split)
            self.assertTrue(all(len(splits) == 1 for splits in groups.values()))


class BalanceAndDedupTests(unittest.TestCase):
    def test_balance_samples_labels_equally_and_dedup_reads_the_chosen_field(self):
        def adapter(src, root):
            for i in range(300):
                # 1 in 5 is true; every case shares the leaked passage, only some leak their claim.
                yield noul(src.name, i, {"passage": LEAKED, "claim": LEAKED if i % 50 == 0 else f"claim {i}"},
                           "supported?", i % 5 == 0)
        with tempfile.TemporaryDirectory() as directory:
            write_suite(directory)
            index = Decontaminator.from_suite(directory, log=lambda *_: None)
            (Path(directory) / "catalog" / "fake").mkdir(parents=True)
            source = Source("fake", "fake", "toy", "en", 1000, "MIT", adapter,
                            dedup=lambda case: case["state"]["claim"], balance=True)
            stats = build_mix([source], Path(directory) / "out", Path(directory) / "catalog", index,
                              log=lambda *_: None)["sources"]["fake"]
        self.assertEqual(stats["overlap_dropped"], 6)                 # only leaked claims, not the shared passage
        self.assertEqual(stats["label:true"], stats["label:false"])
        self.assertEqual(stats["label:true"], 54)                     # 60 true minus 6 leaked


class OrderTests(unittest.TestCase):
    def test_output_is_shuffled_when_the_source_is_sorted_by_label(self):
        def adapter(src, root):                              # raw file sorted by label, under the cap
            for i in range(400):
                yield noul(src.name, i, f"state {i}", "true?", i >= 200)
        for balance in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "catalog" / "fake").mkdir(parents=True)
                source = Source("fake", "fake", "toy", "en", 1000, "MIT", adapter, balance=balance)
                build_mix([source], Path(directory) / "out", Path(directory) / "catalog", validation_cap=0,
                          log=lambda *_: None)
                with gzip.open(Path(directory) / "out" / "toy" / "train.cases.jsonl.gz", "rt", encoding="utf-8") as f:
                    gold = [json.loads(line)["gold"]["q"] for line in f]
            fifth = len(gold) // 5
            for part in (gold[:fifth], gold[-fifth:]):       # both ends hold both labels in similar shares
                self.assertGreater(part.count("true") / len(part), .25)
                self.assertLess(part.count("true") / len(part), .75)


class FractionTests(unittest.TestCase):
    def test_kind_fraction_overrides_the_domain_fraction(self):
        from decision.data.cj import prepared_strata
        with tempfile.TemporaryDirectory() as directory:
            strata = [{"domain": "general", "kind": k, "path": f"{k}.gz", "decisions": 1000} for k in ("choice", "score")]
            (Path(directory) / "manifest.json").write_text(json.dumps(
                {"validation_included": False, "decisions": 2000, "strata": strata}))
            kept = {e["kind"]: e["decisions"] for e in prepared_strata(directory, ["general"],
                                                                      {"general": 0.5, "general:score": 0.1})}
        self.assertEqual(kept, {"choice": 500, "score": 100})


if __name__ == "__main__":
    unittest.main()
