import sys
import unittest
from pathlib import Path

from decision.processing import examples_for
from decision.data.slots import align, decision_case, extraction_row, load_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_slots import score  # noqa: E402

SCHEMA = load_schema(Path(__file__).resolve().parents[1] / "examples/slot_filling/schema.json")


class AlignTests(unittest.TestCase):
    def test_longer_values_are_placed_first_and_never_overlap(self):
        entities = align("把灯光调成暖光灯", {"device": "灯光", "value": "暖光灯"})
        self.assertEqual([(e["start"], e["end"], e["label"]) for e in entities], [(1, 3, "device"), (5, 8, "value")])
        # A value whose first occurrence is taken moves to its next one.
        entities = align("空调太冷，把空调关了", {"device": "空调", "location": "空调"})
        self.assertEqual(sorted(e["start"] for e in entities), [0, 6])

    def test_values_that_are_not_verbatim_or_cannot_avoid_overlap_are_rejected(self):
        self.assertIsNone(align("打开空调", {"device": "空调机"}))
        self.assertIsNone(align("打开空调", {"device": "空调", "function": "空调"}))
        self.assertEqual(align("讲个笑话", {}), [])


class RecordTests(unittest.TestCase):
    def test_a_record_feeds_both_heads(self):
        record = {"query": "打开空调", "intent": "device_control", "slots": {"function": "打开", "device": "空调"}}
        example, = examples_for(decision_case(record, SCHEMA, "train"), "general")
        self.assertEqual(example.options[example.target.index(1.0)][0], "device_control")
        row = extraction_row(record, SCHEMA, "train")
        self.assertEqual([t["key"] for t in row["types"]], list(SCHEMA["intents"]["device_control"]["slots"]))
        self.assertIsNone(extraction_row({"query": "谢谢", "intent": "chitchat", "slots": {}}, SCHEMA, "train"))


class ScoreTests(unittest.TestCase):
    def test_slots_under_a_wrong_intent_are_errors(self):
        rows = [{"query": "打开空调", "intent": "device_control", "slots": {"function": "打开", "device": "空调"}},
                {"query": "谢谢", "intent": "chitchat", "slots": {}}]
        predictions = [{"intent": "device_control", "slots": {"device": "空调"}},
                       {"intent": "navigation", "slots": {"destination": "谢谢"}}]
        metrics = score(rows, predictions)
        self.assertEqual((metrics["intent_accuracy"], metrics["exact_match"]), (0.5, 0.0))
        self.assertEqual((metrics["slots"]["precision"], metrics["slots"]["recall"]), (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
