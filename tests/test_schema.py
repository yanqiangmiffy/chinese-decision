import copy
import random
import unittest

from decision.processing import examples_for, options_for, target_for, user_prompt


class DecisionSchemaTests(unittest.TestCase):
    def test_cj_distribution_and_question_key(self):
        case = {"state": "上下文", "questions": {"decision": {"type": "choice", "instructions": "选择", "criteria": {"x": "甲", "y": "乙"}}},
                "gold": {"decision": {"type": "choice", "probabilities": {"x": .2, "y": .8}}}}
        example, = examples_for(case, "general")
        self.assertEqual(example.target, [.2, .8])
        self.assertFalse(example.hard)
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]
        for seed in range(20):
            prompt, target = example.render(Tokenizer(), random.Random(seed))
            if prompt.index("x —") < prompt.index("y —"):
                self.assertEqual(target, [.2, .8])
            else:
                self.assertEqual(target, [.8, .2])

    def test_noul_order_and_score_levels(self):
        self.assertEqual([k for k, _ in options_for({"type": "noul", "instructions": "是否"})], ["false", "true"])
        q = {"type": "score", "instructions": "打分", "criteria": ["低", "中", "高"]}
        options = options_for(q)
        self.assertEqual([k for k, _ in options], ["0", "1", "2"])
        self.assertEqual(target_for({"probabilities": {"0": 0., "1": .25, "2": .75}}, options, "score"), [0., .25, .75])

    def test_invalid_labels_fail(self):
        options = [("x", "甲"), ("y", "乙")]
        for target in ({"x": 1}, {"x": -1, "y": 2}, {"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": float("nan"), "y": 0}):
            with self.assertRaises(ValueError):
                target_for({"probabilities": target}, options, "choice")

    def test_prompt_does_not_include_labels_or_metadata(self):
        case = {"state": {"内容": "资料"}, "questions": {"q": {"type": "noul", "instructions": "是否"}},
                "gold": {"q": {"probabilities": {"false": 0, "true": 1}}}, "_meta": {"secret": "LEAK_MARKER"}}
        e, = examples_for(copy.deepcopy(case), "general")
        prompt = user_prompt(e.state, e.instructions, e.options)
        self.assertNotIn("LEAK_MARKER", prompt)
        self.assertNotIn("probabilities", prompt)


if __name__ == "__main__":
    unittest.main()
