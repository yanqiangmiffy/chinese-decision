import json
import unittest
from pathlib import Path

from decision.serving.api import build_answer, check_response
from decision.processing import options_for

PAYLOAD = json.loads((Path(__file__).resolve().parents[1] / "examples/request.json").read_text(encoding="utf-8"))
PROBABILITIES = {"urgent": [.2, .8], "department": [.7, .2, .1], "frustration": [.1, .3, .6]}


def response_for(payload):
    answers = {name: build_answer(question["type"], options_for(question), PROBABILITIES[name])
               for name, question in payload["questions"].items()}
    return {"answers": answers, "usage": {"input_tokens": 100, "output_tokens": 0}}


class AnswerFormatTests(unittest.TestCase):
    def test_each_type_reads_out_its_field(self):
        answers = response_for(PAYLOAD)["answers"]
        self.assertEqual(answers["urgent"]["noul"], .8)
        self.assertEqual(answers["department"]["choice"], "billing")
        self.assertAlmostEqual(answers["frustration"]["score"], .3 + 2 * .6)
        self.assertEqual(answers["frustration"]["legend"], {"0": "平静", "1": "不满", "2": "非常愤怒"})

    def test_confidence_is_zero_for_uniform_and_one_for_certain(self):
        options = [("a", "甲"), ("b", "乙"), ("c", "丙")]
        self.assertAlmostEqual(build_answer("choice", options, [1 / 3] * 3)["confidence"], 0)
        self.assertAlmostEqual(build_answer("choice", options, [1., 0., 0.])["confidence"], 1)

    def test_check_response_accepts_valid_and_rejects_tampered(self):
        check_response(response_for(PAYLOAD), PAYLOAD)
        tampered = response_for(PAYLOAD)
        tampered["answers"]["department"]["choice"] = "sales"
        with self.assertRaisesRegex(ValueError, "maximize"):
            check_response(tampered, PAYLOAD)
        generated = response_for(PAYLOAD)
        generated["usage"]["output_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "output tokens"):
            check_response(generated, PAYLOAD)


if __name__ == "__main__":
    unittest.main()
