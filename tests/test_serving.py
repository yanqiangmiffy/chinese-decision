import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from decision.api import check_response
from decision.inference import candidate_probabilities
from scripts.serve import create_app

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads((ROOT / "examples/request.json").read_text(encoding="utf-8"))


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, prompt, **kwargs):
        return list(prompt.encode("utf-8"))


class FakeBackend:
    name, device, max_length = "gguf", "cpu", 4096
    tokenizer = Tokenizer()

    def __init__(self, *args, **kwargs):
        self.closed = False

    def logits(self, ids, count):
        return [1000. + i for i in range(count)]

    def close(self):
        self.closed = True


class ServingTests(unittest.TestCase):
    def test_stable_softmax_and_invalid_temperature(self):
        self.assertAlmostEqual(sum(candidate_probabilities([10000., 10001.], 1.)), 1.)
        self.assertEqual(candidate_probabilities([-10000., -10000.], 2.), [.5, .5])
        for temperature in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                candidate_probabilities([1., 2.], temperature)

    def test_both_routes_all_question_types_and_cleanup(self):
        engine = FakeBackend()
        with patch("scripts.serve.GGUFBackend", return_value=engine):
            with TestClient(create_app("missing-config", backend="gguf", gguf="mock.gguf")) as client:
                health = client.get("/health").json()
                self.assertEqual((health["backend"], health["device"], health["max_length"]),
                                 ("gguf", "cpu", 4096))
                first = client.post("/v1/systemone", json=PAYLOAD).json()
                second = client.post("/api/alpha/decisions", json=PAYLOAD).json()
                check_response(first, PAYLOAD)
                self.assertEqual(first["answers"], second["answers"])
            self.assertTrue(engine.closed)

    def test_invalid_and_overlong_inputs_are_rejected_before_inference(self):
        with patch("scripts.serve.GGUFBackend", FakeBackend):
            with TestClient(create_app("missing-config", backend="gguf", gguf="mock.gguf")) as client:
                invalid = copy.deepcopy(PAYLOAD)
                invalid["questions"]["urgent"]["criteria"] = []
                self.assertEqual(client.post("/v1/systemone", json=invalid).status_code, 422)
                overlong = copy.deepcopy(PAYLOAD)
                overlong["state"] = "长" * 4096
                response = client.post("/v1/systemone", json=overlong)
                self.assertEqual(response.status_code, 422)
                self.assertIn("not truncated", response.json()["detail"])

    def test_gguf_requires_merged_weights(self):
        with self.assertRaisesRegex(ValueError, "does not accept"):
            create_app("missing-config", adapter="adapter", backend="gguf", gguf="model.gguf")


if __name__ == "__main__":
    unittest.main()
