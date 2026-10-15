import copy
import json
import tempfile
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

    def test_debug_adds_logits_and_token_counts(self):
        with patch("scripts.serve.GGUFBackend", FakeBackend):
            with TestClient(create_app("missing-config", backend="gguf", gguf="mock.gguf")) as client:
                plain = client.post("/v1/systemone", json=PAYLOAD).json()
                debug = client.post("/v1/systemone?debug=1", json=PAYLOAD).json()
        check_response(debug, PAYLOAD)
        self.assertNotIn("logits", plain["answers"]["department"])
        department = debug["answers"]["department"]
        self.assertEqual(department["logits"], {"billing": 1000., "technical": 1001., "sales": 1002.})
        self.assertGreater(department["input_tokens"], 0)
        self.assertEqual(sum(a["input_tokens"] for a in debug["answers"].values()), debug["usage"]["input_tokens"])

    def test_batch_isolates_invalid_requests(self):
        invalid = copy.deepcopy(PAYLOAD)
        invalid["questions"]["urgent"]["criteria"] = []
        with patch("scripts.serve.GGUFBackend", FakeBackend):
            with TestClient(create_app("missing-config", backend="gguf", gguf="mock.gguf")) as client:
                single = client.post("/v1/systemone", json=PAYLOAD).json()
                response = client.post("/v1/batch", json={"requests": [PAYLOAD, invalid, "x", PAYLOAD]}).json()
                self.assertEqual(client.post("/v1/batch", json={"requests": []}).status_code, 422)
                too_many = {"requests": [PAYLOAD] * 513}
                self.assertEqual(client.post("/v1/batch", json=too_many).status_code, 422)
        first, bad, junk, last = response["results"]
        for result in (first, last):
            check_response(result, PAYLOAD)
            self.assertEqual(result["answers"], single["answers"])
            self.assertNotIn("latency_ms", result)
        self.assertIn("question urgent", bad["error"])
        self.assertIn("required", junk["error"])
        self.assertEqual(response["decisions"], 6)
        self.assertEqual(response["usage"], {"input_tokens": 2 * single["usage"]["input_tokens"], "output_tokens": 0})

    def test_compile_returns_prompts_without_inference(self):
        engine = FakeBackend()
        with patch("scripts.serve.GGUFBackend", return_value=engine), patch.object(engine, "logits") as logits:
            with TestClient(create_app("missing-config", backend="gguf", gguf="mock.gguf")) as client:
                compiled = client.post("/v1/compile", json=PAYLOAD).json()
                self.assertEqual(client.post("/v1/compile", json={"state": "x"}).status_code, 422)
        logits.assert_not_called()
        department = compiled["questions"]["department"]
        self.assertEqual([o["letter"] + o["key"] for o in department["options"]], ["Abilling", "Btechnical", "Csales"])
        self.assertIn("支付、发票及退款", department["prompt"])
        self.assertEqual(department["input_tokens"], len(department["prompt"].encode("utf-8")))

    def test_info_and_training_without_release_metadata(self):
        with patch("scripts.serve.GGUFBackend", FakeBackend):
            with TestClient(create_app("missing-config", backend="gguf", gguf="mock.gguf")) as client:
                client.post("/v1/systemone", json=PAYLOAD)
                client.post("/v1/batch", json={"requests": [PAYLOAD, PAYLOAD]})
                info = client.get("/v1/info").json()
                training = client.get("/v1/training").json()
                health = client.get("/health").json()
        stats = info["stats"]
        self.assertEqual((stats["requests"], stats["decisions"], stats["forward_passes"]), (3, 9, 9))
        self.assertEqual(stats["avg_forward_batch"], 1.)
        self.assertIsNotNone(stats["latency_p95_ms"])
        self.assertEqual((info["checkpoint"]["backend"], info["checkpoint"]["step"], info["validation"]),
                         ("gguf", None, {}))
        self.assertEqual(info["engine"]["max_batch"], 1)
        self.assertEqual(training, {"available": False})
        self.assertIsNone(health["step"])

    def test_release_metadata_describes_the_checkpoint(self):
        summary = {"selected_checkpoint": {"step": 10000, "metric": "distribution_nll", "value": .6},
                   "training_progress": {"status": "completed", "step": 10132, "examples_seen": 13048000,
                                         "stats": {"trained:general:noul": 3}}}
        report = {"before": {"all": {"accuracy": .7}}, "after": {"all": {"accuracy": .73, "ece": .02, "hard_n": 1192}}}
        with tempfile.TemporaryDirectory() as base:
            (Path(base) / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (Path(base) / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
            with patch("scripts.serve.GGUFBackend", FakeBackend):
                with TestClient(create_app(base, backend="gguf", gguf="mock.gguf")) as client:
                    health = client.get("/health").json()
                    info = client.get("/v1/info").json()
                    training = client.get("/v1/training").json()
        self.assertEqual(health["step"], 10000)
        self.assertEqual(info["checkpoint"]["step"], 10000)
        self.assertEqual(info["validation"], report["after"])
        self.assertEqual((training["available"], training["status"], training["step"], training["progress"]),
                         (True, "completed", 10132, 1.))
        self.assertEqual(training["best"], summary["selected_checkpoint"])
        self.assertIsNotNone(training["updated_at"])

    def test_gguf_requires_merged_weights(self):
        with self.assertRaisesRegex(ValueError, "does not accept"):
            create_app("missing-config", adapter="adapter", backend="gguf", gguf="model.gguf")


if __name__ == "__main__":
    unittest.main()
