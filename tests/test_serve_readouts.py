"""Both GPU-style servers load every model generation: letters-only, letters plus a span head, and pointer.

Tiny exported model directories run on CPU through the real `DecisionModel.from_pretrained`, so these
tests cover the path a released model takes: config -> readout -> encoding -> batched scoring -> answer.
"""
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from decision.configuration import POINTER_MAX_OPTIONS, POINTER_PROMPT_VERSION, PROMPT_VERSION, DecisionConfig
from decision.modeling import DecisionModel
from decision.pipeline import DecisionPipeline
from decision.processing import DecisionProcessor, option_code
from test_model import tiny_qwen
from test_pointer import pointer_model, tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_gpu  # noqa: E402
import serve_unified  # noqa: E402

NOUL = {"type": "noul", "instructions": "the state is red ?"}


def choice(count):
    return {"type": "choice", "instructions": "which color ?",
            "criteria": {f"w{i % 14}x{i}": "red" if i % 2 else "blue" for i in range(count)}}


def export(directory, readout, span_head=False):
    """A tiny model directory with the given readout (and optionally a span head), as save_pretrained writes it."""
    if readout == "pointer":
        model, processor = pointer_model()
    else:
        model = DecisionModel(tiny_qwen(), DecisionConfig())
        processor = DecisionProcessor(tokenizer(), model.config)
    if span_head:
        torch.manual_seed(1)
        model.add_span_head()
    model.config.model_name = f"tiny-{readout}"
    model.save_pretrained(directory, processor)
    return directory


def gpu_args(base):
    return argparse.Namespace(base=str(base), adapter=None, device="cpu", run_dir=None, snapshots="snapshots",
                              epoch_examples=1, token_budget=65536, max_batch=64, max_wait_ms=1.0, admin_token="")


class ReadoutServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.models = {"letters": export(root / "letters", "letters"),
                      "unified": export(root / "unified", "letters", span_head=True),
                      "pointer": export(root / "pointer", "pointer")}
        cls.pipes = {name: DecisionPipeline.from_pretrained(path, device="cpu") for name, path in cls.models.items()}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def reference_logits(self, name, state, question):
        """Logits of one question scored alone, the way scripts/evaluate_cj_bench.py reads them."""
        pipe = self.pipes[name]
        return pipe.candidate_logits([pipe.processor.encode_question(state, question)])[0]

    def servers(self, name):
        """Both servers on the same model directory, as TestClients (the GPU server warms up on entry)."""
        return {"serve_gpu": TestClient(serve_gpu.create_app(gpu_args(self.models[name]))),
                "serve_unified": TestClient(serve_unified.create_app(self.pipes[name], f"tiny-{name}"))}

    def check_routes(self, name, readout, prompt_version, max_options, candidates):
        body = {"state": "the state is red", "questions": {"n": NOUL, "c": choice(candidates)}}
        expected = {key: self.reference_logits(name, body["state"], question)
                    for key, question in body["questions"].items()}
        for server, client in self.servers(name).items():
            with self.subTest(model=name, server=server), client as http:
                self.assertEqual(http.get("/health").json()["readout"], readout)
                info = http.get("/v1/info").json()
                self.assertEqual((info["readout"], info["prompt_version"], info["limits"]["max_candidates"]),
                                 (readout, prompt_version, max_options))

                one = http.post("/v1/systemone?debug=1", json=body)
                self.assertEqual(one.status_code, 200, one.text)
                for key, logits in expected.items():
                    served = list(one.json()["answers"][key]["logits"].values())
                    self.assertEqual(len(served), len(logits))
                    for a, b in zip(served, logits):
                        self.assertAlmostEqual(a, b, places=3)
                    self.assertAlmostEqual(sum(one.json()["answers"][key]["probabilities"].values()), 1., places=4)

                batch = http.post("/v1/batch", json={"requests": [body, {"state": "x"}, body], "debug": True}).json()
                self.assertEqual([("error" in r) for r in batch["results"]], [False, True, False])
                self.assertEqual(batch["decisions"], 4)
                for result in (batch["results"][0], batch["results"][2]):
                    for a, b in zip(result["answers"]["c"]["logits"].values(), expected["c"]):
                        self.assertAlmostEqual(a, b, places=3)

                compiled = http.post("/v1/compile", json=body).json()
                self.assertEqual((compiled["prompt_version"], compiled["readout"]), (prompt_version, readout))
                options = compiled["questions"]["c"]["options"]
                self.assertEqual([o["letter"] for o in options], [option_code(i) for i in range(candidates)])
                prompt = compiled["questions"]["c"]["prompt"]
                pointer_marker = self.pipes[name].processor.config.pointer
                self.assertEqual(pointer_marker is not None and pointer_marker.marker in prompt, readout == "pointer")

                tournament = http.post("/v1/tournament", json={"state": "s", "question": choice(40)})
                self.assertEqual(tournament.status_code, 200, tournament.text)

    def test_letters_model(self):
        self.check_routes("letters", "letters", PROMPT_VERSION, 26, 5)

    def test_unified_model(self):
        self.check_routes("unified", "letters", PROMPT_VERSION, 26, 5)

    def test_pointer_model_reads_more_than_26_candidates(self):
        self.check_routes("pointer", "pointer", POINTER_PROMPT_VERSION, POINTER_MAX_OPTIONS, 30)

    def test_letters_models_reject_more_than_26_candidates(self):
        body = {"state": "s", "questions": {"c": choice(30)}}
        for server, client in self.servers("letters").items():
            with self.subTest(server=server), client as http:
                response = http.post("/v1/systemone", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertIn("2-26 candidates", response.text)

    def test_unified_decide_answers_a_pointer_question_past_26(self):
        with TestClient(serve_unified.create_app(self.pipes["pointer"], "tiny-pointer")) as http:
            response = http.post("/v1/decide", json={"state": "s", "question": choice(30)})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()["answers"]["q"]["probabilities"]), 30)

    def test_gpu_reload_refuses_a_checkpoint_with_another_readout(self):
        args = gpu_args(self.models["letters"])
        args.run_dir, args.admin_token = "run", "secret"
        pointer = serve_gpu.Checkpoint(self.models["pointer"], None, "cpu")
        with TestClient(serve_gpu.create_app(args)) as http, \
                patch("serve_gpu.snapshot_best", return_value=Path("snapshots/step-1")), \
                patch("serve_gpu.Checkpoint", return_value=pointer):
            response = http.post("/v1/admin/reload", headers={"X-Admin-Token": "secret"})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("restart the server", response.text)
            self.assertEqual(http.get("/health").json()["readout"], "letters")
            self.assertEqual(http.post("/v1/systemone", json={"state": "s", "questions": {"n": NOUL}}).status_code, 200)


class PointerSpanPipelineTests(unittest.TestCase):
    """A pointer model with a span head answers combined requests in two passes instead of failing."""

    def test_decide_and_extract_falls_back_to_separate_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = DecisionPipeline.from_pretrained(export(Path(directory), "pointer", span_head=True), device="cpu")
            answers, entities = pipe.decide_and_extract("the state is red", {"n": NOUL, "c": choice(30)},
                                                        "the state is red", ["color"])
            self.assertEqual((len(answers["n"]), len(answers["c"])), (2, 30))
            self.assertIsInstance(entities, list)
            with TestClient(serve_unified.create_app(pipe, "tiny-pointer-span")) as http:
                response = http.post("/v1/predict", json={"state": "the state is red", "questions": {"c": choice(30)},
                                                          "types": ["color"]})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["task"], "decision+extraction")


if __name__ == "__main__":
    unittest.main()
