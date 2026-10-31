import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from decision import DecisionConfig, DecisionModel, DecisionPipeline
from decision.configuration import ExtractionConfig
from decision.modeling import SPAN_PREFIX
from test_model import tiny_qwen, tiny_tokenizer


def pure_torch_kernels():
    stack = ExitStack()
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
        if hasattr(modeling_qwen3_5, name):
            stack.enter_context(mock.patch.object(modeling_qwen3_5, name, None))
    return stack


class ConfigTests(unittest.TestCase):
    def test_released_decision_config_round_trips(self):
        released = {"model_name": "chinese-decision-0.8b", "temperature": 1.108, "prompt_version": "hmm-zh-v1",
                    "max_length": 16384, "max_options": 26, "source_run": "runs/cj-ddp",
                    "merge_probe_max_logit_delta": 0.125}
        config = DecisionConfig.from_dict(released)
        self.assertFalse(config.can_extract)
        self.assertEqual(config.metadata["source_run"], "runs/cj-ddp")
        self.assertEqual(config.to_dict(), released)

    def test_extraction_section_keeps_its_file_layout(self):
        data = {"temperature": 1.5, "extraction": {"head": {"hidden_size": 64, "max_span_units": 4, "inner": 32,
                                                            "out": 16, "width_dim": 32, "dropout": 0.1},
                                                   "none_shift": 1.6, "echo": True}}
        config = DecisionConfig.from_dict(data)
        self.assertEqual(config.extraction.head_kwargs()["inner"], 32)
        self.assertEqual(config.to_dict()["extraction"], data["extraction"])
        self.assertIsNone(config.extraction.decision_shift)
        data["extraction"]["decision_shift"] = 0.75
        config = DecisionConfig.from_dict(data)
        self.assertEqual(config.extraction.decision_shift, 0.75)
        self.assertEqual(config.to_dict()["extraction"], data["extraction"])

    def test_a_raw_base_directory_gets_the_defaults(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(DecisionConfig.from_pretrained(root), DecisionConfig())


class SaveLoadTests(unittest.TestCase):
    def test_one_directory_round_trips_backbone_with_merged_lora_and_span_head(self):
        model = DecisionModel(tiny_qwen())
        model.enable_lora(r=4, targets=["q_proj", "v_proj", "gate_proj"])
        for name, p in model.backbone.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.5)       # a trained adapter, so merging changes the logits
        head = model.add_span_head(max_span_units=4, none_shift=1.6, inner=32, out=16).eval()
        model.config.temperature = 1.25
        ids = torch.tensor([list(range(30, 42))])
        model.eval()
        with torch.inference_mode():
            adapted = model.backbone(ids, use_cache=False).logits
        with tempfile.TemporaryDirectory() as root:
            out = model.save_pretrained(Path(root) / "model")
            self.assertEqual(len(list(out.glob("*.safetensors"))), 1)
            self.assertFalse((out / "span_head.pt").exists())
            saved = json.loads((out / "decision_config.json").read_text(encoding="utf-8"))
            self.assertEqual((saved["temperature"], saved["extraction"]["none_shift"]), (1.25, 1.6))
            with pure_torch_kernels():
                reloaded = DecisionModel.from_pretrained(out, device="cpu")
            with torch.inference_mode():
                torch.testing.assert_close(reloaded.backbone(ids, use_cache=False).logits, adapted, atol=1e-4, rtol=1e-4)
            for key, value in head.state_dict().items():
                torch.testing.assert_close(reloaded.span_head.state_dict()[key], value)
            self.assertFalse(any(k.startswith(SPAN_PREFIX) for k in reloaded.backbone.state_dict()))
            self.assertEqual(reloaded.config.extraction, ExtractionConfig(hidden_size=64, max_span_units=4, inner=32,
                                                                           out=16, none_shift=1.6))

    def test_a_decision_only_directory_loads_without_a_span_head(self):
        with tempfile.TemporaryDirectory() as root:
            DecisionModel(tiny_qwen()).save_pretrained(root)
            with pure_torch_kernels():
                model = DecisionModel.from_pretrained(root, device="cpu")
        self.assertIsNone(model.span_head)
        with self.assertRaises(ValueError):
            model.span_utilities(None)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.pipe = DecisionPipeline.__new__(DecisionPipeline)
        self.pipe.decide = mock.Mock(return_value={"q": {"A": 1.0}})
        self.pipe.extract = mock.Mock(return_value=[])
        self.pipe.decide_and_extract = mock.Mock(return_value=({"q": {"A": 1.0}}, []))

    def test_the_request_shape_picks_the_readout(self):
        self.assertEqual(self.pipe({"state": "s", "question": {"type": "noul", "instructions": "?"}})["task"], "decision")
        self.pipe.decide.assert_called_once()
        self.assertEqual(self.pipe({"text": "t", "types": ["人名"]})["task"], "extraction")
        self.pipe.extract.assert_called_once_with("t", ["人名"], 0.0)

    def test_a_request_with_questions_and_types_gets_both_about_one_input(self):
        question = {"type": "noul", "instructions": "?"}
        result = self.pipe({"state": "s", "question": question, "types": ["人名"]})
        self.assertEqual((result["task"], result["answers"], result["entities"]), ("decision+extraction", {"q": {"A": 1.0}}, []))
        self.pipe.decide_and_extract.assert_called_once_with("s", {"q": question}, "s", ["人名"], 0.0)
        self.pipe.decide.assert_not_called()

    def test_empty_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            self.pipe({"text": "t", "types": ["人名"], "questions": {}})
        with self.assertRaises(ValueError):
            self.pipe({"text": "t"})


class DecideAndExtractTests(unittest.TestCase):
    """Which questions share the span row's forward pass, by padded size."""

    def pipe(self, decision_length, row_length):
        pipe = DecisionPipeline.__new__(DecisionPipeline)
        pipe.processor = mock.Mock()
        pipe.processor.encode_question.side_effect = lambda state, question: {
            "input_ids": [0] * decision_length, "options": [("A", "甲"), ("B", "乙")]}
        pipe.processor.encode_extraction.return_value = {"input_ids": [0] * row_length}
        pipe.processor.decode_spans.return_value = [(0, 1, "人名", 0.9)]
        pipe.model = mock.Mock(config=SimpleNamespace(temperature=1.0, extraction=None))
        pipe.model.joint_readouts.side_effect = lambda decisions, *_: (torch.tensor([[2.0, 0.0]] * len(decisions)), None)
        pipe.model.predict_spans.return_value = [[(0, 1, "人名", 0.9)]]
        pipe.candidate_logits = mock.Mock(side_effect=lambda items, **_: [[2.0, 0.0]] * len(items))
        pipe.letters = None
        return pipe

    def questions(self, count):
        return {f"q{i}": {"type": "choice", "instructions": "?", "criteria": {"A": "甲", "B": "乙"}} for i in range(count)}

    def test_short_inputs_share_one_pass(self):
        pipe = self.pipe(100, 200)
        answers, found = pipe.decide_and_extract("张三", self.questions(2), "张三", ["人名"])
        self.assertEqual(len(pipe.model.joint_readouts.call_args.args[0]), 2)
        pipe.model.predict_spans.assert_not_called()
        self.assertAlmostEqual(answers["q1"]["A"], 0.8808, places=4)
        self.assertEqual(found, [{"text": "张", "label": "人名", "start": 0, "end": 1, "score": 0.9}])

    def test_questions_past_the_joint_budget_get_their_own_batch(self):
        pipe = self.pipe(100, 200)                       # 256 padded tokens a row: 11 questions join the span row
        answers, _ = pipe.decide_and_extract("张三", self.questions(20), "张三", ["人名"])
        self.assertEqual((len(pipe.model.joint_readouts.call_args.args[0]), len(pipe.candidate_logits.call_args.args[0])),
                         (11, 9))
        self.assertEqual(len(answers), 20)

    def test_long_inputs_run_decisions_and_spans_apart(self):
        pipe = self.pipe(800, 2000)
        answers, found = pipe.decide_and_extract("张三", self.questions(2), "张三", ["人名"])
        pipe.model.joint_readouts.assert_not_called()
        self.assertEqual(len(pipe.candidate_logits.call_args.args[0]), 2)
        self.assertEqual((len(answers), found[0]["text"]), (2, "张"))


class PackageTests(unittest.TestCase):
    def test_public_classes_load_lazily(self):
        import decision
        self.assertEqual(set(decision.__all__) >= {"DecisionModel", "DecisionProcessor", "DecisionPipeline"}, True)
        with self.assertRaises(AttributeError):
            decision.NotAThing   # noqa: B018
        self.assertEqual(tiny_tokenizer().padding_side, "left")


if __name__ == "__main__":
    unittest.main()
