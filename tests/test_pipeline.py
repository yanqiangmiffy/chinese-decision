import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
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

    def test_the_request_shape_picks_the_readout(self):
        self.assertEqual(self.pipe({"state": "s", "question": {"type": "noul", "instructions": "?"}})["task"], "decision")
        self.pipe.decide.assert_called_once()
        self.assertEqual(self.pipe({"text": "t", "types": ["人名"]})["task"], "extraction")
        self.pipe.extract.assert_called_once_with("t", ["人名"], 0.0)

    def test_a_request_with_questions_and_types_gets_both_about_one_input(self):
        question = {"type": "noul", "instructions": "?"}
        result = self.pipe({"state": "s", "question": question, "types": ["人名"]})
        self.assertEqual((result["task"], result["answers"], result["entities"]), ("decision+extraction", {"q": {"A": 1.0}}, []))
        self.pipe.decide.assert_called_once_with("s", {"q": question})
        self.pipe.extract.assert_called_once_with("s", ["人名"], 0.0)

    def test_empty_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            self.pipe({"text": "t", "types": ["人名"], "questions": {}})
        with self.assertRaises(ValueError):
            self.pipe({"text": "t"})


class PackageTests(unittest.TestCase):
    def test_public_classes_load_lazily(self):
        import decision
        self.assertEqual(set(decision.__all__) >= {"DecisionModel", "DecisionProcessor", "DecisionPipeline"}, True)
        with self.assertRaises(AttributeError):
            decision.NotAThing   # noqa: B018
        self.assertEqual(tiny_tokenizer().padding_side, "left")


if __name__ == "__main__":
    unittest.main()
