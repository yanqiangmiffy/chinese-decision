import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers.models.qwen3_5 import modeling_qwen3_5

from decision.extraction import SpanHead
from decision.model import load_model
from decision.unified import SPAN_PREFIX, UnifiedModel, load_span_head, save_unified
from test_model import tiny_qwen, tiny_tokenizer


def pure_torch_kernels():
    stack = ExitStack()
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
        if hasattr(modeling_qwen3_5, name):
            stack.enter_context(mock.patch.object(modeling_qwen3_5, name, None))
    return stack


class SaveLoadTests(unittest.TestCase):
    def test_one_directory_round_trips_backbone_with_merged_lora_and_span_head(self):
        model = get_peft_model(tiny_qwen(), LoraConfig(r=4, lora_alpha=8, task_type=TaskType.CAUSAL_LM,
                                                       target_modules=["q_proj", "v_proj", "gate_proj"]))
        for name, p in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.5)       # a trained adapter, so merging changes the logits
        ids = torch.tensor([list(range(30, 42))])
        with torch.inference_mode():
            adapted = model(ids, use_cache=False).logits
        merged = model.merge_and_unload().eval()
        head = SpanHead(merged.config.hidden_size, max_span_units=4, inner=32, out=16).eval()
        with tempfile.TemporaryDirectory() as root:
            out = save_unified(merged, head, tiny_tokenizer(), Path(root) / "model", {"temperature": 1.25},
                               {"none_shift": 1.6, "echo": True})
            weights = list(out.glob("*.safetensors"))
            self.assertEqual(len(weights), 1)
            self.assertFalse((out / "span_head.pt").exists())
            with pure_torch_kernels():
                reloaded = load_model(str(out), device="cpu").eval()
            with torch.inference_mode():
                torch.testing.assert_close(reloaded(ids, use_cache=False).logits, adapted, atol=1e-4, rtol=1e-4)
            loaded, settings = load_span_head(out, "cpu")
            self.assertEqual(settings["none_shift"], 1.6)
            for key, value in head.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)
            self.assertFalse(any(k.startswith(SPAN_PREFIX) for k in reloaded.state_dict()))

    def test_a_decision_only_directory_has_no_span_head(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(load_span_head(root, "cpu"), (None, {}))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.model = UnifiedModel.__new__(UnifiedModel)
        self.model.decide = mock.Mock(return_value={"q": {"A": 1.0}})
        self.model.extract = mock.Mock(return_value=[])

    def test_the_request_shape_picks_the_readout(self):
        self.assertEqual(self.model.predict({"state": "s", "question": {"type": "noul", "instructions": "?"}})["task"],
                         "decision")
        self.model.decide.assert_called_once()
        self.assertEqual(self.model.predict({"text": "t", "types": ["人名"]})["task"], "extraction")
        self.model.extract.assert_called_once_with("t", ["人名"], 0.0)

    def test_ambiguous_or_empty_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict({"text": "t", "types": ["人名"], "questions": {}})
        with self.assertRaises(ValueError):
            self.model.predict({"text": "t"})


if __name__ == "__main__":
    unittest.main()
