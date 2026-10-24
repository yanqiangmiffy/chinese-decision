import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from tokenizers import Tokenizer, models
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from decision.modeling import DecisionModel
from decision.processing import DecisionProcessor

VOCAB = 64
LETTERS = torch.arange(1, 27)


def tiny_qwen():
    # Force the pure-torch kernels so this runs on CPU even where fla / causal-conv1d are
    # installed; padded positions are masked before either kernel is chosen.
    with ExitStack() as stack:
        for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                     "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
            if hasattr(modeling_qwen3_5, name):
                stack.enter_context(mock.patch.object(modeling_qwen3_5, name, None))
        torch.manual_seed(0)
        # No pad_token_id: the pad row must keep a nonzero embedding, like Qwen's <|endoftext|>,
        # or padding that reaches the model would have no effect to detect.
        config = Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                                   num_key_value_heads=1, head_dim=64, linear_num_key_heads=2,
                                   linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16)
        return AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()


def tiny_tokenizer():
    vocab = {f"t{i}": i for i in range(VOCAB)}
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="t1")),
                                   pad_token="t0", padding_side="left")


class ScoreDecisionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backbone = tiny_qwen()
        cls.model, cls.processor = DecisionModel(cls.backbone), DecisionProcessor(tiny_tokenizer())

    def alone(self, ids):
        with torch.inference_mode():
            return self.backbone(torch.tensor([ids]), use_cache=False).logits[0, -1, LETTERS]

    def batch_logits(self, inputs, k):
        batch = [{"input_ids": ids, "target": [1.] + [0.] * (k - 1)} for ids in inputs]
        with torch.inference_mode():
            return self.model.score_decisions(batch, self.processor, LETTERS)[0]

    def test_single_input_matches_unpadded_forward(self):
        # Ten tokens would be left-padded to 64, and Qwen3.5 does not mask a one-row batch.
        ids = list(range(30, 40))
        torch.testing.assert_close(self.batch_logits([ids], 3)[0, :3], self.alone(ids)[:3], atol=1e-4, rtol=1e-4)

    def test_padded_batch_rows_match_unpadded_forward(self):
        inputs = [list(range(30, 30 + n)) for n in (5, 12, 33)]
        for row, ids in zip(self.batch_logits(inputs, 3), inputs):
            torch.testing.assert_close(row[:3], self.alone(ids)[:3], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
