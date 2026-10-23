import unittest

import torch

from decision.extraction import (ExtractionEncoder, SpanHead, SpanLoss, backbone_hidden, collate_extraction, decode,
                                 entity_metrics, text_units)
from decision.rlcd_loss import RLCDLossConfig
from test_model import tiny_qwen

TYPES = [{"key": "name", "name": "人名", "description": "人物姓名"}, {"key": "loc", "name": "地名", "description": ""}]


class CharTokenizer:
    """One token per character, ids inside the tiny model's vocabulary."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(c) % 60 for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<u>" + messages[0]["content"] + "</u>"


def row(text, entities=()):
    return {"id": text, "text": text, "types": TYPES,
            "entities": [{"start": s, "end": e, "label": label} for s, e, label in entities]}


class EncodingTests(unittest.TestCase):
    def test_units_split_cjk_characters_and_keep_ascii_words(self):
        text = "我在NBA打球2010年"
        self.assertEqual([text[s:e] for s, e in text_units(text)], ["我", "在", "NBA", "打", "球", "2010", "年"])

    def test_gold_spans_map_to_candidates_or_count_as_unrepresented(self):
        encoder = ExtractionEncoder(CharTokenizer(), max_span_units=3)
        item = encoder.encode(row("张三在NBA北京市海淀", [(0, 2, "name"), (4, 6, "loc"), (6, 11, "loc")]))
        # "BA" cuts an ASCII unit and 北京市海淀 is five units wide: neither is a candidate.
        self.assertEqual(item["unrepresented"], 2)
        (k, labels), = item["gold"].items()
        self.assertEqual((item["cands"][k], labels), ((0, 1), [0]))

    def test_unit_tokens_point_at_the_span_copy(self):
        tokenizer = CharTokenizer()
        for echo in (True, False):
            item = ExtractionEncoder(tokenizer, echo=echo).encode(row("张三去了NBA"))
            ids, text = item["input_ids"], item["text"]
            for (start, end), (first, last) in zip(item["units"], item["unit_tokens"]):
                self.assertEqual(ids[first:last + 1], tokenizer.encode(text[start:end]))
            self.assertEqual(item["unit_tokens"][-1][1], len(ids) - 1)    # the sequence ends at the last unit
            self.assertEqual(ids.count(tokenizer.encode("张")[0]), 2 if echo else 1)

    def test_type_spans_cover_name_and_description(self):
        tokenizer = CharTokenizer()
        item = ExtractionEncoder(tokenizer).encode(row("张三"))
        (s0, e0), (s1, e1) = item["type_spans"]
        self.assertEqual(item["input_ids"][s0:e0], tokenizer.encode("人名：人物姓名"))
        self.assertEqual(item["input_ids"][s1:e1], tokenizer.encode("地名"))


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_qwen()
        cls.encoder = ExtractionEncoder(CharTokenizer(), max_span_units=4)
        torch.manual_seed(0)
        cls.head = SpanHead(cls.model.config.hidden_size, max_span_units=4, inner=32, out=16).eval()

    def utilities(self, items):
        batch = collate_extraction(items, CharTokenizer.pad_token_id)
        with torch.inference_mode():
            return self.head(backbone_hidden(self.model, batch), batch), batch

    def test_left_padding_does_not_change_span_utilities(self):
        short = self.encoder.encode(row("张三去北京", [(0, 2, "name")]))
        long = self.encoder.encode(row("李四昨天从上海飞到了北京市", [(10, 13, "loc")]))
        alone, _ = self.utilities([short])
        together, batch = self.utilities([long, short])
        torch.testing.assert_close(together[len(long["cands"]):], alone, atol=1e-4, rtol=1e-4)
        self.assertEqual(batch.targets.shape, (len(long["cands"]) + len(short["cands"]), 3))
        torch.testing.assert_close(batch.targets.sum(-1), torch.ones(batch.targets.shape[0]))

    def test_span_loss_trains_the_head_and_weights_none_spans(self):
        item = self.encoder.encode(row("张三去北京", [(0, 2, "name"), (3, 5, "loc")]))
        batch = collate_extraction([item], CharTokenizer.pad_token_id)
        with torch.no_grad():
            hidden = backbone_hidden(self.model, batch)
        head = SpanHead(self.model.config.hidden_size, max_span_units=4, inner=32, out=16)
        utilities = head(hidden, batch)
        plain, _ = SpanLoss(RLCDLossConfig(rlcd_weight=0.0))(utilities, batch)
        expected = -(batch.targets * utilities.log_softmax(-1)).sum(-1).mean()
        torch.testing.assert_close(plain, expected)
        weighted, _ = SpanLoss(RLCDLossConfig(), none_weight=0.1)(utilities, batch, sigma=0.3)
        weighted.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))


class DecodeTests(unittest.TestCase):
    def test_flat_decoding_keeps_the_more_probable_of_overlapping_spans(self):
        item = {"cands": [(0, 1), (1, 2), (3, 3)], "units": [(0, 1), (1, 2), (2, 3), (3, 4)], "types": ["name", "loc"]}
        utilities = torch.tensor([[0., 3., 0.], [0., 0., 5.], [2., 0., 0.]])
        self.assertEqual([p[:3] for p in decode(item, utilities)], [(1, 3, "loc")])
        self.assertEqual(len(decode(item, utilities, flat=False)), 2)
        # Shifting "none" up by log(1 / none_weight) can turn a weak span back into none.
        self.assertEqual(decode(item, utilities, none_shift=6.0), [])

    def test_entity_metrics_count_exact_matches(self):
        items = [{"gold_entities": {(0, 2, "name"), (3, 5, "loc")}, "unrepresented": 0}]
        metrics = entity_metrics(items, [[(0, 2, "name", 0.9), (3, 5, "name", 0.6)]])
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["f1"]), (0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
