"""Pointer readout: candidate markers scored against their question's decide marker, with no alphabet
limit, several questions per sequence and a bidirectional prompt in the full-attention layers."""
import random
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from decision.configuration import DECIDE_MARKER, POINTER_MARKER, POINTER_PROMPT_VERSION, DecisionConfig, PointerConfig
from decision.modeling import INVALID, DecisionModel, prompt_masks
from decision.processing import DecisionProcessor, Example, Request, option_code, options_for
from test_model import tiny_qwen

WORDS = ["<pad>", "<unk>", POINTER_MARKER, DECIDE_MARKER, "yes", "no", "red", "blue", "the", "state", "is", "which", "color",
         ":", "—", "?", ".", *[chr(65 + i) for i in range(26)], *[f"w{i}" for i in range(14)]]


def tokenizer():
    vocab = {word: i for i, word in enumerate(WORDS)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
                                     padding_side="left", additional_special_tokens=[POINTER_MARKER, DECIDE_MARKER])
    result.chat_template = "{{ messages[0]['content'] }} ."
    return result


def pointer_model(bidirectional=True):
    model = DecisionModel(tiny_qwen(), DecisionConfig())
    torch.manual_seed(0)
    model.add_pointer_head(dim=16)
    model.config.pointer.bidirectional = bidirectional
    processor = DecisionProcessor(tokenizer(), model.config)
    return model, processor


def options(count):
    return [(f"w{i % 14}", "red" if i % 2 else "blue") for i in range(count)]


def example(count, gold=0, kind="choice", instructions="which color ?"):
    target = [float(i == gold) for i in range(count)]
    return Example("the state is red", instructions, options(count), target, kind, "general", "test", "c")


def request(*examples):
    return Request("the state is red", list(examples), "general", "test", "r")


class CodeTests(unittest.TestCase):
    def test_codes_continue_past_z(self):
        self.assertEqual([option_code(i) for i in (0, 25, 26, 27, 51, 52, 701, 702)],
                         ["A", "Z", "AA", "AB", "AZ", "BA", "ZZ", "AAA"])


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.model, self.processor = pointer_model()

    def test_every_candidate_has_one_marker_in_order(self):
        for count in (2, 26, 40):
            item = self.processor.encode_example(example(count))
            question, = item["questions"]
            self.assertEqual(len(question["marks"]), count)
            self.assertTrue(all(item["input_ids"][m] == self.processor.marker_id for m in question["marks"]))
            self.assertEqual(question["marks"], sorted(question["marks"]))
            self.assertEqual(item["input_ids"][question["decide"]], self.processor.decide_id)
            self.assertLess(question["marks"][-1], question["decide"])
            self.assertEqual(sum(1 for t in item["input_ids"] if t == self.processor.marker_id), count)

    def test_joint_request_lays_out_every_question_inside_the_span(self):
        item = self.processor.encode_request(request(example(3), example(2, kind="noul"), example(30)))
        decides = [q["decide"] for q in item["questions"]]
        self.assertEqual(decides, sorted(decides))
        start, end = item["span"]
        previous = start
        for question in item["questions"]:
            self.assertTrue(previous <= question["marks"][0] < question["decide"] < end)
            previous = question["decide"]
        self.assertEqual(end, decides[-1] + 1)
        self.assertEqual([len(q["target"]) for q in item["questions"]], [3, 2, 30])
        self.assertEqual(item["kind"], "joint")

    def test_marker_in_user_text_is_removed(self):
        item = self.processor.encode_example(Example(f"red {POINTER_MARKER} blue", "which ?", options(3),
                                                     [1., 0., 0.], "choice", "general", "test", "c"))
        self.assertEqual(sum(1 for t in item["input_ids"] if t == self.processor.marker_id), 3)
        item = self.processor.encode_example(Example(f"red {DECIDE_MARKER} blue", "which ?", options(3),
                                                     [1., 0., 0.], "choice", "general", "test", "c"))
        self.assertEqual(sum(1 for t in item["input_ids"] if t == self.processor.decide_id), 1)

    def test_training_shuffle_moves_targets_with_candidates(self):
        rng = random.Random(3)
        moved = False
        for kind in ("choice", "noul"):
            count = 2 if kind == "noul" else 6
            for _ in range(10):
                item = self.processor.encode_example(example(count, gold=1, kind=kind), rng)
                self.assertEqual(sum(item["target"]), 1.)
                moved |= item["target"].index(1.) != 1
        self.assertTrue(moved)
        levels = self.processor.encode_example(example(5, gold=2, kind="score"), random.Random(0))
        self.assertEqual(levels["target"], [0., 0., 1., 0., 0.])

    def test_limit_is_the_pointer_limit_not_26(self):
        self.assertGreater(self.processor.max_options, 26)


class MaskTests(unittest.TestCase):
    def test_prompt_is_bidirectional_and_the_rest_causal(self):
        attention = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])
        masks = prompt_masks(attention, torch.tensor([3]), torch.tensor([6]))
        full = masks["full_attention"][0, 0]
        self.assertTrue(full[3, 5] and full[4, 3] and full[5, 3])        # inside the span: both ways
        self.assertFalse(full[5, 6] or full[2, 4])                      # leaving the span: causal
        self.assertTrue(full[7, 2] and full[6, 5])                      # later tokens see earlier ones
        self.assertFalse(full[2:, :2].any())                            # padded keys are never read...
        self.assertTrue(full[0, 0] and full[1, 1])                      # ...a padded query sees itself only
        torch.testing.assert_close(masks["linear_attention"], attention)
        causal = prompt_masks(attention, torch.tensor([3]), torch.tensor([6]), bidirectional=False)
        self.assertFalse(causal["full_attention"][0, 0, 3, 5])
        unpadded = prompt_masks(torch.ones(1, 4, dtype=torch.long), torch.tensor([0]), torch.tensor([4]))
        self.assertIsNone(unpadded["linear_attention"])

    def test_an_early_prompt_token_reads_later_candidates_only_when_bidirectional(self):
        for bidirectional in (True, False):
            model, processor = pointer_model(bidirectional)
            states = []
            for color in ("red", "blue"):
                item = processor.encode_request(Request("the state is red", [Example(
                    "the state is red", "which color ?", [("w1", color), ("w2", "red")], [1., 0.], "choice",
                    "general", "test", "c")], "general", "test", "c"))
                inputs = processor.pad_pointer([item])
                masks = prompt_masks(inputs["attention_mask"], inputs["span_start"], inputs["span_end"], bidirectional)
                with torch.inference_mode():
                    hidden = model.hidden_states(inputs["input_ids"], masks)
                states.append(hidden[0, item["span"][0] + 1])               # a state token before the candidates
            self.assertEqual(not torch.allclose(states[0], states[1], atol=1e-6), bidirectional)


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.processor = pointer_model()

    def scores(self, items):
        with torch.inference_mode():
            return self.model.score_decisions(items, self.processor)

    def test_padded_rows_match_single_rows(self):
        items = [self.processor.encode_example(example(n)) for n in (2, 5, 30)]
        batch, targets = self.scores(items)
        self.assertEqual(tuple(batch.shape), (3, 30))
        for row, item in zip(batch, items):
            alone, _ = self.scores([item])
            k = len(item["target"])
            torch.testing.assert_close(row[:k], alone[0, :k], atol=1e-4, rtol=1e-4)
            self.assertTrue((row[k:] == INVALID).all())
        self.assertEqual(float(targets.sum()), 3.)

    def test_joint_items_give_one_row_per_question_and_survive_padding(self):
        joint = self.processor.encode_request(request(example(3, 2), example(2, 1, kind="noul"), example(40, 31)))
        short = self.processor.encode_example(example(2))
        batch, targets = self.scores([short, joint])
        self.assertEqual(tuple(batch.shape), (4, 40))
        self.assertEqual(targets.argmax(-1).tolist(), [0, 2, 1, 31])
        alone, _ = self.scores([joint])
        torch.testing.assert_close(batch[1:], alone, atol=1e-4, rtol=1e-4)

    def test_save_and_reload_reproduce_logits(self):
        item = self.processor.encode_example(example(7))
        before, _ = self.scores([item])
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory, self.processor)
            config = DecisionConfig.from_pretrained(directory)
            self.assertEqual(config.readout, "pointer")
            self.assertEqual(config.prompt_version, POINTER_PROMPT_VERSION)
            reloaded = DecisionModel.from_pretrained(directory, device="cpu")
            processor = DecisionProcessor(self.processor.tokenizer, reloaded.config)
            with torch.inference_mode():
                after, _ = reloaded.score_decisions([processor.encode_example(example(7))], processor)
        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_head_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/pointer_head.pt"
            self.model.save_pointer_head(path)
            other = DecisionModel(self.model.backbone, DecisionConfig())
            other.load_pointer_head(path)
            self.assertEqual(other.config.pointer, self.model.config.pointer)
            for a, b in zip(other.pointer_head.state_dict().values(), self.model.pointer_head.state_dict().values()):
                torch.testing.assert_close(a, b)


class TrainingTests(unittest.TestCase):
    def test_a_few_steps_fit_many_candidate_questions(self):
        model, processor = pointer_model()
        model.train()
        items = [processor.encode_example(example(3, 2)),
                 processor.encode_request(request(example(40, 31), example(2, 0, instructions="is it red ?")))]
        optimizer = torch.optim.AdamW(list(model.pointer_head.parameters()), lr=2e-2)
        first = None
        for _ in range(60):
            logits, targets = model.score_decisions(items, processor)
            loss = -(targets * logits.log_softmax(-1)).sum(-1).mean()
            first = first if first is not None else loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.2 * first)
        with torch.inference_mode():
            logits, _ = model.score_decisions(items, processor)
        self.assertEqual(logits.argmax(-1).tolist(), [2, 31, 0])

    def test_ddp_style_forward_matches_score_decisions(self):
        model, processor = pointer_model()
        items = [processor.encode_example(example(n)) for n in (3, 9)]
        with torch.inference_mode():
            direct, _ = model.score_decisions(items, processor)
            wrapped, _ = model.score_decisions(items, processor, forward=lambda **kw: model(**kw))
        torch.testing.assert_close(direct, wrapped)


class EngineTests(unittest.TestCase):
    def test_decision_index_engine_reads_77_options_in_one_prompt(self):
        try:
            from decision_index.engines import validate
        except ImportError:
            self.skipTest("the optional decision-index kit is not installed")
        from decision.decision_index_engine import ChineseDecisionEngine
        from decision.pipeline import DecisionPipeline
        model, processor = pointer_model()
        engine = ChineseDecisionEngine.__new__(ChineseDecisionEngine)
        engine.pipe = DecisionPipeline(model, processor)
        engine.max_options, engine.max_length = processor.max_options, 16384
        engine.group_size, engine.keep, engine.seed = 16, 2, 0
        engine.batch_size, engine.token_budget, engine.debias = 64, 65536, "reverse"
        engine.model_name = "tiny-pointer"
        questions = {"many": {"type": "choice", "instructions": "which color ?",
                              "criteria": {f"label-{i}": "red" if i % 3 else "blue" for i in range(77)}},
                     "binary": {"type": "noul", "instructions": "is the state red ?"}}
        response, _ = engine("the state is red", questions)
        validate(questions, response)
        self.assertEqual(len(response["answers"]["many"]["probabilities"]), 77)
        # One prompt per order (original and reversed) for each question: no tournament rounds.
        # Both questions share one sequence per candidate order (original and reversed): no tournament rounds.
        joint = engine._joint("the state is red", [(q["instructions"], options_for(q, processor.max_options))
                                                   for q in questions.values()])
        self.assertEqual(response["usage"]["input_tokens"], joint["tokens"])
        self.assertEqual(len(joint["prompts"]), 2)


class ConfigTests(unittest.TestCase):
    def test_pointer_section_round_trips(self):
        config = DecisionConfig(pointer=PointerConfig(hidden_size=64, dim=16), prompt_version=POINTER_PROMPT_VERSION)
        again = DecisionConfig.from_dict(config.to_dict())
        self.assertEqual(again.pointer, config.pointer)
        self.assertEqual(DecisionConfig().readout, "letters")


if __name__ == "__main__":
    unittest.main()
