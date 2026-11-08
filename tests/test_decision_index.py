"""Adapter contract with the optional official Decision Index runner (no model download)."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

if importlib.util.find_spec("decision_index") is None:
    raise unittest.SkipTest("install the optional official decision-index kit to test this integration")

from decision_index.engines import Unsupported, validate
from decision_index.runner import run

from decision.configuration import DecisionConfig
from decision.decision_index_engine import ChineseDecisionEngine, spread
from decision.processing import DecisionProcessor
from decision.serving.backends import candidate_probabilities


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, prompt, **kwargs):
        return list(prompt.encode("utf-8"))


def question(count=2):
    return {"type": "choice", "instructions": "Choose a label.",
            "criteria": {f"label-{i}": f"Description {i}" for i in range(count)}}


def engine():
    result = ChineseDecisionEngine.__new__(ChineseDecisionEngine)
    result.max_options, result.max_length = 26, 16384
    result.model_name = "test-decision"
    result.group_size, result.keep, result.seed = 16, 2, 0
    result.batch_size, result.token_budget, result.debias = 64, 65536, "none"
    # Logit = the candidate's number ("label-7" -> 7), so the largest label always wins.
    result.pipe = SimpleNamespace(processor=DecisionProcessor(Tokenizer(), DecisionConfig()), temperature=2.,
                                  candidate_logits=Mock(side_effect=lambda items, **kwargs: [
                                      [float(key.split("-")[-1]) if "-" in key else float(i)
                                       for i, (key, _) in enumerate(item["options"])] for item in items]))
    return result


class DecisionIndexTests(unittest.TestCase):
    def test_cuda_unavailable_fails_before_loading_weights(self):
        with patch("decision.decision_index_engine.torch.cuda.is_available", return_value=False), \
                patch("decision.decision_index_engine.resolve_pretrained") as resolve:
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                ChineseDecisionEngine("unused", device="cuda")
            resolve.assert_not_called()

    def test_real_small_model_load_and_probabilities_match_native_pipeline(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from decision.modeling import DecisionModel
        from test_model import tiny_qwen

        vocab = {"<eos>": 0, "<unk>": 1, **{chr(65 + i): i + 2 for i in range(26)}}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>",
                                            unk_token="<unk>", pad_token="<eos>", padding_side="left")
        tokenizer.chat_template = "{{ messages[0]['content'] }}"
        config = DecisionConfig(model_name="tiny-selfcheck", temperature=1.7)
        processor = DecisionProcessor(tokenizer, config)
        questions = {"q": question(), "yes": {"type": "noul", "instructions": "Is this true?"}}
        with tempfile.TemporaryDirectory() as directory:
            DecisionModel(tiny_qwen(), config).save_pretrained(directory, processor)
            adapter = ChineseDecisionEngine(directory, device="cpu", threads=1)
            try:
                response, _ = adapter("A state", questions)
                validate(questions, response)
                native = adapter.pipe.decide("A state", questions)
                for name in questions:
                    for key, p in native[name].items():
                        self.assertAlmostEqual(response["answers"][name]["probabilities"][key], p, places=5)
                self.assertIn("model.safetensors", adapter.provenance["artifact_sha256"])
                self.assertEqual(adapter.runtime()["device"], "cpu")
            finally:
                adapter.close()

    def test_choice_and_noul_pass_official_validation_without_mutating_input(self):
        adapter = engine()
        questions = {"choice": question(), "binary": {"type": "noul", "instructions": "Is it true?"}}
        before = copy.deepcopy(questions)
        response, raw = adapter("A state", questions)
        validate(questions, response)
        self.assertEqual(questions, before)
        self.assertIsNone(raw)
        self.assertEqual(response["answers"]["choice"]["choice"], "label-1")
        expected = candidate_probabilities([0., 1.], 2.)[1]
        self.assertAlmostEqual(response["answers"]["binary"]["noul"], expected)
        self.assertGreater(response["usage"]["input_tokens"], 0)
        self.assertEqual(response["usage"]["output_tokens"], 0)

    def test_maximum_candidate_count_keeps_every_option(self):
        questions = {"q": question(26)}
        response, _ = engine()("A state", questions)
        validate(questions, response)
        self.assertEqual(list(response["answers"]["q"]["probabilities"]), list(questions["q"]["criteria"]))

    def test_capacity_rejection_precedes_all_forward_passes(self):
        for oversized in ({**question(), "instructions": "x" * 16385}, {**question(77), "instructions": "x" * 16385}):
            adapter = engine()
            with self.assertRaises(Unsupported):
                adapter("A state", {"valid": question(), "oversized": oversized})
            adapter.pipe.candidate_logits.assert_not_called()

    def test_tournament_answers_every_candidate_and_keeps_the_winner_most_probable(self):
        for count in (27, 77, 151):
            adapter = engine()
            questions = {"many": question(count), "binary": {"type": "noul", "instructions": "Is it true?"}}
            response, _ = adapter("A state", questions)
            validate(questions, response)
            answer = response["answers"]["many"]
            self.assertEqual(answer["choice"], f"label-{count - 1}")
            self.assertEqual(list(answer["probabilities"]), list(questions["many"]["criteria"]))
            self.assertEqual(max(answer["probabilities"], key=answer["probabilities"].get), answer["choice"])
            self.assertTrue(all(p > 0 for p in answer["probabilities"].values()))
            # The plain question rides in the first round's batch; each later round is one more call.
            self.assertEqual(adapter.pipe.candidate_logits.call_count, {27: 2, 77: 2, 151: 3}[count])
            groups = adapter.pipe.candidate_logits.call_args_list[0].args[0]
            self.assertTrue(all(len(item["options"]) <= 16 for item in groups))

    def test_long_choice_prompt_shrinks_groups_until_they_fit(self):
        adapter = engine()
        adapter.max_length = 1200
        long = {"type": "choice", "instructions": "Choose a label.",
                "criteria": {f"label-{i}": "y" * 100 for i in range(20)}}
        response, _ = adapter("A state", {"q": long})
        validate({"q": long}, response)
        self.assertEqual(response["answers"]["q"]["choice"], "label-19")
        first = adapter.pipe.candidate_logits.call_args_list[0].args[0]
        self.assertTrue(all(len(item["input_ids"]) <= 1200 for item in first))

    def test_reverse_order_cancels_a_position_preference(self):
        adapter = engine()
        # A model that only likes the last position: logit = position.
        adapter.pipe.candidate_logits = Mock(side_effect=lambda items, **kwargs: [
            [float(i) for i in range(len(item["options"]))] for item in items])
        questions = {"pair": question(2), "four": question(4), "yes": {"type": "noul", "instructions": "Is it true?"}}
        biased, _ = adapter("A state", questions)
        self.assertEqual(biased["answers"]["pair"]["choice"], "label-1")
        adapter.debias = "reverse"
        fair, _ = adapter("A state", questions)
        validate(questions, fair)
        self.assertEqual(adapter.pipe.candidate_logits.call_count, 2)  # both orders in one batch
        for name in ("pair", "four"):
            values = list(fair["answers"][name]["probabilities"].values())
            self.assertAlmostEqual(values[0], values[-1])
        self.assertAlmostEqual(fair["answers"]["yes"]["noul"], .5)
        shown = [item["options"] for item in adapter.pipe.candidate_logits.call_args.args[0]]
        self.assertIn([("label-1", "Description 1"), ("label-0", "Description 0")], shown)

    def test_reverse_order_keeps_an_order_free_model_unchanged(self):
        plain, _ = engine()("A state", {"q": question(5)})
        adapter = engine()
        adapter.debias = "reverse"
        both, _ = adapter("A state", {"q": question(5)})
        for key, p in plain["answers"]["q"]["probabilities"].items():
            self.assertAlmostEqual(both["answers"]["q"]["probabilities"][key], p)

    def test_spread_orders_rounds_and_sums_to_one(self):
        match = SimpleNamespace(criteria={k: k for k in "abcdef"}, rounds=[{"groups": [
            {"probabilities": {"a": .5, "b": .3, "c": .2}, "advanced": ["a", "b"]},
            {"probabilities": {"d": .1, "e": .1, "f": .8}, "advanced": ["f", "d"]}]}])
        values = dict(zip("abcdef", spread(match, {"a": .4, "b": .1, "f": .45, "d": .05})))
        self.assertAlmostEqual(sum(values.values()), 1)
        self.assertEqual(max(values, key=values.get), "f")
        self.assertLessEqual(max(values["c"], values["e"]), min(values[k] for k in "abdf"))

    def test_malformed_request_is_not_misreported_as_capacity(self):
        with self.assertRaises(ValueError) as caught:
            engine()("A state", {"q": {**question(), "criteria": []}})
        self.assertNotIsInstance(caught.exception, Unsupported)

    def test_official_runner_records_ok_and_unsupported_and_resumes(self):
        adapter = engine()
        adapter.synchronize = Mock()
        adapter.runtime = Mock(return_value={})
        adapter.close = Mock()
        rows = [{"state": "A state", "questions": {"q": question(count)},
                 "_evaluation": {"run_id": str(count)}} for count in (2, 27)]
        rows.append({"state": "A state", "questions": {"q": {**question(), "instructions": "x" * 16385}},
                     "_evaluation": {"run_id": "long"}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "rows.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with patch("decision_index.runner.load_engine", return_value=adapter):
                run("test", {}, source, path / "run", warm=False, log=lambda *args: None)
                run("test", {}, source, path / "run", warm=False, log=lambda *args: None)
            results = [json.loads(line) for line in (path / "run/results.jsonl").read_text().splitlines()]
        self.assertEqual([row["status"] for row in results], ["ok", "ok", "unsupported"])
        self.assertEqual(adapter.pipe.candidate_logits.call_count, 3)


if __name__ == "__main__":
    unittest.main()
