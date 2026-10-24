import json
import math
import tempfile
import unittest
from pathlib import Path

from decision.metrics import Accumulator, BenchmarkMetrics, summarize
from decision.processing import PROMPT_VERSION
from scripts.evaluate import deployment_metadata


class ValidationSummaryTests(unittest.TestCase):
    def test_soft_targets_enter_nll_but_not_accuracy(self):
        rows = [{"logits": [2., 0.], "target": [1., 0.], "hard": True, "domain": "general", "kind": "noul", "source": "s"},
                {"logits": [0., 2.], "target": [.5, .5], "hard": False, "domain": "general", "kind": "noul", "source": "s"}]
        result = summarize(rows)["all"]
        self.assertEqual((result["n"], result["hard_n"], result["soft_n"]), (2, 1, 1))
        self.assertEqual(result["accuracy"], 1.)
        p = 1 / (1 + math.exp(-2))
        self.assertAlmostEqual(result["distribution_nll"], (-math.log(p) - .5 * math.log(p * (1 - p))) / 2, places=5)


class BenchmarkMetricTests(unittest.TestCase):
    def test_ece_uses_15_bins_and_excludes_soft_targets(self):
        metric = Accumulator(15)
        metric.add([1, 0], [math.log(.61), math.log(.39)], "noul", True)
        metric.add([0, 1], [math.log(.65), math.log(.35)], "noul", True)
        metric.add([.5, .5], [math.log(.9), math.log(.1)], "noul", False)
        result = metric.result()
        # .61 and .65 both fall in bin 9/15; pooled ACC=.5 and confidence=.63.
        self.assertAlmostEqual(result["ece"], .13)
        self.assertEqual(result["hard_n"], 2)
        self.assertEqual(result["soft_n"], 1)
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["distribution_nll"],
                               (-math.log(.61) - math.log(.35) - .5 * math.log(.9 * .1)) / 3)

    def test_probability_one_and_score_expectation(self):
        metric = Accumulator(15)
        metric.add([0, 1], [-1000, 0], "score", True)
        result = metric.result()
        self.assertEqual(metric.bin_n[-1], 1)
        self.assertEqual(result["ece"], 0)
        self.assertEqual(result["expected_score_mae"], 0)

    def test_domain_macro_does_not_follow_medical_sample_weight(self):
        metric = BenchmarkMetrics()
        row = {"target": [1, 0], "hard": True, "kind": "noul", "source": "demo"}
        for _ in range(9):
            metric.add(row | {"domain": "medical"}, [math.log(.9), math.log(.1)])
        metric.add(row | {"domain": "legal"}, [math.log(.1), math.log(.9)])
        result = metric.result()
        self.assertAlmostEqual(result["groups"]["all"]["accuracy"], .9)
        self.assertAlmostEqual(result["macro_domain_accuracy"], .5)

    def test_merged_mode_preserves_validation_temperature_and_checks_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = {"step": 2000, "metric": "distribution_nll", "value": .7389}
            decision_config = {"temperature": 1.7, "prompt_version": PROMPT_VERSION, "max_length": 16384}
            (root / "decision_config.json").write_text(json.dumps(decision_config))
            (root / "training_summary.json").write_text(json.dumps({"selected_checkpoint": selection}))
            (root / "model.safetensors").write_bytes(b"test weight bytes")
            temperature, hashes = deployment_metadata(root, selection, {"max_length": 16384})
            self.assertEqual(temperature, 1.7)
            self.assertEqual(len(hashes["model.safetensors"]), 64)
            with self.assertRaisesRegex(ValueError, "different checkpoints"):
                deployment_metadata(root, selection | {"step": 3000}, {"max_length": 16384})
            decision_config["temperature"] = -1
            (root / "decision_config.json").write_text(json.dumps(decision_config))
            with self.assertRaisesRegex(ValueError, "temperature"):
                deployment_metadata(root, selection, {"max_length": 16384})


if __name__ == "__main__":
    unittest.main()
