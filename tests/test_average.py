"""Checkpoint averaging (rank-stacked LoRA) and the per-source checkpoint selection score."""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AverageTests(unittest.TestCase):
    def test_stacked_lora_equals_the_mean_weight_update(self):
        average = load("average_checkpoints")
        torch.manual_seed(0)
        r, n_in, n_out, scale = 4, 6, 5, 2.0                     # scale = lora_alpha / r
        states = [{"m.lora_A.weight": torch.randn(r, n_in), "m.lora_B.weight": torch.randn(n_out, r)} for _ in range(3)]
        weights = [0.5, 0.3, 0.2]
        stacked = average.average_adapters(states, weights)
        self.assertEqual(tuple(stacked["m.lora_A.weight"].shape), (3 * r, n_in))
        # rank 3r with lora_alpha * 3 keeps the scale lora_alpha / r
        merged = scale * stacked["m.lora_B.weight"] @ stacked["m.lora_A.weight"]
        expected = sum(w * scale * s["m.lora_B.weight"] @ s["m.lora_A.weight"] for w, s in zip(weights, states))
        torch.testing.assert_close(merged, expected)


class SelectionTests(unittest.TestCase):
    def test_source_accuracy_sees_a_collapsed_source(self):
        train = load("train")
        metrics = {"all": {"distribution_nll": 0.5, "accuracy": 0.8},
                   "source:a": {"accuracy": 0.9, "n": 50}, "source:b": {"accuracy": 0.1, "n": 50},
                   "source:tiny": {"accuracy": 0.0, "n": 3}, "source:soft": {"accuracy": None, "n": 40},
                   "domain:x": {"accuracy": 0.8, "n": 100}}
        self.assertEqual(train.selection_score(metrics, "distribution_nll"), 0.5)
        self.assertAlmostEqual(train.selection_score(metrics, "source_accuracy"), -0.5)   # tiny source ignored
        pooled = {"all": {"accuracy": 0.8}, "source:t/x": {"accuracy": 1.0, "n": 30, "hard_n": 30},
                  "source:t/y": {"accuracy": 0.0, "n": 10, "hard_n": 10}, "source:e": {"accuracy": 0.25, "n": 20, "hard_n": 20}}
        self.assertAlmostEqual(train.selection_score(pooled, "source_accuracy"), -(0.75 + 0.25) / 2)  # t pooled to 0.75


if __name__ == "__main__":
    unittest.main()
