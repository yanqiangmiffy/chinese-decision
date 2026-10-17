import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.finalize import parse_args
from scripts.serve_gpu import Checkpoint


class ExportTests(unittest.TestCase):
    def test_calibration_has_independent_single_gpu_limits(self):
        args = parse_args(["--batch-size", "16", "--token-budget", "8192"])
        self.assertEqual((args.batch_size, args.token_budget), (16, 8192))
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--batch-size", "0"])

    def test_gpu_server_reads_merged_checkpoint_and_calibrated_metrics(self):
        with tempfile.TemporaryDirectory() as base:
            metadata = {
                "decision_config.json": {"model_name": "chinese-decision-2b", "temperature": 1.2},
                "training_summary.json": {"selected_checkpoint": {"step": 4994, "metric": "distribution_nll", "value": .586}},
                "validation_report.json": {"before": {"all": {"ece": .03}}, "after": {"all": {"ece": .02}}},
            }
            for name, contents in metadata.items():
                (Path(base) / name).write_text(json.dumps(contents), encoding="utf-8")
            with patch("scripts.serve_gpu.load_tokenizer"), \
                 patch("scripts.serve_gpu.load_model", return_value=MagicMock()), \
                 patch("scripts.serve_gpu.letter_tokens"):
                checkpoint = Checkpoint(base, None, "cuda")
            self.assertEqual(checkpoint.model_name, "chinese-decision-2b")
            self.assertEqual(checkpoint.describe()["step"], 4994)
            self.assertEqual(checkpoint.temperature, 1.2)
            self.assertEqual(checkpoint.validation, metadata["validation_report.json"]["after"])


if __name__ == "__main__":
    unittest.main()
