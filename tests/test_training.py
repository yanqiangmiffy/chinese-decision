import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from decision.training.common import evaluate_decisions, replace_directory, with_last
from scripts.train import parse_args, resume_conflicts, run_config

# Keys and values of the released run's config.json. --resume compares every key, so
# changing this set or these defaults makes existing runs impossible to resume.
RELEASED_RUN = {
    "base": "models/Qwen3.5-0.8B", "data": "datasets/CJ-Dataset", "prepared_data": "datasets/CJ-mixed",
    "out": "runs/cj-ddp", "global_batch_size": 1024, "token_budget": 262144, "pool_size": 4096,
    "max_length": 16384, "epochs": 1, "max_steps": 0, "debug_examples": 0, "val_per_stratum": 100,
    "eval_every": 1000, "save_every": 200, "lr": 1e-4, "lora_r": 32, "seed": 42,
    "domains": ["general", "medical", "legal", "finance"], "prompt_version": "hmm-zh-v1",
    "batch_size": 1024, "training_algorithm": "ddp-record-mean-v2-shapes64",
}


class TrainingHelperTests(unittest.TestCase):
    def test_with_last_marks_only_the_final_item(self):
        self.assertEqual(list(with_last("abc")), [("a", False), ("b", False), ("c", True)])
        self.assertEqual(list(with_last([])), [])

    def test_replace_directory_swaps_in_new_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "best").mkdir()
            (run / "best/marker").write_text("old")
            (run / "best.next").mkdir()
            (run / "best.next/marker").write_text("new")
            replace_directory(run / "best.next", run / "best")
            self.assertEqual((run / "best/marker").read_text(), "new")
            self.assertEqual(sorted(p.name for p in run.iterdir()), ["best"])

    def test_evaluate_scores_every_item_within_batch_limits(self):
        items = [{"input_ids": [1] * n, "target": [1., 0.], "domain": "general", "kind": "noul",
                  "source": "s", "hard": True, "case_id": str(n)} for n in (50, 10, 30, 20)]
        seen = []

        def score(batch, processor, letters):
            seen.append([e["case_id"] for e in batch])
            return torch.tensor([[2., 0.]] * len(batch)), None
        model = mock.Mock(training=True, score_decisions=score)
        metrics, rows = evaluate_decisions(model, mock.Mock(), items, batch_size=2, token_budget=1000)
        self.assertEqual(seen, [["10", "20"], ["30", "50"]])
        self.assertEqual(metrics["all"]["accuracy"], 1.)
        self.assertEqual([row["logits"] for row in rows], [[2., 0.]] * 4)
        model.train.assert_called_with(True)          # the previous mode comes back


class ResumeCompatibilityTests(unittest.TestCase):
    def test_defaults_reproduce_the_released_run_config(self):
        config = run_config(parse_args([]), world_size=4, data_revision="revision")
        ignored = {"resume", "log_every", "world_size", "data_revision", "keep_every"}
        self.assertEqual(set(config) - ignored, set(RELEASED_RUN))
        self.assertEqual({k: config[k] for k in RELEASED_RUN}, RELEASED_RUN)

    def test_runs_saved_before_keep_every_still_resume(self):
        config = run_config(parse_args(["--keep-every", "500"]), world_size=4, data_revision="r")
        old = {k: v for k, v in config.items() if k != "keep_every"}
        self.assertEqual(resume_conflicts(old, config), [])

    def test_rlcd_runs_are_recorded_and_never_resume_as_cross_entropy_runs(self):
        plain = run_config(parse_args([]), world_size=4, data_revision="r")
        rlcd = run_config(parse_args(["--rlcd", "--rlcd-sigma-end", ".2"]), world_size=4, data_revision="r")
        self.assertEqual(rlcd["training_algorithm"], RELEASED_RUN["training_algorithm"] + "+rlcd")
        self.assertEqual((rlcd["rlcd_group_size"], rlcd["rlcd_sigma_start"], rlcd["rlcd_sigma_end"]), (4, .4, .2))
        self.assertIn("training_algorithm", resume_conflicts(plain, rlcd))
        self.assertIn("rlcd_sigma_end", resume_conflicts(plain, rlcd))
        self.assertEqual(resume_conflicts(plain, run_config(parse_args(["--log-every", "1"]), 2, "r")), [])

    def test_invalid_rlcd_settings_are_rejected(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parse_args(["--rlcd", "--rlcd-group-size", "1"])


if __name__ == "__main__":
    unittest.main()
