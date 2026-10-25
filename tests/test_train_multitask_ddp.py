import json
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_replay_data  # noqa: E402
import train_multitask_ddp as trainer  # noqa: E402


def entry(step, ie, cj, checkpoint=True):
    return {"step": step, "checkpoint": f"checkpoint-{step}" if checkpoint else None,
            "scores": {"ie_score": ie, "cj_accuracy": cj}}


class SelectionTests(unittest.TestCase):
    def test_best_new_task_score_within_the_retention_limit(self):
        history = [entry(0, 0.1, 0.80, checkpoint=False), entry(500, 0.70, 0.795), entry(1000, 0.80, 0.78),
                   entry(1500, 0.75, 0.792)]
        self.assertEqual(trainer.select(history, max_drop=1.0)["step"], 1500)     # 1000 drops 2 points
        self.assertEqual(trainer.select(history, max_drop=3.0)["step"], 1000)

    def test_joint_selection_weighs_cj_accuracy_and_the_new_tasks(self):
        history = [entry(0, 0.1, 0.30, checkpoint=False), entry(500, 0.80, 0.60), entry(1000, 0.70, 0.75)]
        self.assertEqual(trainer.select(history, max_drop=1.0, mode="retention")["step"], 500)
        self.assertEqual(trainer.select(history, max_drop=1.0, mode="joint")["step"], 1000)

    def test_cj_train_is_the_replay_flag_under_its_raw_base_name(self):
        self.assertEqual(trainer.parse_args(["--out", "x", "--cj-train", "cj.jsonl"]).replay, "cj.jsonl")

    def test_smallest_drop_when_nothing_qualifies_and_ie_only_without_cj_dev(self):
        history = [entry(0, 0.1, 0.80, checkpoint=False), entry(500, 0.70, 0.75), entry(1000, 0.80, 0.70)]
        self.assertEqual(trainer.select(history, max_drop=1.0)["step"], 500)
        history = [entry(0, 0.1, None, checkpoint=False), entry(500, 0.70, None), entry(1000, 0.80, None)]
        self.assertEqual(trainer.select(history, max_drop=1.0)["step"], 1000)


class LossAndGradientTests(unittest.TestCase):
    def test_kl_is_zero_for_the_reference_and_ignores_masked_options(self):
        reference = torch.tensor([[2.0, 0.5, -1e4], [0.0, 1.0, 3.0]])
        self.assertAlmostEqual(trainer.kl_to_reference(reference.clone(), reference).item(), 0.0, places=6)
        moved = reference.clone()
        moved[0, 0] = 0.0
        self.assertGreater(trainer.kl_to_reference(moved, reference).item(), 0.0)

    def test_lora_grads_average_over_micro_batches_and_the_head_over_extraction_ones(self):
        lora, head = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))
        lora.grad, head.grad = torch.full((3,), 8.0), torch.full((2,), 6.0)
        trainer.sync_gradients([lora], [head], micro_batches=4, extraction_batches=3, world=1)
        torch.testing.assert_close(lora.grad, torch.full((3,), 2.0))
        torch.testing.assert_close(head.grad, torch.full((2,), 2.0))
        lora.grad, head.grad = torch.ones(3), None
        trainer.sync_gradients([lora], [head], micro_batches=1, extraction_batches=0, world=1)
        self.assertIsNone(head.grad)                 # no extraction batch: AdamW leaves the head alone


class PlanTests(unittest.TestCase):
    def test_every_task_is_batched_and_ranks_get_disjoint_equal_shares(self):
        args = trainer.parse_args(["--out", "unused", "--decision-batch", "2", "--extraction-batch", "2"])
        lengths = {"decision": [10, 20, -1, 30], "replay": [5, 5, 5], "extraction": [40, 50]}
        plans = trainer.epoch_plan(args, 0, lengths, world=2)
        self.assertEqual(len(plans[0]), len(plans[1]))
        seen = [(task, i) for plan in plans for task, indices, _ in plan for i in indices]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertNotIn(("decision", 2), seen)      # unusable (too long) items never enter a batch
        self.assertEqual({task for task, _ in seen}, {"decision", "replay", "extraction"})

    def test_rlcd_is_on_by_default(self):
        self.assertTrue(trainer.parse_args(["--out", "x"]).rlcd)
        self.assertFalse(trainer.parse_args(["--out", "x", "--no-rlcd"]).rlcd)


class ReplayAllocationTests(unittest.TestCase):
    def test_square_root_shares_capped_by_shard_size_and_summing_to_the_budget(self):
        sizes = {"big": 1_000_000, "mid": 10_000, "small": 100}
        counts = build_replay_data.allocate(sizes, 20_000)
        self.assertEqual(sum(counts.values()), 20_000)
        self.assertEqual(counts["small"], 100)                     # capped, the rest handed on
        self.assertGreater(counts["mid"] / sizes["mid"], counts["big"] / sizes["big"])
        self.assertEqual(build_replay_data.allocate(sizes, 0), sizes)   # 0 keeps the whole domain

    def test_retag_keeps_the_original_family(self):
        case = build_replay_data.retag({"_meta": {"source_family": "c3"}}, "legal")
        self.assertEqual(case["_meta"]["source_family"], "cj:legal")
        self.assertEqual(case["_meta"]["cj_source_family"], "c3")
        self.assertTrue(math.isfinite(len(case)))


class LineStoreTests(unittest.TestCase):
    def test_lines_are_read_on_demand_through_saved_offsets(self):
        import pickle
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a.jsonl", Path(root) / "b.jsonl"
            a.write_text('{"x": 1}\n\n{"x": "二"}\n', encoding="utf-8", newline="\n")
            b.write_text('{"y": 3}\n{"y": 4}\n{"y": 5}\n', encoding="utf-8", newline="\n")
            store = trainer.LineStore.build([a, b], limit=2)
            self.assertEqual([json.loads(store[i]) for i in range(len(store))],
                             [{"x": 1}, {"x": "二"}, {"y": 3}, {"y": 4}])      # empty lines skipped, 2 per file
            store.save(Path(root) / "index")
            loaded = trainer.LineStore.load([a, b], Path(root) / "index")
            self.assertEqual(json.loads(loaded[3]), {"y": 4})
            copy = pickle.loads(pickle.dumps(loaded))                         # as DataLoader/pool workers get it
            self.assertEqual(json.loads(copy[1]), {"x": "二"})
            for opened in (store, loaded, copy):
                opened.close()
            del loaded, copy                                                  # release the memory maps (Windows)


if __name__ == "__main__":
    unittest.main()
