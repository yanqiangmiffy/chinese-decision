import unittest

import torch

from decision.rlcd import proper_reward, rlcd_loss, rlcd_sigma


def items(*kinds_and_sizes):
    return [{"kind": kind, "target": [0.] * size} for kind, size in kinds_and_sizes]


class ProperRewardTests(unittest.TestCase):
    def test_reporting_the_target_distribution_scores_best(self):
        generator = torch.Generator().manual_seed(0)
        mask = torch.ones(1, 4, dtype=torch.bool)
        for kind in (False, True):
            is_score = torch.tensor([kind])
            target = torch.softmax(torch.randn(1, 4, generator=generator), -1)
            best = proper_reward(target, target, is_score, mask)
            others = torch.softmax(torch.randn(256, 1, 4, generator=generator) * 2, -1)
            self.assertTrue((proper_reward(others, target, is_score, mask) <= best + 1e-6).all())

    def test_rps_only_penalizes_score_questions(self):
        q = torch.tensor([[.1, .2, .7]])
        target = torch.tensor([[1., 0., 0.]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        gap = proper_reward(q, target, torch.tensor([False]), mask) - proper_reward(q, target, torch.tensor([True]), mask)
        # CDF differences .9 and .7 over K - 1 = 2 levels.
        self.assertAlmostEqual(float(gap), (.9 ** 2 + .7 ** 2) / 2, places=6)

    def test_sigma_anneals_linearly_and_clamps(self):
        self.assertAlmostEqual(rlcd_sigma(.4, .1, 0.), .4)
        self.assertAlmostEqual(rlcd_sigma(.4, .1, .5), .25)
        self.assertAlmostEqual(rlcd_sigma(.4, .1, 2.), .1)


class RlcdLossTests(unittest.TestCase):
    def test_gradient_reaches_only_valid_candidates(self):
        torch.manual_seed(0)
        batch = items(("choice", 3), ("score", 4), ("noul", 2))
        raw = torch.randn(3, 26, requires_grad=True)
        valid = torch.arange(26)[None, :] < torch.tensor([3, 4, 2])[:, None]
        logits = raw.masked_fill(~valid, -1e4)
        targets = torch.zeros(3, 26)
        targets[0, 1] = targets[1, 2] = targets[2, 0] = 1.
        loss, reward = rlcd_loss(logits, targets, batch, sigma=.4, group_size=4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(reward))
        self.assertEqual(float(raw.grad[~valid].abs().sum()), 0.)
        self.assertGreater(float(raw.grad[valid].abs().sum()), 0.)

    def test_same_seed_gives_same_loss(self):
        logits, targets = torch.randn(2, 26), torch.zeros(2, 26)
        targets[:, 0] = 1.
        batch = items(("choice", 5), ("noul", 2))
        def run():
            torch.manual_seed(7)
            return rlcd_loss(logits, targets, batch, sigma=.3, group_size=4)[0]
        self.assertEqual(float(run()), float(run()))

    def test_group_of_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "group_size"):
            rlcd_loss(torch.zeros(1, 26), torch.zeros(1, 26), items(("noul", 2)), sigma=.4, group_size=1)


if __name__ == "__main__":
    unittest.main()
