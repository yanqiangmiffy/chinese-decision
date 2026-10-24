import unittest

import torch
import torch.nn.functional as F

from decision.training.group_rlcd import rlcd_loss
from decision.losses import (RLCDLoss, RLCDLossConfig, masked_log_softmax, proper_score)


def random_decisions(sizes, slots, seed):
    generator = torch.Generator().manual_seed(seed)
    mask = torch.arange(slots)[None] < torch.tensor(sizes)[:, None]
    logits = torch.randn(len(sizes), slots, generator=generator)
    targets = masked_log_softmax(torch.randn(len(sizes), slots, generator=generator), mask)[1]
    return logits, targets, mask


class ProperScoreTests(unittest.TestCase):
    def test_target_distribution_scores_best(self):
        generator = torch.Generator().manual_seed(0)
        mask = torch.ones(1, 4, dtype=torch.bool)
        for ordinal in (False, True):
            ordinal_mask = torch.tensor([ordinal])
            target = torch.softmax(torch.randn(1, 4, generator=generator), -1)
            best = proper_score(target.log(), target, target, mask, ordinal_mask)
            others = torch.softmax(2 * torch.randn(512, 1, 4, generator=generator), -1)
            scores = proper_score(others.log(), others, target, mask, ordinal_mask)
            self.assertTrue((scores <= best + 1e-6).all())


class RLCDLossTests(unittest.TestCase):
    def test_ce_only_matches_soft_cross_entropy(self):
        logits, targets, mask = random_decisions([5, 5, 5], 5, seed=1)
        out = RLCDLoss(rlcd_weight=0.0)(logits, targets, mask)
        self.assertAlmostEqual(float(out.loss), float(F.cross_entropy(logits, targets)), places=6)
        self.assertIsNone(out.reward)

    def test_padded_candidates_get_no_gradient(self):
        for estimator in ("reinforce", "reparam"):
            logits, targets, mask = random_decisions([1, 2, 4, 6], 6, seed=2)
            logits.requires_grad_()
            ordinal = torch.tensor([False, True, False, True])
            out = RLCDLoss(estimator=estimator)(logits, targets, mask, ordinal)
            out.loss.backward()
            self.assertTrue(torch.isfinite(out.loss))
            self.assertEqual(float(logits.grad[~mask].abs().sum()), 0.)
            # Softmax is shift invariant, so each decision's gradient sums to zero.
            self.assertTrue(torch.allclose(logits.grad.sum(-1), torch.zeros(4), atol=1e-6))

    def test_reductions_agree(self):
        logits, targets, mask = random_decisions([3, 4], 4, seed=3)
        losses = {}
        for reduction in ("none", "sum", "mean"):
            generator = torch.Generator().manual_seed(0)
            losses[reduction] = RLCDLoss(reduction=reduction)(logits, targets, mask, generator=generator).loss
        self.assertEqual(losses["none"].shape, (2,))
        self.assertAlmostEqual(float(losses["sum"]), float(losses["none"].sum()), places=5)
        self.assertAlmostEqual(float(losses["mean"]), float(losses["none"].mean()), places=5)

    def test_estimators_agree_in_expectation(self):
        logits, targets, mask = random_decisions([5, 3, 2, 4], 5, seed=4)
        ordinal = torch.tensor([True, False, False, True])
        gradients = {}
        for estimator in ("reinforce", "reparam"):
            z = logits.clone().requires_grad_()
            loss_fn = RLCDLoss(ce_weight=0.0, estimator=estimator, num_samples=100_000, sigma=0.5)
            loss_fn(z, targets, mask, ordinal, generator=torch.Generator().manual_seed(5)).loss.backward()
            gradients[estimator] = z.grad[mask]
        similarity = F.cosine_similarity(gradients["reinforce"], gradients["reparam"], dim=0)
        self.assertGreater(float(similarity), 0.999)

    def test_chinese_jev_preset_reproduces_project_term(self):
        sizes, kinds = [3, 4, 2, 1], ["choice", "score", "noul", "choice"]
        raw, targets, valid = random_decisions(sizes, 26, seed=6)
        batch = [{"kind": kind, "target": [0.] * size} for kind, size in zip(kinds, sizes)]
        ordinal = torch.tensor([kind == "score" for kind in kinds])

        reference = raw.clone().requires_grad_()
        logits = reference.masked_fill(~valid, -1e4)
        torch.manual_seed(0)
        term, _ = rlcd_loss(logits, targets, batch, sigma=0.3, group_size=4)
        expected = -(targets * logits.log_softmax(-1)).sum(-1).mean() + term
        expected.backward()

        candidate = raw.clone().requires_grad_()
        torch.manual_seed(0)
        out = RLCDLoss(RLCDLossConfig.chinese_jev())(candidate.masked_fill(~valid, -1e4), targets,
                                                    valid, ordinal, sigma=0.3)
        out.loss.backward()
        self.assertAlmostEqual(out.loss.item(), expected.item(), places=5)
        self.assertTrue(torch.allclose(candidate.grad, reference.grad, atol=1e-5))

    def test_rejects_target_mass_on_padding(self):
        logits, targets, mask = random_decisions([2, 3], 3, seed=7)
        targets[0] = torch.tensor([.5, .25, .25])
        with self.assertRaises(ValueError):
            RLCDLoss()(logits, targets, mask)


if __name__ == "__main__":
    unittest.main()
