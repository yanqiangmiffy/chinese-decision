import unittest

import torch
import torch.nn.functional as F

from decision.losses import RLCDLoss, RLCDLossConfig, linear_schedule, masked_log_softmax, proper_score


def legacy_rlcd_term(logits, targets, batch, sigma, group_size):
    """The `--rlcd` term earlier runs of scripts/train.py trained with, kept frozen as a reference.

    Adapted from gulucaptain/Chinese-Jev (Apache-2.0), revision aed6193be569: `modeling.proper_reward`
    and `train.decision_loss`; see THIRD_PARTY_NOTICES.md. `RLCDLossConfig.chinese_jev()` must match it.
    """
    sizes = torch.tensor([len(e["target"]) for e in batch])
    mask = torch.arange(logits.shape[-1])[None, :] < sizes[:, None]
    is_score = torch.tensor([e["kind"] == "score" for e in batch])
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1) * mask
    with torch.no_grad():
        target = targets.unsqueeze(0)
        logq = torch.log(q.clamp_min(1e-12)).clamp_min(-9.21)
        reward = (target * logq).sum(-1) + 0.75 * (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
        rps = (((q.cumsum(-1) - target.cumsum(-1)) ** 2) * mask).sum(-1) / (mask.sum(-1).clamp(min=2).float() - 1)
        reward = reward - rps * is_score.float()
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    return -(advantage * logp).mean(), reward.mean()


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

    def test_rps_only_penalizes_ordinal_decisions(self):
        q = torch.tensor([[.1, .2, .7]])
        target = torch.tensor([[1., 0., 0.]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        gap = (proper_score(q.log(), q, target, mask, torch.tensor([False]))
               - proper_score(q.log(), q, target, mask, torch.tensor([True])))
        # CDF differences .9 and .7 over K - 1 = 2 levels.
        self.assertAlmostEqual(float(gap), (.9 ** 2 + .7 ** 2) / 2, places=6)

    def test_sigma_anneals_linearly_and_clamps(self):
        self.assertAlmostEqual(linear_schedule(.4, .1, 0.), .4)
        self.assertAlmostEqual(linear_schedule(.4, .1, .5), .25)
        self.assertAlmostEqual(linear_schedule(.4, .1, 2.), .1)


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
        term, _ = legacy_rlcd_term(logits, targets, batch, sigma=0.3, group_size=4)
        expected = -(targets * logits.log_softmax(-1)).sum(-1).mean() + term
        expected.backward()

        candidate = raw.clone().requires_grad_()
        torch.manual_seed(0)
        out = RLCDLoss(RLCDLossConfig.chinese_jev())(candidate.masked_fill(~valid, -1e4), targets,
                                                    valid, ordinal, sigma=0.3)
        out.loss.backward()
        self.assertAlmostEqual(out.loss.item(), expected.item(), places=5)
        self.assertTrue(torch.allclose(candidate.grad, reference.grad, atol=1e-5))

    def test_same_global_seed_gives_same_loss(self):
        logits, targets, mask = random_decisions([5, 2], 6, seed=8)
        loss_fn = RLCDLoss(RLCDLossConfig.chinese_jev())

        def run():
            torch.manual_seed(7)
            return loss_fn(logits, targets, mask, sigma=0.3).loss.item()
        self.assertEqual(run(), run())

    def test_rejects_target_mass_on_padding(self):
        logits, targets, mask = random_decisions([2, 3], 3, seed=7)
        targets[0] = torch.tensor([.5, .25, .25])
        with self.assertRaises(ValueError):
            RLCDLoss()(logits, targets, mask)

    def test_group_of_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "num_samples"):
            RLCDLossConfig(num_samples=1)


if __name__ == "__main__":
    unittest.main()
