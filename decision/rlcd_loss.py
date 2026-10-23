"""RLCD loss for calibrated candidate distributions.

Reference PyTorch implementation of the RLCD-style objective used by public
reproductions (Chinese-Jev, CalOn, OpenJev-RLCD). It is not TypeSafe's unpublished
Jev training code. The module depends only on PyTorch and plugs into any model that
produces one logit per answer candidate.

Objective
---------
For one decision with candidate logits ``z``, a mask of valid candidates and a target
distribution ``t`` over those candidates::

    p      = softmax(z)                                  # over valid candidates only
    loss   = ce_weight * CE(t, p) + brier_weight * Brier(t, p) + rlcd_weight * L_rlcd
    L_rlcd = -E_eps[ S(softmax(z + sigma * eps), t) ]

``eps`` is standard Gaussian noise projected onto the zero-sum subspace of the valid
candidates: softmax is shift invariant, so the removed direction could never change
the reward and would only add variance. ``S`` is a strictly proper scoring rule,
higher is better::

    S(q, t) = sum_k t_k * log(q_k)                       logarithmic score
            + spherical_weight * <t, q> / ||q||_2        spherical score
            - rps_weight * RPS(q, t)                     ranked probability score,
                                                         ordinal decisions only

Hard labels are one-hot targets; soft targets must already sum to one.

Estimators
----------
The expectation is estimated with ``num_samples`` perturbations per decision.

``"reinforce"``  Score-function (policy-gradient) estimator. Each perturbed logit
                 vector is an action of the Gaussian policy ``N(z, sigma^2 I)``. The
                 reward is detached; gradients reach ``z`` only through the action's
                 log-density, weighted by a leave-one-out advantage. This is the RL form.
``"reparam"``    Pathwise estimator of the same objective: gradients flow through
                 ``z + sigma * eps``. Lower variance; train it as the control that shows
                 whether the policy-gradient form itself adds anything.

Both estimate the gradient of the same Gaussian-smoothed score. For ``sigma > 0`` its
maximiser is close to, but not exactly, ``p = t``; as ``sigma -> 0`` it becomes the
proper score of the clean distribution. Anneal ``sigma`` with :func:`linear_schedule`.

Shapes
------
``B`` decisions, ``K`` padded candidate slots, ``G = num_samples``.
``logits``, ``targets`` and ``candidate_mask`` are ``[B, K]``; ``ordinal_mask`` is ``[B]``.

Example
-------
>>> loss_fn = RLCDLoss(RLCDLossConfig(estimator="reinforce", num_samples=8))
>>> out = loss_fn(logits, targets, candidate_mask, ordinal_mask, sigma=0.3)
>>> out.loss.backward()
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = [
    "RLCDLoss",
    "RLCDLossConfig",
    "RLCDLossOutput",
    "leave_one_out_advantages",
    "linear_schedule",
    "logarithmic_score",
    "masked_log_softmax",
    "proper_score",
    "ranked_probability_score",
    "sample_zero_sum_noise",
    "spherical_score",
]


@dataclass
class RLCDLossConfig:
    """Hyper-parameters of :class:`RLCDLoss`.

    Args:
        ce_weight: Weight of the cross-entropy on the clean distribution.
        rlcd_weight: Weight of the RLCD term. ``0`` disables sampling and leaves plain
            distribution cross-entropy.
        estimator: ``"reinforce"`` or ``"reparam"``; see the module docstring.
        num_samples: Perturbations per decision. At least 2, for the leave-one-out baseline.
        sigma: Default perturbation scale. ``RLCDLoss.forward`` accepts an annealed value.
        spherical_weight: Weight of the spherical score inside the reward.
        rps_weight: Weight of the ranked probability score for ordinal decisions.
        brier_weight: Weight of an extra Brier term on the clean distribution.
        log_score_floor: Optional lower clamp on ``log q`` inside the reward. It bounds the
            reward, and so the policy-gradient variance, but the score stops being strictly
            proper below the floor. ``None`` disables it.
        normalize_advantages: Divide advantages by their standard deviation over the whole
            batch. This acts as an adaptive step size; the gradient is no longer unbiased.
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` over decisions.
    """

    ce_weight: float = 1.0
    rlcd_weight: float = 1.0
    estimator: Literal["reinforce", "reparam"] = "reinforce"
    num_samples: int = 8
    sigma: float = 0.2
    spherical_weight: float = 0.75
    rps_weight: float = 1.0
    brier_weight: float = 0.0
    log_score_floor: float | None = None
    normalize_advantages: bool = False
    reduction: Literal["mean", "sum", "none"] = "mean"

    def __post_init__(self):
        if self.estimator not in ("reinforce", "reparam"):
            raise ValueError(f"unknown estimator: {self.estimator!r}")
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction: {self.reduction!r}")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2 for the leave-one-out baseline")
        weights = (self.ce_weight, self.rlcd_weight, self.spherical_weight,
                   self.rps_weight, self.brier_weight)
        if any(not 0 <= w < float("inf") for w in weights):
            raise ValueError("loss weights must be finite and non-negative")
        if not 0 < self.sigma < float("inf"):
            raise ValueError("sigma must be finite and positive")
        if self.log_score_floor is not None and not self.log_score_floor < 0:
            raise ValueError("log_score_floor must be negative")

    @classmethod
    def chinese_jev(cls, **overrides) -> RLCDLossConfig:
        """Settings that reproduce this repository's ``decision/rlcd.py`` term.

        Group size 4, log score floored at log(1e-4), batch-normalised advantages, and the
        RLCD term added to cross-entropy with weight 1. Anneal ``sigma`` from 0.4 to 0.1.
        """
        preset = dict(num_samples=4, sigma=0.4, log_score_floor=-9.21, normalize_advantages=True)
        return cls(**(preset | overrides))


@dataclass
class RLCDLossOutput:
    """Result of :class:`RLCDLoss`.

    Only ``loss`` carries gradients. The other fields are detached means over the batch,
    for logging. With the ``"reinforce"`` estimator, ``rlcd`` is the surrogate value, which
    is not the objective; monitor ``reward`` instead.
    """

    loss: Tensor
    ce: Tensor
    brier: Tensor
    rlcd: Tensor
    reward: Tensor | None = None


def masked_log_softmax(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Log-probabilities and probabilities over the valid candidates.

    ``logits`` is ``[..., K]`` and ``mask`` broadcasts to it. Padded slots get probability 0
    and log-probability 0, so weighted sums over the last dimension can ignore the mask.
    """
    log_probs = logits.masked_fill(~mask, float("-inf")).log_softmax(dim=-1)
    probs = log_probs.exp()
    return log_probs.masked_fill(~mask, 0.0), probs


def logarithmic_score(log_probs: Tensor, targets: Tensor, floor: float | None = None) -> Tensor:
    """``sum_k t_k log q_k``; higher is better."""
    if floor is not None:
        log_probs = log_probs.clamp_min(floor)
    return (targets * log_probs).sum(-1)


def spherical_score(probs: Tensor, targets: Tensor) -> Tensor:
    """``<t, q> / ||q||_2``; higher is better."""
    return (targets * probs).sum(-1) / probs.norm(dim=-1).clamp_min(1e-12)


def ranked_probability_score(probs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Squared distance between cumulative distributions, divided by ``K - 1``.

    Lower is better, and a decision with a single level scores 0. Valid levels must be
    in their natural order along the last dimension.
    """
    error = (probs.cumsum(-1) - targets.cumsum(-1)).square() * mask
    return error.sum(-1) / (mask.sum(-1) - 1).clamp_min(1)


def proper_score(log_probs: Tensor, probs: Tensor, targets: Tensor, mask: Tensor,
                 ordinal_mask: Tensor | None = None, *, spherical_weight: float = 0.75,
                 rps_weight: float = 1.0, log_score_floor: float | None = None) -> Tensor:
    """Reward of a candidate distribution against its target; higher is better.

    A non-negative combination of strictly proper scores is strictly proper, so the
    expected reward is maximised by reporting the target distribution itself.
    """
    score = logarithmic_score(log_probs, targets, log_score_floor)
    score = score + spherical_weight * spherical_score(probs, targets)
    if ordinal_mask is not None and rps_weight > 0:
        rps = ranked_probability_score(probs, targets, mask)
        score = score - rps_weight * ordinal_mask.to(rps.dtype) * rps
    return score


def sample_zero_sum_noise(num_samples: int, mask: Tensor,
                          generator: torch.Generator | None = None) -> Tensor:
    """Standard Gaussian noise ``[G, B, K]``, zero on padded slots, summing to 0 per decision."""
    noise = torch.randn((num_samples, *mask.shape), generator=generator, device=mask.device) * mask
    mean = noise.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
    return (noise - mean) * mask


def leave_one_out_advantages(rewards: Tensor, normalize: bool = False, eps: float = 1e-6) -> Tensor:
    """``A_g = R_g - mean_{j != g} R_j`` along dim 0, detached.

    This equals ``G / (G - 1) * (R_g - mean_j R_j)``. The baseline excludes the sample it
    corrects, so the policy gradient stays unbiased.
    """
    rewards = rewards.detach()
    group = rewards.shape[0]
    advantages = (rewards - rewards.mean(0, keepdim=True)) * (group / (group - 1))
    if normalize:
        advantages = advantages / (advantages.std() + eps)
    return advantages


def linear_schedule(start: float, end: float, progress: float) -> float:
    """Interpolate from ``start`` to ``end`` as ``progress`` goes from 0 to 1 (clamped)."""
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


class RLCDLoss(nn.Module):
    """Cross-entropy plus an RLCD term over Gaussian-perturbed candidate logits.

    Args:
        config: Loss hyper-parameters; defaults to :class:`RLCDLossConfig`.
        **overrides: Fields that replace those of ``config``.
    """

    def __init__(self, config: RLCDLossConfig | None = None, **overrides):
        super().__init__()
        self.config = replace(config or RLCDLossConfig(), **overrides)

    def forward(self, logits: Tensor, targets: Tensor, candidate_mask: Tensor | None = None,
                ordinal_mask: Tensor | None = None, *, sigma: float | None = None,
                generator: torch.Generator | None = None) -> RLCDLossOutput:
        """Compute the loss for one batch of decisions.

        Args:
            logits: Candidate logits ``[B, K]``. Any dtype; computed in float32.
            targets: Target distributions ``[B, K]``, zero on padded slots.
            candidate_mask: Valid candidates ``[B, K]`` (bool). ``None`` means all valid.
            ordinal_mask: Decisions whose candidates are ordered levels ``[B]`` (bool).
            sigma: Perturbation scale for this step; defaults to ``config.sigma``.
            generator: Random generator for the perturbations, on the logits' device.
        """
        cfg = self.config
        logits = logits.float()
        targets = targets.to(logits.dtype)
        mask = torch.ones_like(logits, dtype=torch.bool) if candidate_mask is None else candidate_mask
        _check_inputs(logits, targets, mask, ordinal_mask)

        log_probs, probs = masked_log_softmax(logits, mask)
        ce = -(targets * log_probs).sum(-1)
        brier = (probs - targets).square().sum(-1)
        per_decision = cfg.ce_weight * ce + cfg.brier_weight * brier
        rlcd = torch.zeros_like(ce)
        reward = None

        if cfg.rlcd_weight > 0:
            sigma = cfg.sigma if sigma is None else float(sigma)
            if not 0 < sigma < float("inf"):
                raise ValueError("sigma must be finite and positive")
            reinforce = cfg.estimator == "reinforce"
            noise = sigma * sample_zero_sum_noise(cfg.num_samples, mask, generator)
            # REINFORCE samples actions from a fixed policy; the pathwise estimator
            # differentiates through the sample.
            center = logits.detach() if reinforce else logits
            perturbed = center.unsqueeze(0) + noise                                 # [G, B, K]
            sample_log_probs, sample_probs = masked_log_softmax(perturbed, mask)
            rewards = proper_score(sample_log_probs, sample_probs, targets, mask, ordinal_mask,
                                   spherical_weight=cfg.spherical_weight, rps_weight=cfg.rps_weight,
                                   log_score_floor=cfg.log_score_floor)             # [G, B]
            if reinforce:
                advantages = leave_one_out_advantages(rewards, cfg.normalize_advantages)
                # log N(perturbed; logits, sigma^2 I) up to a constant that does not depend on logits.
                log_density = -((perturbed - logits.unsqueeze(0)).square() * mask).sum(-1) / (2 * sigma ** 2)
                rlcd = -(advantages * log_density).mean(0)
            else:
                rlcd = -rewards.mean(0)
            reward = rewards.detach().mean()
            per_decision = per_decision + cfg.rlcd_weight * rlcd

        if cfg.reduction == "mean":
            loss = per_decision.mean()
        elif cfg.reduction == "sum":
            loss = per_decision.sum()
        else:
            loss = per_decision
        return RLCDLossOutput(loss=loss, ce=ce.detach().mean(), brier=brier.detach().mean(),
                              rlcd=rlcd.detach().mean(), reward=reward)


def _check_inputs(logits: Tensor, targets: Tensor, mask: Tensor, ordinal_mask: Tensor | None) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, K], got {tuple(logits.shape)}")
    if targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("logits, targets and candidate_mask must share shape [B, K]")
    if mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be a bool tensor")
    if ordinal_mask is not None and (ordinal_mask.shape != logits.shape[:1] or ordinal_mask.dtype != torch.bool):
        raise ValueError("ordinal_mask must be a bool tensor of shape [B]")
    if not mask.any(-1).all():
        raise ValueError("every decision needs at least one valid candidate")
    if (targets < 0).any() or targets.masked_select(~mask).any():
        raise ValueError("targets must be non-negative and zero on padded candidates")
    if not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]), atol=1e-4, rtol=0):
        raise ValueError("each target distribution must sum to 1")


def demo(steps: int = 400, seed: int = 0) -> None:
    """Fit free logits to random soft targets and report how close each variant gets.

    Cross-entropy alone is already optimal for free logits, so this checks that every
    variant trains stably and lands near the target; it says nothing about which one
    calibrates a real model better.
    """
    generator = torch.Generator().manual_seed(seed)
    batch, slots = 512, 6
    sizes = torch.randint(1, slots + 1, (batch,), generator=generator)
    mask = torch.arange(slots)[None] < sizes[:, None]
    targets = masked_log_softmax(2 * torch.randn(batch, slots, generator=generator), mask)[1]
    ordinal = torch.rand(batch, generator=generator) < 0.3
    log_targets = targets.clamp_min(1e-12).log()

    variants = {
        "ce only": RLCDLossConfig(rlcd_weight=0.0),
        "reinforce": RLCDLossConfig(estimator="reinforce"),
        "reparam": RLCDLossConfig(estimator="reparam"),
        "chinese_jev preset": RLCDLossConfig.chinese_jev(),
    }
    print(f"{'variant':<20}{'KL(t||p)':>10}{'mean |p-t|':>12}{'reward':>9}")
    for name, config in variants.items():
        loss_fn = RLCDLoss(config)
        logits = torch.zeros(batch, slots, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.05)
        noise = torch.Generator().manual_seed(seed + 1)
        for step in range(steps):
            sigma = linear_schedule(0.4, 0.1, step / (steps - 1))
            out = loss_fn(logits, targets, mask, ordinal, sigma=sigma, generator=noise)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
        log_probs, probs = masked_log_softmax(logits.detach(), mask)
        kl = (targets * (log_targets - log_probs)).sum(-1).mean()
        reward = "-" if out.reward is None else f"{out.reward:.4f}"
        print(f"{name:<20}{kl:>10.4f}{(probs - targets).abs().sum(-1).mean():>12.4f}{reward:>9}")


if __name__ == "__main__":
    demo()
