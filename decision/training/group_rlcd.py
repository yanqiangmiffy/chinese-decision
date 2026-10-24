"""RLCD: a policy-gradient term over a proper-scoring-rule reward on candidate logits.

Adapted from gulucaptain/Chinese-Jev (Apache-2.0), revision aed6193be569:
`modeling.proper_reward` and `train.decision_loss`. Changes from the original:
- Operates on this project's option-letter logits; validity and question type come
  from the encoded batch items instead of Chinese-Jev's marker mask and type ids.
- Returns only the RLCD term; the distribution cross-entropy stays in scripts/train.py.
- The spherical weight defaults to 0.75, the value Chinese-Jev's training passes.
See THIRD_PARTY_NOTICES.md.

Each step draws `group_size` zero-mean Gaussian perturbations of the detached logits,
scores every perturbed distribution with log + spherical scores (minus a ranked
probability score for ordered `score` questions), and pushes the logits toward the
perturbations that score above their group's average.
"""
import torch

LOG_FLOOR = -9.21
SPHERICAL_WEIGHT = 0.75
RPS_WEIGHT = 1.0


def proper_reward(q, target, is_score, mask, w_sph=SPHERICAL_WEIGHT, w_rps=RPS_WEIGHT, log_floor=LOG_FLOOR):
    """Strictly proper scoring-rule reward for distributions `q` [..., B, K] against `target` [B, K]."""
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    spherical = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    reward = (target * logq).sum(-1) + w_sph * spherical
    if is_score.any():
        # Ranked probability score: squared CDF distance, so far-off levels cost more.
        k = mask.sum(-1).clamp(min=2).float()
        rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
        reward = reward - w_rps * rps * is_score.float()
    return reward


def rlcd_sigma(start, end, fraction):
    """Perturbation scale, annealed linearly over training progress `fraction` in [0, 1]."""
    return start + (end - start) * min(max(fraction, 0.), 1.)


def rlcd_loss(logits, targets, batch, sigma, group_size):
    """RLCD term for one micro-batch, and the mean reward of its perturbations.

    `logits` and `targets` are `forward_batch` outputs [B, 26]; `batch` holds the encoded
    items, which give each row's candidate count and question type.
    """
    if group_size < 2:
        raise ValueError("RLCD needs group_size >= 2: the advantage subtracts a group mean")
    device = logits.device
    sizes = torch.tensor([len(e["target"]) for e in batch], device=device)
    mask = torch.arange(logits.shape[-1], device=device)[None, :] < sizes[:, None]
    is_score = torch.tensor([e["kind"] == "score" for e in batch], device=device)
    k = mask.sum(-1, keepdim=True).float()
    # Zero-mean noise over the valid options keeps every perturbation comparable.
    eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(q, targets.unsqueeze(0), is_score, mask)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    # Gaussian log-density surrogate: the only gradient path is (z - logits)^2.
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    return -(advantage * logp).mean(), reward.mean()
