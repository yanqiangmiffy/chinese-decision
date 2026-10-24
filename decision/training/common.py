"""Training-time helpers shared by the training, export and evaluation scripts."""
import shutil

import torch

from ..configuration import LETTERS
from ..data.batching import greedy_batches
from ..metrics import summarize

# Full attention, Gated DeltaNet linear attention, and MLP projections of Qwen3.5.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


def with_last(stream):
    """Yield `(item, is_last)` pairs."""
    iterator = iter(stream)
    try:
        current = next(iterator)
    except StopIteration:
        return
    for following in iterator:
        yield current, False
        current = following
    yield current, True


@torch.inference_mode()
def evaluate_decisions(model, processor, items, batch_size=8, token_budget=8192):
    """Score encoded decisions with a `DecisionModel`; returns the metric summary and per-item logits."""
    was_training = model.training
    model.eval()
    letters = model.letter_tensor(processor)
    rows = []
    for batch in greedy_batches(sorted(items, key=lambda e: len(e["input_ids"])), batch_size, token_budget):
        logits, _ = model.score_decisions(batch, processor, letters)
        for e, values in zip(batch, logits.cpu().tolist()):
            rows.append({k: e[k] for k in ("target", "domain", "kind", "source", "hard", "case_id")}
                        | {"logits": values[:len(e["target"])]})
    model.train(was_training)
    return summarize(rows), rows


def fit_temperature(rows):
    """Fit one softmax temperature on hard-label rows by minimizing cross-entropy."""
    hard = [row for row in rows if row["hard"]]
    if not hard:
        raise ValueError("no hard calibration targets")
    logits = torch.full((len(hard), len(LETTERS)), -1e4)
    labels = torch.tensor([max(range(len(row["target"])), key=row["target"].__getitem__) for row in hard])
    for i, row in enumerate(hard):
        logits[i, :len(row["logits"])] = torch.tensor(row["logits"])
    log_temperature = torch.tensor(0., requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp().clamp(0.05, 20), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20).detach()), len(hard)


def replace_directory(source, target):
    """Move `source` to `target`, keeping the old target until the move succeeds."""
    # Only remove a previous artifact in the explicitly selected run directory.
    previous = target.with_name(target.name + ".previous")
    if previous.exists():
        if previous.parent.resolve() != source.parent.resolve():
            raise ValueError("backup destination escapes run directory")
        shutil.rmtree(previous)
    if target.exists():
        if target.parent.resolve() != source.parent.resolve():
            raise ValueError("artifact destination escapes run directory")
        target.rename(previous)
    source.rename(target)
    if previous.exists():
        shutil.rmtree(previous)
