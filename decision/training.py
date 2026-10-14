"""Training-time helpers shared by scripts/train.py and scripts/finalize.py."""
import shutil

import torch

from .data import greedy_batches
from .metrics import summarize
from .model import forward_batch

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
def evaluate(model, tokenizer, letters, items, batch_size=8, token_budget=8192):
    """Score encoded validation items; returns the metric summary and per-item logits."""
    model.eval()
    rows = []
    for batch in greedy_batches(sorted(items, key=lambda e: len(e["input_ids"])), batch_size, token_budget):
        logits, _ = forward_batch(model, tokenizer, letters, batch)
        for e, values in zip(batch, logits.cpu().tolist()):
            rows.append({k: e[k] for k in ("target", "domain", "kind", "source", "hard", "case_id")}
                        | {"logits": values[:len(e["target"])]})
    model.train()
    return summarize(rows), rows


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
