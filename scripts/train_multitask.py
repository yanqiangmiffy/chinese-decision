"""Joint LoRA fine-tuning of the decision model on classification decisions and span extraction.

One GPU, small data (see scripts/build_multitask_data.py); scripts/train_multitask_ddp.py is the
multi-GPU, large-data version. Both tasks share the backbone and its LoRA adapter:
- decisions (classification): option-letter logits at the last position, as in scripts/train.py;
- extraction: a span head over last-layer states, one [none, types] choice per candidate span
  (decision.modeling.SpanHead).
Every epoch mixes both tasks' micro-batches in proportion to their data and shuffles them.
Both heads use decision.losses.RLCDLoss: plain cross-entropy unless --rlcd is set.

    python scripts/train_multitask.py --base models/chinese-decision-0.8b \\
        --data datasets/multitask-mini --out runs/multitask-0.8b

`--eval-only` scores the decision test set with the unmodified base model (no adapter, no span head).
"""
import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from decision.data.batching import greedy_batches
from decision.losses import RLCDLoss, RLCDLossConfig, SpanLoss, linear_schedule
from decision.metrics import entity_metrics, summarize
from decision.modeling import DecisionModel
from decision.processing import DecisionProcessor, examples_for
from decision.training.common import evaluate_decisions
from decision.utils import log, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="models/chinese-decision-0.8b")
    parser.add_argument("--data", default="datasets/multitask-mini")
    parser.add_argument("--out", default="runs/multitask-0.8b")
    parser.add_argument("--tasks", default="decision,extraction", help="comma-separated subset, for ablations")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--head-lr", type=float, default=5e-4, help="span head learning rate")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--decision-batch", type=int, default=16)
    parser.add_argument("--extraction-batch", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=12288, help="unpadded tokens per micro-batch")
    parser.add_argument("--extraction-weight", type=float, default=1.0)
    parser.add_argument("--none-weight", type=float, default=0.2, help="loss weight of spans whose answer is none")
    parser.add_argument("--max-span-units", type=int, default=20)
    parser.add_argument("--no-echo", dest="echo", action="store_false", help="spans read the only copy of the text")
    parser.add_argument("--rlcd", action="store_true", help="add the RLCD term to both heads")
    parser.add_argument("--rlcd-weight", type=float, default=1.0)
    parser.add_argument("--rlcd-samples", type=int, default=4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=0, help="dev evaluation interval in steps (0: per epoch)")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="rows per file, for smoke tests")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.tasks = [task for task in args.tasks.split(",") if task]
    if not set(args.tasks) <= {"decision", "extraction"} or not args.tasks:
        parser.error("--tasks takes decision and/or extraction")
    return args


def read_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def decision_examples(path, limit=0):
    return [example for case in read_jsonl(path, limit) for example in examples_for(case, "general")]


def extraction_rows(data, split, limit=0):
    return {path.name.split(".")[0]: read_jsonl(path, limit)
            for path in sorted((Path(data) / "extraction").glob(f"*.{split}.jsonl"))}


def encoded(items):
    return [item for item in items if item is not None]


def decision_groups(rows):
    """Metrics per source; every other CJ source is pooled into `cj_other`."""
    groups = {}
    for row in rows:
        source = row["source"]
        groups.setdefault("cj_other" if source.startswith("cj_other:") else source, []).append(row)
    return {name: summarize(items)["all"] for name, items in sorted(groups.items())}


def evaluate_all(model, processor, decision_items, extraction_items, args, none_shift):
    result = {}
    if decision_items:
        _, rows = evaluate_decisions(model, processor, decision_items, batch_size=32, token_budget=16384)
        result["decision"] = decision_groups(rows)
    if extraction_items and model.span_head is not None:
        result["extraction"] = {}
        for source, items in extraction_items.items():
            predictions = model.predict_spans(items, processor, none_shift=none_shift, token_budget=args.token_budget)
            result["extraction"][source] = entity_metrics(items, predictions)
    return result


def epoch_batches(args, rng, processor, decision_train, extraction_train):
    """This epoch's micro-batches of both tasks, shuffled together; choices and type lists are re-shuffled."""
    batches = []
    if "decision" in args.tasks:
        items = encoded(processor.encode_example(example, rng) for example in decision_train)
        items.sort(key=lambda e: len(e["input_ids"]))
        batches += [("decision", batch) for batch in greedy_batches(items, args.decision_batch, args.token_budget)]
    if "extraction" in args.tasks:
        items = [item for rows in extraction_train.values() for row in rows
                 for item in [processor.encode_extraction(row, rng)] if item is not None and item["cands"]]
        items.sort(key=lambda e: len(e["input_ids"]))
        batches += [("extraction", batch) for batch in greedy_batches(items, args.extraction_batch, args.token_budget)]
    rng.shuffle(batches)
    return batches


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", vars(args))
    processor = DecisionProcessor.from_pretrained(args.base, max_span_units=args.max_span_units, echo=args.echo)
    model = DecisionModel.from_pretrained(args.base, device=args.device)
    letters = model.letter_tensor(processor)

    data = Path(args.data)
    decision_test = encoded(processor.encode_example(e)
                            for e in decision_examples(data / "decision/test.cases.jsonl", args.limit))
    decision_dev = encoded(processor.encode_example(e)
                           for e in decision_examples(data / "decision/dev.cases.jsonl", args.limit))
    extraction_dev = {s: encoded(processor.encode_extraction(r) for r in rows)
                      for s, rows in extraction_rows(args.data, "dev", args.limit).items()}
    extraction_test = {s: encoded(processor.encode_extraction(r) for r in rows)
                       for s, rows in extraction_rows(args.data, "test", args.limit).items()}
    if args.eval_only:
        metrics = evaluate_all(model, processor, decision_test, {}, args, 0.0)
        write_json(out / "metrics_base.json", metrics)
        log("base_metrics", **metrics)
        return

    decision_train = decision_examples(data / "decision/train.cases.jsonl", args.limit)
    extraction_train = extraction_rows(args.data, "train", args.limit)
    lora_params = model.enable_lora(r=args.lora_r)
    if args.grad_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.backbone.enable_input_require_grads()
    loss_config = RLCDLossConfig(rlcd_weight=args.rlcd_weight if args.rlcd else 0.0, num_samples=args.rlcd_samples)
    decision_loss = RLCDLoss(loss_config)
    span_loss = SpanLoss(loss_config, args.none_weight)
    head = (model.add_span_head(args.max_span_units, none_shift=span_loss.none_shift, echo=args.echo)
            if "extraction" in args.tasks else None)
    model.train()
    groups = [{"params": lora_params, "lr": args.lr}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": args.head_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    peak = [group["lr"] for group in groups]

    rng = random.Random(args.seed)
    plan = [epoch_batches(args, rng, processor, decision_train, extraction_train) for _ in range(args.epochs)]
    total = sum(map(len, plan))
    log("training_start", steps=total, decision_examples=len(decision_train),
        extraction_rows={k: len(v) for k, v in extraction_train.items()},
        batches=dict(Counter(task for epoch in plan for task, _ in epoch)), none_shift=span_loss.none_shift)
    step, started, recent, history = 0, time.time(), Counter(), []
    for epoch, batches in enumerate(plan):
        for task, batch in batches:
            scale = min(1.0, (step + 1) / args.warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
            for group, value in zip(optimizer.param_groups, peak):
                group["lr"] = value * scale
            sigma = linear_schedule(args.sigma_start, args.sigma_end, step / max(total - 1, 1))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if task == "decision":
                    logits, targets = model.score_decisions(batch, processor, letters)
                    sizes = torch.tensor([len(e["target"]) for e in batch], device=device)
                    mask = torch.arange(logits.shape[-1], device=device)[None] < sizes[:, None]
                    ordinal = torch.tensor([e["kind"] == "score" for e in batch], device=device)
                    out_ = decision_loss(logits, targets, mask, ordinal, sigma=sigma)
                    loss = out_.loss
                else:
                    tensors = processor.collate_extraction(batch).to(device)
                    loss, out_ = span_loss(model.span_utilities(tensors), tensors, sigma=sigma)
                    loss = args.extraction_weight * loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {task} loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            recent[f"{task}_loss"] += loss.item()
            recent[f"{task}_ce"] += out_.ce.item()
            recent[f"{task}_n"] += 1
            if step % 20 == 0 or step == total:
                fields = {k: round(recent[f"{k}_loss"] / recent[f"{k}_n"], 4) for k in ("decision", "extraction")
                          if recent[f"{k}_n"]}
                log("train", step=step, total=total, epoch=epoch, lr=optimizer.param_groups[0]["lr"],
                    grad_norm=round(norm.item(), 3), seconds=round(time.time() - started), **fields)
                recent.clear()
            if args.eval_every and step % args.eval_every == 0 and step < total:
                dev = evaluate_all(model, processor, decision_dev, extraction_dev, args, span_loss.none_shift)
                history.append({"step": step, "dev": dev})
                log("dev", step=step, **dev)
        dev = evaluate_all(model, processor, decision_dev, extraction_dev, args, span_loss.none_shift)
        history.append({"step": step, "epoch": epoch, "dev": dev})
        log("dev", step=step, epoch=epoch, **dev)
        write_json(out / "history.json", history)

    model.backbone.save_pretrained(out / "adapter")
    if head is not None:
        torch.save({"state_dict": head.state_dict(), "config": head.config, "none_shift": span_loss.none_shift,
                    "echo": args.echo}, out / "span_head.pt")
    test = evaluate_all(model, processor, decision_test, extraction_test, args, span_loss.none_shift)
    write_json(out / "metrics_test.json", test)
    log("test", seconds=round(time.time() - started), **test)


if __name__ == "__main__":
    main()
