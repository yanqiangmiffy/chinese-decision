"""Multi-GPU joint training of the decision and span heads on a large mix such as CJ-IE.

Same model, heads, losses and batching rule as scripts/train_multitask.py, built for data that does
not fit that script's habit of encoding every epoch up front:
- rows stay as JSON lines; their token lengths are computed once (process pool, cached in the run
  directory) and every epoch's length-sorted batches are encoded on the fly by DataLoader workers,
  with choices and type lists re-shuffled per batch;
- every rank takes its share of the same shuffled batch list; LoRA and span-head gradients are
  all-reduced by hand (the heads reach into the backbone past a DDP wrapper's hooks). LoRA
  gradients average over every micro-batch of the step, span-head gradients over its extraction
  micro-batches;
- dev, test and extra decision sets are scored with every rank taking a shard;
- checkpoints (`checkpoint-<step>/adapter`, `span_head.pt`, `trainer_state.pt`) load with
  SpanExtractor.from_run, and --resume continues from one.

Keeping the original decision abilities (joint training on CJ-IE alone took CJ-Bench from 83.1% to
71.0%; see docs/UNIFIED_MODEL.md):
- RLCD is on by default for both heads (--no-rlcd for plain cross-entropy);
- --replay mixes CJ-Dataset training decisions (scripts/build_replay_data.py) into every epoch; on
  those batches a KL term (--kl-weight) ties the option distribution to the starting model, read
  with the LoRA adapter disabled, so no second model is loaded;
- --accumulate averages that many micro-batches per rank into each optimizer step; steps, warmup
  and the evaluation interval count optimizer steps;
- --cj-dev (CJ-Dataset validation, never CJ-Bench) is scored at step 0 and at every evaluation; the
  selected checkpoint maximizes the mean of new-task decision accuracy and macro extraction F1
  among those within --max-retention-drop points of the step-0 CJ-dev accuracy (the smallest drop
  wins if none is);
- the selected checkpoint is tested and exported as one unified model directory (decision/unified.py:
  merged weights and span head in one safetensors), its temperature fitted on --cj-calibration.

    export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1
    python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_multitask_ddp.py \\
        --data datasets/cj-ie --out runs/cj-ie-0.8b-v2 --replay datasets/cj-replay/train.cases.jsonl \\
        --cj-dev datasets/cj-replay/dev.cases.jsonl --cj-calibration datasets/cj-replay/calibration.cases.jsonl \\
        --retention datasets/multitask-mini/decision/test.cases.jsonl --accumulate 8
"""
import argparse
import datetime
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict

from decision.data import encode
from decision.extraction import (ExtractionEncoder, SpanHead, SpanLoss, backbone_hidden, collate_extraction,
                                 entity_metrics, predict)
from decision.metrics import summarize
from decision.model import forward_batch, letter_tokens, load_model, load_tokenizer
from decision.rlcd_loss import RLCDLoss, RLCDLossConfig, linear_schedule
from decision.schema import LETTERS, MAX_LENGTH, PROMPT_VERSION, examples_for
from decision.training import LORA_TARGETS, evaluate
from decision.unified import save_unified
from decision.utils import log, write_json

TASKS = ("decision", "replay", "extraction")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="models/chinese-decision-0.8b")
    parser.add_argument("--data", default="datasets/cj-ie")
    parser.add_argument("--out", required=True)
    parser.add_argument("--retention", default="", help="extra decision cases scored with dev and test")
    parser.add_argument("--replay", default="", help="CJ decision cases replayed with the KL anchor")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="KL(start model || model) on replay batches")
    parser.add_argument("--cj-dev", default="", help="independent decision validation for checkpoint selection")
    parser.add_argument("--cj-calibration", default="", help="decision cases for the exported temperature")
    parser.add_argument("--max-retention-drop", type=float, default=1.0,
                        help="largest CJ-dev accuracy drop (points) a selectable checkpoint may have")
    parser.add_argument("--export", default="model", help="unified model directory under --out ('' to skip)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=100, help="optimizer steps")
    parser.add_argument("--accumulate", type=int, default=1, help="micro-batches per rank per optimizer step")
    parser.add_argument("--decision-batch", type=int, default=16)
    parser.add_argument("--extraction-batch", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=12288)
    parser.add_argument("--extraction-weight", type=float, default=1.0)
    parser.add_argument("--none-weight", type=float, default=0.2)
    parser.add_argument("--max-span-units", type=int, default=20)
    parser.add_argument("--no-echo", dest="echo", action="store_false")
    parser.add_argument("--no-rlcd", dest="rlcd", action="store_false", help="plain cross-entropy for both heads")
    parser.add_argument("--rlcd-weight", type=float, default=1.0)
    parser.add_argument("--rlcd-samples", type=int, default=4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=500,
                        help="dev evaluation and checkpoint interval in optimizer steps (0: per epoch)")
    parser.add_argument("--resume", default="", help="checkpoint directory to continue from")
    parser.add_argument("--workers", type=int, default=6, help="DataLoader workers per rank")
    parser.add_argument("--length-procs", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="rows per file, for smoke tests")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.accumulate < 1:
        parser.error("--accumulate must be at least 1")
    return args


def read_lines(path, limit=0):
    lines = []
    with open(path, "rb") as stream:
        for line in stream:
            if line.strip():
                lines.append(line)
                if limit and len(lines) >= limit:
                    break
    return lines


def extraction_files(data, split):
    return sorted((Path(data) / "extraction").glob(f"*.{split}.jsonl"))


# --- token lengths, computed once in a process pool ------------------------------------------------

_WORKER = {}


def _init_lengths(base, max_span_units, echo):
    _WORKER["tokenizer"] = tokenizer = load_tokenizer(base)
    _WORKER["encoder"] = ExtractionEncoder(tokenizer, max_span_units, echo)


def _lengths(task_lines):
    """Per line: a list of example lengths (decisions) or one length (span rows); -1 = unusable."""
    task, lines = task_lines
    out = []
    for line in lines:
        if task == "extraction":
            item = _WORKER["encoder"].encode(json.loads(line))
            out.append(len(item["input_ids"]) if item is not None and item["cands"] else -1)
        else:
            items = [encode(e, _WORKER["tokenizer"], MAX_LENGTH) for e in examples_for(json.loads(line), "general")]
            out.append([len(i["input_ids"]) if i else -1 for i in items])
    return out


def compute_lengths(args, stores):
    chunks = [(task, lines[i:i + 2000]) for task in TASKS for i in range(0, len(stores[task]), 2000)
              for lines in [stores[task]]]
    with Pool(args.length_procs, initializer=_init_lengths, initargs=(args.base, args.max_span_units, args.echo)) as pool:
        results = pool.map(_lengths, chunks, chunksize=1)
    lengths = {task: [] for task in TASKS}
    for (task, _), values in zip(chunks, results):
        lengths[task].extend(values)
    return lengths


def train_files(args):
    files = [Path(args.data) / "decision/train.cases.jsonl", *extraction_files(args.data, "train")]
    return files + ([Path(args.replay)] if args.replay else [])


def load_lengths(args, out, rank, stores):
    """Rank 0 computes (or reuses) the cache; the other ranks wait for its file."""
    path = out / "lengths.json"
    key = {"base": str(args.base), "data": str(Path(args.data).resolve()), "limit": args.limit, "echo": args.echo,
           "files": {str(p): p.stat().st_size for p in train_files(args)}}
    if rank == 0:
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if not cached or cached["key"] != key:
            started = time.time()
            write_json(path, {"key": key, **compute_lengths(args, stores)})
            log("lengths", seconds=round(time.time() - started), **{t: len(stores[t]) for t in TASKS})
    while True:
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["key"] == key:
                return {task: cached[task] for task in TASKS}
        time.sleep(5)


# --- batch plan --------------------------------------------------------------------------------

def length_batches(order, lengths, batch_size, budget):
    """`decision.data.greedy_batches` on precomputed lengths: fill in length order up to the row or token limit."""
    batches, batch = [], []
    for i in order:
        if batch and (len(batch) >= batch_size or (len(batch) + 1) * lengths[i] > budget):
            batches.append(batch)
            batch = []
        batch.append(i)
    if batch:
        batches.append(batch)
    return batches


def flatten_decisions(lengths_by_line):
    """Per decision example: ((line, question), length)."""
    items = [(line, q) for line, lengths in enumerate(lengths_by_line) for q in range(len(lengths))]
    return items, [n for lengths in lengths_by_line for n in lengths]


def epoch_plan(args, epoch, lengths, world):
    """One epoch's micro-batches for every rank, `[(task, indices, seed)]` per rank, equal counts across ranks.

    `lengths[task]` holds one length per item (decision tasks already flattened per question).
    """
    rng, batches = random.Random(f"{args.seed}:{epoch}"), []
    for task in TASKS:
        size = args.extraction_batch if task == "extraction" else args.decision_batch
        usable = sorted((i for i, n in enumerate(lengths[task]) if n > 0), key=lambda i: lengths[task][i])
        batches += [(task, b) for b in length_batches(usable, lengths[task], size, args.token_budget)]
    rng.shuffle(batches)
    per_rank = len(batches) // world
    plan = [(task, indices, f"{args.seed}:{epoch}:{k}") for k, (task, indices) in enumerate(batches[:per_rank * world])]
    return [plan[r::world] for r in range(world)]


class Batches(torch.utils.data.Dataset):
    """Encodes one planned micro-batch per item; each worker builds its own tokenizer and encoder."""

    def __init__(self, plan, stores, decision_items, args):
        self.plan, self.stores, self.decision_items = plan, stores, decision_items
        self.settings = (args.base, args.max_span_units, args.echo)
        self.tokenizer = self.encoder = None

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, k):
        if self.encoder is None:
            base, units, echo = self.settings
            self.tokenizer = load_tokenizer(base)
            self.encoder = ExtractionEncoder(self.tokenizer, units, echo)
        task, indices, seed = self.plan[k]
        rng = random.Random(seed)
        if task == "extraction":
            items = [self.encoder.encode(json.loads(self.stores[task][i]), rng) for i in indices]
            return task, collate_extraction(items, self.tokenizer.pad_token_id)
        items = []
        for i in indices:
            line, question = self.decision_items[task][i]
            example = list(examples_for(json.loads(self.stores[task][line]), "general"))[question]
            items.append(encode(example, self.tokenizer, MAX_LENGTH, rng))
        return task, items


# --- losses, gradients and evaluation across ranks -------------------------------------------------

def kl_to_reference(logits, reference):
    """Mean KL(reference || model) over the batch; both are letter logits with invalid options at -1e4."""
    reference = torch.log_softmax(reference.float(), -1)
    return (reference.exp() * (reference - torch.log_softmax(logits.float(), -1))).sum(-1).mean()


def sync_gradients(lora_params, head_params, micro_batches, extraction_batches, world):
    """Average LoRA grads over all micro-batches of the step and head grads over its extraction micro-batches."""
    groups = [(lora_params, micro_batches)]
    if head_params:
        groups.append((head_params, max(1, extraction_batches)))
    for params, divisor in groups:
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        for dtype in sorted({g.dtype for g in grads}, key=str):       # the same order on every rank
            chosen = [(p, g) for p, g in zip(params, grads) if g.dtype == dtype]
            flat = torch.cat([g.reshape(-1) for _, g in chosen])
            if world > 1:
                dist.all_reduce(flat)
            flat /= divisor
            offset = 0
            for p, g in chosen:
                p.grad = flat[offset:offset + g.numel()].view_as(g).clone()
                offset += g.numel()
    if head_params and extraction_batches == 0:
        for p in head_params:
            p.grad = None


def gather(values, rank, world):
    if world == 1:
        return [values]
    collected = [None] * world if rank == 0 else None
    dist.gather_object(values, collected, dst=0)
    return collected


def decision_groups(rows):
    """Metrics per source plus `all`; every other CJ source is pooled into `cj_other`."""
    groups = {}
    for row in rows:
        source = row["source"]
        groups.setdefault("cj_other" if source.startswith("cj_other:") else source, []).append(row)
    result = {name: summarize(items)["all"] for name, items in sorted(groups.items())}
    if rows:
        result["all"] = summarize(rows)["all"]
    return result


def decision_rows(model, tokenizer, letters, lines, rank, world):
    """This rank's shard of decision lines, scored; rank 0 receives every row."""
    items = [encode(e, tokenizer, MAX_LENGTH) for line in lines[rank::world]
             for e in examples_for(json.loads(line), "general")]
    items = [i for i in items if i is not None]
    rows = evaluate(model, tokenizer, letters, items, batch_size=32, token_budget=16384)[1] if items else []
    collected = gather(rows, rank, world)
    return [row for part in collected for row in part] if rank == 0 else None


def evaluate_sets(model, head, tokenizer, letters, encoder, sets, args, rank, world, none_shift):
    """`sets`: {"decision": {name: lines}, "extraction": {source: lines}}; every rank scores its shard."""
    result = {}
    for name, lines in sets["decision"].items():
        rows = decision_rows(model, tokenizer, letters, lines, rank, world)
        if rank == 0:
            result[name] = decision_groups(rows)
    if sets["extraction"]:
        result["extraction"] = {}
        for source, lines in sets["extraction"].items():
            items = [encoder.encode(json.loads(line)) for line in lines[rank::world]]
            items = [i for i in items if i is not None]
            predictions = predict(model, head, items, tokenizer.pad_token_id, none_shift,
                                  token_budget=args.token_budget) if items else []
            light = [({"gold_entities": i["gold_entities"], "unrepresented": i["unrepresented"]}, p)
                     for i, p in zip(items, predictions)]
            collected = gather(light, rank, world)
            if rank == 0:
                pairs = [pair for part in collected for pair in part]
                result["extraction"][source] = entity_metrics([a for a, _ in pairs], [b for _, b in pairs])
    model.train()
    head.train()
    return result


def eval_sets(args, split):
    data, limit = Path(args.data), args.limit
    sets = {"decision": {"decision": read_lines(data / f"decision/{split}.cases.jsonl", limit)}, "extraction": {}}
    if args.retention:
        sets["decision"]["retention"] = read_lines(args.retention, limit)
    if args.cj_dev and split == "dev":
        sets["decision"]["cj_dev"] = read_lines(args.cj_dev, limit)
    for path in extraction_files(data, split):
        sets["extraction"][path.name.split(".")[0]] = read_lines(path, limit)
    return sets


def summary_scores(dev):
    """New-task decision accuracy, macro extraction F1, their mean, and CJ-dev accuracy/NLL."""
    decision = dev["decision"]["all"]["accuracy"]
    f1s = [m["f1"] for m in dev.get("extraction", {}).values()]
    extraction = sum(f1s) / len(f1s) if f1s else 0.0
    cj = dev.get("cj_dev", {}).get("all", {})
    return {"ie_decision_accuracy": decision, "ie_extraction_f1": extraction, "ie_score": (decision + extraction) / 2,
            "cj_accuracy": cj.get("accuracy"), "cj_hard_nll": cj.get("hard_nll")}


def select(history, max_drop):
    """The trained checkpoint to keep: best ie_score within the CJ-dev drop limit, else the smallest drop."""
    base = history[0]["scores"]["cj_accuracy"]
    trained = [h for h in history if h["step"] > 0 and h.get("checkpoint")]
    if not trained:
        return None
    if base is None:
        return max(trained, key=lambda h: h["scores"]["ie_score"])
    eligible = [h for h in trained if 100 * (base - h["scores"]["cj_accuracy"]) <= max_drop]
    if eligible:
        return max(eligible, key=lambda h: h["scores"]["ie_score"])
    return min(trained, key=lambda h: (base - h["scores"]["cj_accuracy"], -h["scores"]["ie_score"]))


def save_checkpoint(path, model, head, optimizer, step, none_shift, echo):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    torch.save({"state_dict": head.state_dict(), "config": head.config, "none_shift": none_shift, "echo": echo},
               path / "span_head.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, path / "trainer_state.pt")


def load_checkpoint(path, model, head, device):
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(Path(path) / "adapter/adapter_model.safetensors")))
    head.load_state_dict(torch.load(Path(path) / "span_head.pt", map_location=device)["state_dict"])


def fit_temperature(rows):
    """One softmax temperature on hard-label rows, as scripts/finalize.py fits the released models'."""
    hard = [row for row in rows if row["hard"]]
    if not hard:
        return 1.0, 0
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


def export(args, out, model, head, tokenizer, letters, selected, none_shift):
    """Merge the selected adapter, write one unified directory, and check it reproduces the trained heads."""
    probe = [encode(e, tokenizer, MAX_LENGTH) for line in read_lines(Path(args.data) / "decision/dev.cases.jsonl", 4)
             for e in examples_for(json.loads(line), "general")]
    model.eval()
    with torch.inference_mode():
        before, _ = forward_batch(model, tokenizer, letters, probe)
    merged = model.merge_and_unload(safe_merge=True).eval()
    with torch.inference_mode():
        after, _ = forward_batch(merged, tokenizer, letters, probe)
    delta = float((before - after).abs().max())
    if not torch.equal(before.argmax(-1), after.argmax(-1)):
        raise ValueError("merging the adapter changed probe decisions")
    temperature, hard_n = 1.0, 0
    if args.cj_calibration:
        _, rows = evaluate(merged, tokenizer, letters,
                           [i for line in read_lines(args.cj_calibration)
                            for i in [encode(e, tokenizer, MAX_LENGTH) for e in examples_for(json.loads(line), "general")]
                            if i is not None], batch_size=32, token_budget=16384)
        temperature, hard_n = fit_temperature(rows)
    target = out / args.export
    if target.exists():
        shutil.rmtree(target)
    base_config = json.loads((Path(args.base) / "decision_config.json").read_text()) \
        if (Path(args.base) / "decision_config.json").exists() else {}
    save_unified(merged, head, tokenizer, target,
                 {"model_name": f"{base_config.get('model_name', Path(args.base).name)}-ie", "temperature": temperature,
                  "calibration_hard_n": hard_n, "prompt_version": PROMPT_VERSION, "max_length": MAX_LENGTH,
                  "max_options": len(LETTERS), "source_run": str(out), "merge_probe_max_logit_delta": delta},
                 {"none_shift": none_shift, "echo": args.echo},
                 summary={"base": str(args.base), "selected": selected, "training_config": vars(args)})
    log("exported", path=str(target), temperature=temperature, calibration_hard_n=hard_n, merge_delta=delta)
    return target


# --- main ------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    distributed = "WORLD_SIZE" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if distributed else (0, 1)
    local = int(os.environ.get("LOCAL_RANK", 0))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(out / "config.json", vars(args) | {"world_size": world})

    tokenizer = load_tokenizer(args.base)
    encoder = ExtractionEncoder(tokenizer, args.max_span_units, args.echo)
    stores = {"decision": read_lines(Path(args.data) / "decision/train.cases.jsonl", args.limit),
              "replay": read_lines(args.replay, args.limit) if args.replay else [],
              "extraction": [line for path in extraction_files(args.data, "train") for line in read_lines(path, args.limit)]}
    # Before CUDA and NCCL start: the length pool forks.
    raw_lengths = load_lengths(args, out, rank, stores)
    decision_items, lengths = {}, {"extraction": raw_lengths["extraction"]}
    for task in ("decision", "replay"):
        decision_items[task], lengths[task] = flatten_decisions(raw_lengths[task])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if distributed:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    letters = letter_tokens(tokenizer, device)
    dev_sets, test_sets = eval_sets(args, "dev"), eval_sets(args, "test")

    model = load_model(args.base, device=str(device))
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=.05,
                                             task_type=TaskType.CAUSAL_LM, target_modules=LORA_TARGETS))
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    head = SpanHead(model.config.hidden_size, args.max_span_units).to(device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    if world > 1:
        for p in lora_params + head_params:          # identical starting point on every rank
            dist.broadcast(p.data, src=0)
    loss_config = RLCDLossConfig(rlcd_weight=args.rlcd_weight if args.rlcd else 0.0, num_samples=args.rlcd_samples)
    decision_loss, span_loss = RLCDLoss(loss_config), SpanLoss(loss_config, args.none_weight)
    groups = [{"params": lora_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    peak = [group["lr"] for group in groups]

    plans = [epoch_plan(args, epoch, lengths, world)[rank] for epoch in range(args.epochs)]
    steps_per_epoch = [math.ceil(len(plan) / args.accumulate) for plan in plans]
    total = sum(steps_per_epoch)
    step, history = 0, []
    if args.resume:
        load_checkpoint(args.resume, model, head, device)
        state = torch.load(Path(args.resume) / "trainer_state.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        if rank == 0 and (out / "history.json").exists():
            history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    if rank == 0:
        log("training_start", world=world, steps=total, start_step=step, accumulate=args.accumulate, rlcd=args.rlcd,
            kl_weight=args.kl_weight if args.replay else 0.0,
            items={t: sum(n > 0 for n in lengths[t]) for t in TASKS},
            micro_batches=dict(Counter(task for plan in plans for task, _, _ in plan)), none_shift=span_loss.none_shift)

    def dev_eval(checkpoint):
        dev = evaluate_sets(model, head, tokenizer, letters, encoder, dev_sets, args, rank, world, span_loss.none_shift)
        if rank == 0:
            entry = {"step": step, "checkpoint": checkpoint, "scores": summary_scores(dev), "dev": dev}
            history.append(entry)
            log("dev", step=step, **entry["scores"])
            write_json(out / "history.json", history)

    def checkpoint_and_eval():
        path = out / f"checkpoint-{step}"
        if rank == 0:
            save_checkpoint(path, model, head, optimizer, step, span_loss.none_shift, args.echo)
        dev_eval(str(path))

    if step == 0 and not history:
        dev_eval(None)                                   # step 0: the starting model, the retention baseline

    started, start_step = time.time(), step
    recent = torch.zeros(8, device=device)               # loss sums and counts: decision, replay, extraction, kl
    done = 0
    for epoch, plan in enumerate(plans):
        skip = min(len(plan), max(0, step - done) * args.accumulate)
        done += steps_per_epoch[epoch]
        if skip >= len(plan):
            continue
        loader = torch.utils.data.DataLoader(Batches(plan[skip:], stores, decision_items, args), batch_size=None,
                                             shuffle=False, num_workers=args.workers,
                                             prefetch_factor=4 if args.workers else None)
        micro, extraction_count = 0, 0
        remaining = len(plan) - skip
        for position, (task, batch) in enumerate(loader):
            if micro == 0:
                scale = min(1.0, (step + 1) / args.warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
                for group, value in zip(optimizer.param_groups, peak):
                    group["lr"] = value * scale
                sigma = linear_schedule(args.sigma_start, args.sigma_end, step / max(total - 1, 1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "extraction":
                    tensors = batch.to(device)
                    utilities = head(backbone_hidden(model, tensors), tensors)
                    loss = args.extraction_weight * span_loss(utilities, tensors, sigma=sigma)[0]
                else:
                    logits, targets = forward_batch(model, tokenizer, letters, batch)
                    sizes = torch.tensor([len(e["target"]) for e in batch], device=device)
                    mask = torch.arange(logits.shape[-1], device=device)[None] < sizes[:, None]
                    ordinal = torch.tensor([e["kind"] == "score" for e in batch], device=device)
                    loss = decision_loss(logits, targets, mask, ordinal, sigma=sigma).loss
                    if task == "replay" and args.kl_weight > 0:
                        with torch.no_grad(), model.disable_adapter():
                            reference, _ = forward_batch(model, tokenizer, letters, batch)
                        kl = kl_to_reference(logits, reference)
                        recent[6] += kl.detach()
                        recent[7] += 1
                        loss = loss + args.kl_weight * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {task} loss at step {step} on rank {rank}")
            loss.backward()
            slot = 2 * TASKS.index(task)
            recent[slot] += loss.detach()
            recent[slot + 1] += 1
            micro += 1
            extraction_count += task == "extraction"
            if micro < args.accumulate and position + 1 < remaining:
                continue
            counts = torch.tensor([float(micro), float(extraction_count)], device=device)
            if world > 1:
                dist.all_reduce(counts)
            sync_gradients(lora_params, head_params, int(counts[0].item()), int(counts[1].item()), world)
            norm = torch.nn.utils.clip_grad_norm_([p for p in lora_params + head_params if p.grad is not None], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            micro, extraction_count = 0, 0
            if step % 10 == 0 or step == total:
                if world > 1:
                    dist.all_reduce(recent)
                if rank == 0:
                    values = recent.tolist()
                    fields = {name: round(values[2 * i] / values[2 * i + 1], 4)
                              for i, name in enumerate(TASKS + ("kl",)) if values[2 * i + 1]}
                    elapsed = time.time() - started
                    log("train", step=step, total=total, epoch=epoch, lr=optimizer.param_groups[0]["lr"],
                        grad_norm=round(norm.item(), 3), seconds=round(elapsed),
                        eta_hours=round(elapsed / max(1, step - start_step) * (total - step) / 3600, 2), **fields)
                recent.zero_()
            if args.eval_every and step % args.eval_every == 0 and step < total:
                checkpoint_and_eval()
    checkpoint_and_eval()                               # the final step is a candidate too

    selected = None
    if rank == 0:
        selected = select(history, args.max_retention_drop)
        log("selected", step=selected["step"], checkpoint=selected["checkpoint"], **selected["scores"])
        write_json(out / "selection.json", {"selected": selected["step"], "checkpoint": selected["checkpoint"],
                                            "scores": selected["scores"], "baseline": history[0]["scores"],
                                            "max_retention_drop_points": args.max_retention_drop})
    if world > 1:
        box = [selected["checkpoint"] if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        checkpoint = box[0]
    else:
        checkpoint = selected["checkpoint"]
    load_checkpoint(checkpoint, model, head, device)
    test = evaluate_sets(model, head, tokenizer, letters, encoder, test_sets, args, rank, world, span_loss.none_shift)
    if rank == 0:
        write_json(out / "metrics_test.json", {"checkpoint": checkpoint, **test})
        log("test", checkpoint=checkpoint, seconds=round(time.time() - started))
    if distributed:
        dist.destroy_process_group()
    if rank == 0 and args.export:
        export(args, out, model, head, tokenizer, letters, {"checkpoint": checkpoint, **selected["scores"]},
               span_loss.none_shift)


if __name__ == "__main__":
    main()
