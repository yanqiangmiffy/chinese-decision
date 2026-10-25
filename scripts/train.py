"""LoRA decision training on the prepared CJ-Dataset; each decision is seen once per epoch.

Launch with `torchrun --nproc_per_node=N` for N GPUs, or plain `python` for one GPU.
The defaults reproduce the released run; `--resume` continues `--out` exactly.
"""
import argparse
import contextlib
import itertools
import json
import math
import os
import random
import shutil
import socket
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from decision.data.batching import token_batches
from decision.data.cj import DOMAINS, FULL_TRAIN_DECISIONS, prepared_examples, validation_items
from decision.modeling import DecisionModel
from decision.processing import MAX_LENGTH, PROMPT_VERSION, DecisionProcessor
from decision.training.common import evaluate_decisions, replace_directory, with_last
from decision.losses import RLCDLoss, RLCDLossConfig, linear_schedule
from decision.utils import log

# Stored in config.json. --resume compares every config key, so renaming or adding
# arguments (or changing this tag) makes existing runs impossible to resume.
TRAINING_ALGORITHM = "ddp-record-mean-v2-shapes64"
RESUME_IGNORED = {"resume", "log_every", "world_size", "keep_every"}
# Recorded only when --rlcd is on, so cross-entropy-only runs keep their original config.
RLCD_ARGS = ("rlcd", "rlcd_group_size", "rlcd_sigma_start", "rlcd_sigma_end")
VALIDATION_BATCH_SIZE, VALIDATION_TOKEN_BUDGET = 32, 16384


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="models/Qwen3.5-0.8B")
    parser.add_argument("--data", default="datasets/CJ-Dataset")
    parser.add_argument("--prepared-data", default="datasets/CJ-mixed")
    parser.add_argument("--out", default="runs/cj-ddp")
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--token-budget", type=int, default=262144, help="global padded-token budget, shared across GPUs")
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--debug-examples", type=int, default=0)
    parser.add_argument("--val-per-stratum", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--keep-every", type=int, default=1000,
                        help="also keep a frozen copy of last/ as step-N/ every N steps (0 = off)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    rlcd = parser.add_argument_group("RLCD", "optional policy-gradient term over a proper-scoring-rule reward "
                                             "(Chinese-Jev); off by default")
    rlcd.add_argument("--rlcd", action="store_true", help="add the RLCD term to the distribution cross-entropy")
    rlcd.add_argument("--rlcd-group-size", type=int, default=4, help="perturbations per decision")
    rlcd.add_argument("--rlcd-sigma-start", type=float, default=0.4, help="logit noise scale at the start of training")
    rlcd.add_argument("--rlcd-sigma-end", type=float, default=0.1, help="noise scale at the end, annealed linearly")
    args = parser.parse_args(argv)
    if args.rlcd and (args.rlcd_group_size < 2 or min(args.rlcd_sigma_start, args.rlcd_sigma_end) <= 0):
        parser.error("--rlcd needs --rlcd-group-size >= 2 and positive sigmas")
    return args


def run_config(args, world_size, data_revision):
    config = vars(args) | {"world_size": world_size, "domains": list(DOMAINS), "prompt_version": PROMPT_VERSION,
                           "batch_size": args.global_batch_size, "training_algorithm": TRAINING_ALGORITHM,
                           "data_revision": data_revision}
    if args.rlcd:
        config["training_algorithm"] += "+rlcd"
    else:
        for key in RLCD_ARGS:
            del config[key]
    return config


def resume_conflicts(old_config, config):
    """Config keys whose values differ between a saved run and the current arguments."""
    return sorted(k for k in old_config.keys() | config.keys()
                  if k not in RESUME_IGNORED and old_config.get(k) != config.get(k))


def init_distributed():
    """Read torchrun's environment, or describe a one-process group for plain `python`."""
    if "RANK" not in os.environ:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])


def main():
    args = parse_args()
    rank, local_rank, world = init_distributed()
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(6)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20), device_id=device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    out = Path(args.out)
    manifest = json.loads((Path(args.prepared_data) / "manifest.json").read_text())
    if manifest["validation_included"] or manifest["decisions"] != FULL_TRAIN_DECISIONS:
        raise ValueError("unexpected prepared corpus")
    config = run_config(args, world, manifest["source_revision"])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        cfg = out / "config.json"
        if cfg.exists() and not args.resume:
            raise ValueError("run exists; use --resume or another output directory")
        if args.resume:
            conflicts = resume_conflicts(json.loads(cfg.read_text()), config)
            if conflicts:
                raise ValueError(f"resume requires the original distributed configuration; differs in {conflicts}")
        else:
            cfg.write_text(json.dumps(config, indent=2))
    dist.barrier()
    processor = DecisionProcessor.from_pretrained(args.base)
    processor.config.max_length = args.max_length
    tokenizer = processor.tokenizer
    letters = torch.tensor(processor.letter_ids, device=device)
    val_path = out / "validation_tokens.pt"
    if rank == 0 and not val_path.exists():
        val, audit = validation_items(args.data, DOMAINS, processor, args.val_per_stratum, seed=args.seed)
        torch.save(val, val_path)
        (out / "validation_sampling.json").write_text(json.dumps(audit, indent=2))
    dist.barrier()
    val = torch.load(val_path, weights_only=False)
    decision_model = DecisionModel.from_pretrained(args.base, adapter=out / "last" if args.resume else None,
                                                   trainable=args.resume, device=device)
    if rank == 0 and not args.resume:
        metrics, _ = evaluate_decisions(decision_model, processor, val, VALIDATION_BATCH_SIZE, VALIDATION_TOKEN_BUDGET)
        (out / "baseline.json").write_text(json.dumps(metrics, indent=2))
        log("baseline", metrics=metrics["all"], validation_n=len(val))
    dist.barrier()
    if not args.resume:
        decision_model.enable_lora(r=args.lora_r)
    core = decision_model.backbone
    core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    core.enable_input_require_grads()
    model = DDP(core, device_ids=[local_rank], output_device=local_rank, gradient_as_bucket_view=True,
                broadcast_buffers=False, bucket_cap_mb=25)
    # from_pretrained starts in eval mode; every rank must enable checkpointing's
    # training path, including ranks which did not run the baseline evaluation.
    model.train()
    optimizer = torch.optim.AdamW([x for x in core.parameters() if x.requires_grad], lr=args.lr)
    step, examples_seen, start_epoch, skip_batches, best, stats = 0, 0, 0, 0, float("inf"), Counter()
    resume_checkpoint_pending = args.resume
    if args.resume:
        state = torch.load(out / "last/training_state.pt", weights_only=False, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        step, examples_seen, start_epoch, skip_batches, best = (state[k] for k in ("step", "examples_seen", "epoch", "batches_done", "best"))
        stats.update(state["stats"])
        previous_world = len(state["random_states"])
        if state.get("world_size", previous_world) != previous_world:
            raise ValueError("checkpoint worker count does not match its random states")
        if rank < previous_world:
            torch.set_rng_state(state["random_states"][rank]["torch"])
            torch.cuda.set_rng_state(state["random_states"][rank]["cuda"], device)
        else:
            # Extra workers have no old dropout stream. Initialize a reproducible
            # stream without changing the seed used by global data batching.
            torch.manual_seed(args.seed + rank + step * 1000003)
        if rank == 0:
            if previous_world != world:
                migration = {"step": step, "examples_seen": examples_seen, "batches_done": skip_batches,
                             "previous_world_size": previous_world, "world_size": world,
                             "changed_training_parameters": [], "new_rng_ranks": list(range(previous_world, world))}
                (out / "world_size_migration.json").write_text(json.dumps(migration, indent=2))
                log("world_size_migration", **migration)
            (out / "config.json").write_text(json.dumps(config, indent=2))
            log("resume", step=step, examples_seen=examples_seen, skip_batches=skip_batches,
                previous_world_size=previous_world, world_size=world)
        del state
    dist.barrier()
    total = args.debug_examples or manifest["decisions"]
    if rank == 0:
        log("training_start", nominal_decisions=total, world_size=world, global_batch_size=args.global_batch_size,
            global_token_budget=args.token_budget, config=config)
    started, recent_sum, recent_n, accumulated = time.time(), 0., 0, 0
    recent_rlcd = [0., 0., 0.] if args.rlcd else []  # cross-entropy, RLCD term, reward; summed over records
    # The Chinese-Jev preset is the term earlier --rlcd runs trained with, so they resume unchanged.
    rlcd = RLCDLoss(RLCDLossConfig.chinese_jev(num_samples=args.rlcd_group_size)) if args.rlcd else None
    last_stream_stats = Counter()

    def checkpoint(epoch, batches_done, keep=False):
        random_states = [None] * world
        dist.all_gather_object(random_states, {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)})
        if rank == 0:
            temporary = out / "last.next"
            temporary.mkdir(exist_ok=True)
            core.save_pretrained(temporary)
            tokenizer.save_pretrained(temporary)
            torch.save({"optimizer": optimizer.state_dict(), "step": step, "examples_seen": examples_seen,
                        "world_size": world,
                        "epoch": epoch, "batches_done": batches_done, "best": best,
                        "stats": dict(stats), "random_states": random_states}, temporary / "training_state.pt")
            replace_directory(temporary, out / "last")
            if keep:
                # last/ and best/ are replaced in place; step-N/ stays for later evaluation.
                kept = out / f"step-{step}.next"
                shutil.copytree(out / "last", kept, dirs_exist_ok=True)
                replace_directory(kept, out / f"step-{step}")
                log("checkpoint_kept", step=step, path=str(out / f"step-{step}"))
            (out / "progress.json").write_text(json.dumps({"status": "running", "step": step, "examples_seen": examples_seen,
                                                           "world_size": world, "global_batch_size": args.global_batch_size,
                                                           "stats": dict(stats), "reader_stats": dict(last_stream_stats)}, indent=2))
            log("checkpoint", step=step, path=str(out / "last"))
        dist.barrier()

    def validation():
        nonlocal best
        if rank == 0:
            metrics, _ = evaluate_decisions(decision_model, processor, val, VALIDATION_BATCH_SIZE,
                                            VALIDATION_TOKEN_BUDGET)
            log("validation", step=step, metrics=metrics["all"])
            (out / "latest_metrics.json").write_text(json.dumps(metrics, indent=2))
            score = metrics["all"]["distribution_nll"]
            if score < best:
                best = score
                temporary = out / "best.next"
                temporary.mkdir(exist_ok=True)
                core.save_pretrained(temporary)
                tokenizer.save_pretrained(temporary)
                (temporary / "metrics.json").write_text(json.dumps(metrics, indent=2))
                (temporary / "selection.json").write_text(json.dumps({"step": step, "metric": "distribution_nll", "value": best}))
                replace_directory(temporary, out / "best")
        best_tensor = torch.tensor(best, dtype=torch.float64, device=device)
        dist.broadcast(best_tensor, 0)
        best = float(best_tensor)
        dist.barrier()

    done = bool(args.max_steps and step >= args.max_steps)
    for epoch in range(start_epoch, args.epochs):
        if done:
            break
        last_stream_stats = Counter()
        stream = prepared_examples(args.prepared_data, DOMAINS, args.seed + epoch, last_stream_stats)
        if args.debug_examples:
            stream = itertools.islice(stream, args.debug_examples)
        batches = token_batches(stream, processor, args.global_batch_size, args.token_budget, args.seed + epoch,
                                last_stream_stats, pool_size=args.pool_size)
        optimizer.zero_grad(set_to_none=True)
        batches_done = 0
        for batch_index, (global_batch, last_batch) in enumerate(with_last(batches)):
            batches_done = batch_index + 1
            if epoch == start_epoch and batch_index < skip_batches:
                continue
            # 100-step linear warmup, then cosine decay to 10% of the peak rate.
            fraction = step / args.max_steps if args.max_steps else examples_seen / (total * args.epochs)
            scale = min(1., (step + 1) / 100) * (.1 + .45 * (1 + math.cos(math.pi * min(fraction, 1.))))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            local_batch = global_batch[rank::world]
            real_count = len(local_batch)
            if not local_batch:
                local_batch = global_batch[:1]
            will_step = accumulated + len(global_batch) >= args.global_batch_size or last_batch
            sync = contextlib.nullcontext() if will_step else model.no_sync()
            with sync:
                logits, targets = decision_model.score_decisions(local_batch, processor, letters, forward=model)
                if args.rlcd:
                    sigma = linear_schedule(args.rlcd_sigma_start, args.rlcd_sigma_end, fraction)
                    sizes = torch.tensor([len(e["target"]) for e in local_batch], device=device)
                    mask = torch.arange(logits.shape[-1], device=device)[None, :] < sizes[:, None]
                    ordinal = torch.tensor([e["kind"] == "score" for e in local_batch], device=device)
                    out = rlcd(logits, targets, mask, ordinal, sigma=sigma)
                    for i, value in enumerate((out.ce, out.rlcd, out.reward)):
                        recent_rlcd[i] += value.item() * real_count
                    loss = out.loss
                else:
                    # Hard labels are one-hot special cases of the same distribution cross-entropy.
                    loss = -(targets * logits.log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite distributed loss")
                # Weight by real records so a long single-record micro-batch is not over-weighted.
                (loss * real_count * world / args.global_batch_size).backward()
            recent_sum += loss.item() * real_count
            recent_n += real_count
            accumulated += len(global_batch)
            examples_seen += len(global_batch)
            for example in global_batch:
                stats["trained:" + example["domain"] + ":" + example["kind"]] += 1
            if not will_step:
                continue
            for parameter in core.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(args.global_batch_size / accumulated)
            grad_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite distributed gradients")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % args.log_every == 0 or resume_checkpoint_pending:
                sums = torch.tensor([recent_sum, recent_n, *recent_rlcd], dtype=torch.float64, device=device)
                dist.all_reduce(sums)
                peaks = [None] * world
                dist.all_gather_object(peaks, round(torch.cuda.max_memory_allocated(device) / 1e9, 3))
                if rank == 0:
                    parts = {}
                    if args.rlcd:
                        parts = {"ce": float(sums[2] / sums[1]), "rlcd": float(sums[3] / sums[1]),
                                 "reward": float(sums[4] / sums[1]), "sigma": sigma}
                    log("step", step=step, loss=float(sums[0] / sums[1]), examples_seen=examples_seen,
                        seconds=round(time.time() - started, 2), batch_size=accumulated,
                        micro_batch_size=len(global_batch), world_size=world,
                        gpu_peak_gb=peaks, lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm), **parts)
                recent_sum, recent_n = 0., 0
                recent_rlcd = [0.] * len(recent_rlcd)
            accumulated = 0
            terminal = bool(args.max_steps and step >= args.max_steps)
            if step % args.eval_every == 0 or terminal:
                validation()
            keep = bool(args.keep_every and step % args.keep_every == 0)
            if step == 1 or step % args.save_every == 0 or keep or terminal or resume_checkpoint_pending:
                checkpoint(epoch, batches_done, keep)
                resume_checkpoint_pending = False
            if terminal:
                done = True
                break
        skip_batches = 0
        if done:
            break
        checkpoint(epoch + 1, 0)
    validation()
    if rank == 0:
        core.save_pretrained(out / "final")
        tokenizer.save_pretrained(out / "final")
        (out / "progress.json").write_text(json.dumps({"status": "completed", "step": step, "examples_seen": examples_seen,
                                                       "stats": dict(stats), "reader_stats": dict(last_stream_stats)}, indent=2))
        log("completed", step=step, examples_seen=examples_seen)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
