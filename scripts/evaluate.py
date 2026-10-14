"""One-GPU CJ-Bench quick/full evaluation of a frozen adapter snapshot (see snapshot_best.py).

Test labels are used only to score predictions. No checkpoint selection,
threshold search or temperature fitting is performed on the benchmark.
"""
import argparse
import gzip
import hashlib
import heapq
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import decision.metrics
from decision.data import DOMAINS, shape_batches
from decision.metrics import BenchmarkMetrics
from decision.schema import PROMPT_VERSION, examples_for, group_key
from decision.utils import log, sha256, write_json


def deployment_metadata(merged, selection, config):
    merged = Path(merged)
    decision_config = json.loads((merged / "decision_config.json").read_text())
    merged_summary = json.loads((merged / "training_summary.json").read_text())
    if merged_summary["selected_checkpoint"] != selection:
        raise ValueError("merged model and adapter snapshot selected different checkpoints")
    if decision_config["prompt_version"] != PROMPT_VERSION or decision_config["max_length"] != config["max_length"]:
        raise ValueError("merged model prompt/input configuration mismatch")
    temperature = float(decision_config["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid existing deployment temperature")
    weights = sorted(merged.glob("*.safetensors"))
    if not weights:
        raise ValueError("merged model has no safetensors weights")
    return temperature, {p.name:sha256(p) for p in weights}


def read_records(data, domain):
    with gzip.open(data / domain / "test.cases.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            meta = case.get("_meta") or {}
            for example in examples_for(case, domain):
                yield {"example": example, "parent_id": group_key(case),
                       "task": str(meta.get("task_id") or "unknown"), "line": line_number}


def select_quick(data, manifest, limit, seed):
    """Keep the smallest seeded ID hashes in each domain/type, across all rows."""
    heaps, counts = defaultdict(list), Counter()
    serial = 0
    for domain in DOMAINS:
        types = Counter()
        for record in read_records(data, domain):
            example = record["example"]
            serial += 1
            counts[domain] += 1
            types[example.kind] += 1
            rank = int(hashlib.sha256(f"{seed}:{domain}:{example.case_id}".encode()).hexdigest(), 16)
            entry = (-rank, -serial, record)
            heap = heaps[(domain, example.kind)]
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
        expected = manifest["domains"][domain]
        if counts[domain] != expected["decisions"] or dict(types) != expected["question_types"]:
            raise ValueError(f"manifest/count mismatch in {domain}: {counts[domain]}, {dict(types)}")
        log("sample_scan", domain=domain, scanned=counts[domain], types=dict(types))
    records = [entry[2] for key in sorted(heaps) for entry in sorted(heaps[key], reverse=True)]
    return records, {f"{d}:{k}": len(heap) for (d, k), heap in sorted(heaps.items())}


def encode_pool(records, tokenizer, max_length):
    prompts, rows = [], []
    for record in records:
        example = record["example"]
        prompt, target = example.render(tokenizer)  # Preserve candidate order; no test-time shuffle.
        prompts.append(prompt)
        rows.append({"case_id": example.case_id, "domain": example.domain, "kind": example.kind,
                     "source": example.source, "target": target, "hard": example.hard,
                     "option_keys": [key for key, _ in example.options],
                     "parent_id": record["parent_id"], "task": record["task"], "line": record["line"]})
    encoded = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    for row, ids in zip(rows, encoded):
        if len(ids) > max_length:
            raise ValueError(f"overlength benchmark case {row['case_id']}: {len(ids)} > {max_length}; no truncation")
        row["input_ids"] = ids
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-run", required=True, help="directory created by scripts/snapshot_best.py")
    parser.add_argument("--model", help="merged deployment model with its validation-fitted decision_config.json")
    parser.add_argument("--data", default="datasets/CJ-Bench")
    parser.add_argument("--sample-per-stratum", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=65536)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--latency-per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-only", action="store_true")
    args = parser.parse_args()
    if min(args.sample_per_stratum, args.batch_size, args.token_budget, args.pool_size) <= 0:
        raise ValueError("batch, pool, token and sample limits must be positive")
    run, data = Path(args.snapshot_run), Path(args.data)
    if (run / "full.json").exists() or (run / "quick.json").exists():
        raise ValueError("use a new snapshot run; existing evaluation reports are not overwritten")
    config = json.loads((run / "training_config.json").read_text())
    selection = json.loads((run / "adapter/selection.json").read_text())
    if config["prompt_version"] != PROMPT_VERSION:
        raise ValueError("snapshot prompt/compiler mismatch")
    manifest = json.loads((data / "MANIFEST.json").read_text())
    verified = []
    for domain, entry in manifest["domains"].items():
        path = data / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"benchmark file hash/size mismatch: {path}")
        verified.append({"domain": domain, "sha256": entry["sha256"], "bytes": entry["bytes"]})
    import torch
    from decision.model import forward_batch, letter_tokens, load_model, load_tokenizer
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() != 1:
        raise ValueError("evaluation requires exactly one visible GPU")
    adapter_digest = sha256(run / "adapter/adapter_model.safetensors")
    setup = json.loads((run / "setup.json").read_text())
    if adapter_digest != setup["adapter_sha256"]:
        raise ValueError("frozen adapter hash changed")
    temperature = 1.0
    merged_files = None
    if args.model:
        temperature, merged_files = deployment_metadata(args.model, selection, config)
    metadata = {"dataset": "bbldCVer-hf/CJ-Bench", "dataset_source_revision": manifest["source_revision"],
                "manifest_sha256": sha256(data / "MANIFEST.json"), "verified_files": verified,
                "selected_checkpoint": selection, "adapter_sha256": adapter_digest,
                "base": config["base"], "prompt_version": PROMPT_VERSION,
                "candidate_order": "original; no permutation", "temperature": temperature,
                "calibration": ("existing deployment temperature fitted on held-out validation; no test fitting"
                                if args.model else "uncalibrated snapshot; no fitting on test labels"),
                "merged_model": args.model, "merged_weight_sha256": merged_files,
                "max_length": config["max_length"], "length_policy": "fail on overflow; no truncation",
                "ece_bins": 15, "accuracy_scope": "hard targets only; soft targets included in NLL/Brier",
                "gpu": torch.cuda.get_device_name(0), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "world_size": 1, "dtype": "bfloat16", "arguments": vars(args),
                "evaluator_sha256": sha256(Path(__file__)),
                "metric_code_sha256": sha256(decision.metrics.__file__)}
    write_json(run / "evaluation_config.json", metadata)
    write_json(run / "progress.json", {"status": "preparing", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    log("evaluation_start", **metadata)
    records, strata = select_quick(data, manifest, args.sample_per_stratum, args.seed)
    ids = [{"domain": r["example"].domain, "case_id": r["example"].case_id,
            "kind": r["example"].kind, "line": r["line"], "parent_id": r["parent_id"]} for r in records]
    write_json(run / "quick_selected_ids.json", ids)
    sample_hash = sha256(run / "quick_selected_ids.json")
    tokenizer = load_tokenizer(args.model or config["base"])
    quick_rows = encode_pool(records, tokenizer, config["max_length"])
    torch.save(quick_rows, run / "quick_tokens.pt")
    model = (load_model(args.model) if args.model else load_model(config["base"], str(run / "adapter"))).eval()
    letters = letter_tokens(tokenizer, "cuda")
    with torch.inference_mode():
        forward_batch(model, tokenizer, letters, quick_rows[:2])
    torch.cuda.synchronize()
    log("model_ready", sample_n=len(quick_rows), sample_strata=strata,
        gpu_memory_gb=round(torch.cuda.memory_allocated() / 1e9, 3))

    def run_evaluation(phase, pools, expected_total, sampling):
        metrics = BenchmarkMetrics(bins=15)
        counts, types, hard, parents = Counter(), defaultdict(Counter), Counter(), defaultdict(set)
        lengths, tokens, padded_tokens, batch_count = [], 0, 0, 0
        started = last_log = time.monotonic()
        n = 0
        output_path = run / f"{phase}_predictions.jsonl.gz"
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=1) as predictions:
            for pool in pools:
                for batch in shape_batches(sorted(pool, key=lambda row: len(row["input_ids"])),
                                           args.batch_size, args.token_budget):
                    with torch.inference_mode():
                        logits, _ = forward_batch(model, tokenizer, letters, batch)
                        if not torch.isfinite(logits).all().item():
                            raise FloatingPointError("non-finite benchmark logits")
                        values = (logits / temperature).log_softmax(-1).cpu().tolist()
                    for row, logp in zip(batch, values):
                        logp = logp[:len(row["target"])]
                        norm = math.log(sum(math.exp(x) for x in logp))
                        logp = [x - norm for x in logp]
                        metrics.add(row, logp)
                        result = {k:v for k,v in row.items() if k != "input_ids"}
                        result["log_probabilities"] = logp
                        result["probabilities"] = [math.exp(x) for x in logp]
                        predictions.write(json.dumps(result, ensure_ascii=False) + "\n")
                        domain = row["domain"]
                        counts[domain] += 1
                        types[domain][row["kind"]] += 1
                        hard[domain] += int(row["hard"])
                        parents[domain].add(row["parent_id"])
                        length = len(row["input_ids"])
                        lengths.append(length)
                        tokens += length
                        n += 1
                    padded_tokens += len(batch) * (((max(len(r["input_ids"]) for r in batch) + 63) // 64) * 64)
                    batch_count += 1
                    now = time.monotonic()
                    if now - last_log >= 30 or n == expected_total:
                        elapsed = now - started
                        progress = {"status": "running", "phase": phase, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "evaluated": n, "expected": expected_total, "percent": 100 * n / expected_total,
                                    "seconds": elapsed, "decisions_per_second": n / elapsed,
                                    "estimated_remaining_minutes": (expected_total - n) / (n / elapsed) / 60,
                                    "by_domain": dict(counts), "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
                        predictions.flush()
                        write_json(run / "progress.json", progress)
                        log("evaluation_progress", **progress)
                        last_log = now
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if n != expected_total:
            raise ValueError(f"{phase} coverage mismatch: {n} != {expected_total}")
        if phase == "full":
            for domain, entry in manifest["domains"].items():
                if counts[domain] != entry["decisions"] or dict(types[domain]) != entry["question_types"]:
                    raise ValueError(f"full coverage/type mismatch: {domain}")
                if hard[domain] != entry["targets"]["hard"]:
                    raise ValueError(f"hard-target coverage mismatch: {domain}")
        report = metadata | {"phase": phase, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "status": "completed", "sampling": sampling, "metrics": metrics.result(),
                            "coverage": {"evaluated": n, "expected": expected_total, "filtered": 0,
                                         "truncated": 0, "by_domain": dict(counts),
                                         "by_domain_type": {d:dict(v) for d,v in types.items()},
                                         "hard_by_domain": dict(hard), "unique_parent_units": {d:len(v) for d,v in parents.items()}},
                            "timing": {"seconds": elapsed, "decisions_per_second": n / elapsed,
                                       "input_tokens": tokens, "padded_tokens": padded_tokens,
                                       "max_input_tokens": max(lengths), "mean_input_tokens": statistics.mean(lengths),
                                       "batches": batch_count, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                                       "note": "batched throughput, including CPU processing; not official H200 batch-one latency"},
                            "predictions": str(output_path), "predictions_sha256": sha256(output_path)}
        write_json(run / f"{phase}.json", report)
        log("evaluation_completed", phase=phase, all=report["metrics"]["groups"]["all"],
            macro_domain_accuracy=report["metrics"]["macro_domain_accuracy"], seconds=elapsed)
        return report

    quick = run_evaluation("quick", [quick_rows], len(quick_rows),
                           {"method": "smallest seeded SHA256 ID hashes per domain/type; full-file scan",
                            "seed": args.seed, "per_stratum": args.sample_per_stratum,
                            "strata": strata, "selected_ids_sha256": sample_hash,
                            "note": "balanced subset; overall micro accuracy does not use full benchmark domain weights"})
    if args.latency_per_stratum:
        chosen, used = [], Counter()
        for record in records:
            example = record["example"]
            key = (example.domain, example.kind)
            if used[key] < args.latency_per_stratum:
                chosen.append(record)
                used[key] += 1
        elapsed_samples = []
        for record in chosen:
            # Warm this input shape once; measure input compilation through final probabilities.
            row = encode_pool([record], tokenizer, config["max_length"])
            with torch.inference_mode():
                forward_batch(model, tokenizer, letters, row)
            torch.cuda.synchronize()
            started = time.perf_counter()
            row = encode_pool([record], tokenizer, config["max_length"])
            with torch.inference_mode():
                logits, _ = forward_batch(model, tokenizer, letters, row)
                probabilities = (logits[0, :len(row[0]["target"])] / temperature).softmax(-1).cpu().tolist()
            answer = dict(zip(row[0]["option_keys"], probabilities))
            if row[0]["kind"] == "score":
                answer["expected_score"] = sum(i * p for i,p in enumerate(probabilities))
            elapsed_samples.append(1000 * (time.perf_counter() - started))
        latency = {"batch_size": 1, "n": len(chosen), "median_ms": statistics.median(elapsed_samples),
                   "gpu": metadata["gpu"], "physical_gpu": metadata["physical_gpu"],
                   "scope": "warmed local input preparation + model forward + candidate probabilities; excludes HTTP/network",
                   "note": f"measured on {metadata['gpu']}; not hardware-equivalent to H200 latencies in the Chinese-Jev paper"}
        write_json(run / "latency.json", latency)
        log("latency_completed", **latency)

    if not args.quick_only:
        def full_pools():
            for domain in DOMAINS:
                pool = []
                for record in read_records(data, domain):
                    pool.append(record)
                    if len(pool) >= args.pool_size:
                        yield encode_pool(pool, tokenizer, config["max_length"])
                        pool = []
                if pool:
                    yield encode_pool(pool, tokenizer, config["max_length"])
        run_evaluation("full", full_pools(), manifest["total_decisions"], {"method": "all released test decisions"})
    if sha256(run / "adapter/adapter_model.safetensors") != adapter_digest:
        raise ValueError("adapter snapshot changed during evaluation")
    write_json(run / "progress.json", {"status": "completed", "phase": "quick" if args.quick_only else "full",
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                        "selected_checkpoint": selection})
    log("CJ_BENCH_COMPLETED", run=str(run), checkpoint=selection)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--snapshot-run" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--snapshot-run") + 1])
            if directory.exists():
                write_json(directory / "progress.json", {"status": "failed", "error": str(exc),
                                                         "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        log("evaluation_failed", error=str(exc))
        traceback.print_exc()
        sys.exit(1)
