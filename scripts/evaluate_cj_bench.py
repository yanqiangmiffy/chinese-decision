"""Full CJ-Bench evaluation of an exported model directory (DecisionModel.save_pretrained output).

Every decision of every domain is scored once, candidates in their original order, no truncation and
no fitting on test labels. Probabilities use `--temperature` (default 1.0, the uncalibrated logits, so
models can be compared with each other and with the reference report); `--temperature model` uses the
model's own decision_config.json value instead. Writes `full.json` (same layout as scripts/evaluate.py:
`metrics.groups` per domain / type / source / task) and `full_predictions.jsonl.gz` to `--out`.

    CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_cj_bench.py --model runs/cj-ie-0.8b-v2/model \\
        --data /data/users/yanqiang/chinese-decision/datasets/CJ-Bench --out runs/cj-bench/v2

`--limit-per-domain N` scores the first N decisions of each domain, for smoke tests only; such a run
is marked `"complete": false` and never matches the manifest.
"""
import argparse
import gzip
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from decision.data.batching import shape_batches
from decision.data.cj import DOMAINS
from decision.metrics import BenchmarkMetrics
from decision.modeling import DecisionModel
from decision.processing import PROMPT_VERSION, DecisionProcessor
from decision.utils import log, sha256, write_json
from scripts.evaluate import encode_pool, read_records


def pools(records, size):
    pool = []
    for record in records:
        pool.append(record)
        if len(pool) >= size:
            yield pool
            pool = []
    if pool:
        yield pool


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="exported model directory")
    parser.add_argument("--data", default="datasets/CJ-Bench")
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", default="1.0", help="a number, or 'model' for decision_config.json's value")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--limit-per-domain", type=int, default=0)
    args = parser.parse_args()
    out, data = Path(args.out), Path(args.data)
    if (out / "full.json").exists():
        raise FileExistsError(f"{out}/full.json exists; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    processor = DecisionProcessor.from_pretrained(args.model)
    model = DecisionModel.from_pretrained(args.model).eval()
    letters = model.letter_tensor(processor)
    temperature = model.config.temperature if args.temperature == "model" else float(args.temperature)
    expected = sum(args.limit_per_domain or entry["decisions"] for entry in manifest["domains"].values())
    metadata = {"dataset": "bbldCVer-hf/CJ-Bench", "manifest_sha256": sha256(data / "MANIFEST.json"),
                "model": str(args.model), "model_name": model.config.model_name, "prompt_version": PROMPT_VERSION,
                "candidate_order": "original; no permutation", "temperature": temperature,
                "calibration": "no fitting on test labels", "max_length": processor.max_length,
                "length_policy": "fail on overflow; no truncation", "ece_bins": 15,
                "accuracy_scope": "hard targets only; soft targets included in NLL/Brier",
                "gpu": torch.cuda.get_device_name(0), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "dtype": "bfloat16", "arguments": vars(args), "complete": not args.limit_per_domain}
    log("evaluation_start", expected=expected, **{k: metadata[k] for k in ("model", "temperature", "gpu")})
    metrics, counts, types, hard = BenchmarkMetrics(bins=15), Counter(), defaultdict(Counter), Counter()
    started = last = time.monotonic()
    done, path = 0, out / "full_predictions.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as predictions:
        for domain in DOMAINS:
            records = read_records(data, domain)
            if args.limit_per_domain:
                records = (r for _, r in zip(range(args.limit_per_domain), records))
            for pool in pools(records, args.pool_size):
                rows = encode_pool(pool, processor.tokenizer, processor.max_length)
                for batch in shape_batches(sorted(rows, key=lambda row: len(row["input_ids"])),
                                           args.batch_size, args.token_budget):
                    with torch.inference_mode():
                        logits, _ = model.score_decisions(batch, processor, letters)
                        if not torch.isfinite(logits).all().item():
                            raise FloatingPointError("non-finite benchmark logits")
                        values = (logits / temperature).log_softmax(-1).cpu().tolist()
                    for row, logp in zip(batch, values):
                        logp = logp[:len(row["target"])]
                        norm = math.log(sum(math.exp(x) for x in logp))
                        logp = [x - norm for x in logp]
                        metrics.add(row, logp)
                        result = {k: v for k, v in row.items() if k != "input_ids"}
                        result["log_probabilities"], result["probabilities"] = logp, [math.exp(x) for x in logp]
                        predictions.write(json.dumps(result, ensure_ascii=False) + "\n")
                        counts[row["domain"]] += 1
                        types[row["domain"]][row["kind"]] += 1
                        hard[row["domain"]] += int(row["hard"])
                        done += 1
                    if time.monotonic() - last >= 60:
                        elapsed = time.monotonic() - started
                        progress = {"evaluated": done, "expected": expected, "percent": round(100 * done / expected, 2),
                                    "eta_minutes": round((expected - done) / (done / elapsed) / 60, 1)}
                        write_json(out / "progress.json", progress)
                        log("evaluation_progress", **progress)
                        last = time.monotonic()
    if done != expected:
        raise ValueError(f"coverage mismatch: {done} != {expected}")
    if not args.limit_per_domain:
        for domain, entry in manifest["domains"].items():
            if counts[domain] != entry["decisions"] or dict(types[domain]) != entry["question_types"] \
                    or hard[domain] != entry["targets"]["hard"]:
                raise ValueError(f"full coverage/type/hard-target mismatch in {domain}")
    report = metadata | {"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "metrics": metrics.result(),
                         "coverage": {"evaluated": done, "expected": expected, "by_domain": dict(counts),
                                      "hard_by_domain": dict(hard)},
                         "seconds": time.monotonic() - started, "predictions": str(path),
                         "predictions_sha256": sha256(path)}
    write_json(out / "full.json", report)
    groups = report["metrics"]["groups"]
    log("evaluation_completed", all=groups["all"], domains={d: groups[f"domain:{d}"]["accuracy"] for d in DOMAINS},
        seconds=report["seconds"])


if __name__ == "__main__":
    main()
