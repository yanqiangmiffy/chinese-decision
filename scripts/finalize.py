"""Calibrate the validation-selected adapter, merge it, and verify the exported model.

The temperature is fitted on validation partition 0, which training never uses for
model selection; the merged model must reproduce the adapter's decisions.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from decision.data.cj import validation_items
from decision.metrics import summarize
from decision.modeling import DecisionModel
from decision.processing import LETTERS, DecisionProcessor
from decision.training.common import evaluate_decisions, fit_temperature
from decision.utils import log, model_name_for


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="runs/cj-ddp")
    parser.add_argument("--out", default="models/chinese-decision-0.8b")
    parser.add_argument("--calibration-per-stratum", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32, help="single-GPU calibration/validation batch limit")
    parser.add_argument("--token-budget", type=int, default=16384, help="single-GPU evaluation token budget")
    parser.add_argument("--base-model-id", help="original model repository recorded in the export metadata")
    args = parser.parse_args(argv)
    if min(args.calibration_per_stratum, args.batch_size, args.token_budget) <= 0:
        parser.error("calibration, batch and token limits must be positive")
    return args


def main():
    args = parse_args()
    run, out = Path(args.run), Path(args.out)
    if out.exists():
        raise FileExistsError(f"export destination already exists; use a new output directory: {out}")
    config = json.loads((run / "config.json").read_text())
    model_name = model_name_for(config["base"])
    base_name = Path(config["base"]).name
    base_model_id = args.base_model_id or (f"Qwen/{base_name}" if base_name.startswith("Qwen3.5-") else config["base"])
    log("export_stage", stage="loading", run=str(run), model_name=model_name)
    processor = DecisionProcessor.from_pretrained(config["base"])
    processor.config.max_length = config["max_length"]
    model = DecisionModel.from_pretrained(config["base"], adapter=run / "best")
    letters = model.letter_tensor(processor)
    validation = torch.load(run / "validation_tokens.pt", weights_only=False)
    model.eval()
    probe = validation[:2]
    with torch.inference_mode():
        adapter_logits, _ = model.score_decisions(probe, processor, letters)
    model = model.merge_adapter(safe_merge=True).eval()
    with torch.inference_mode():
        merged_logits, _ = model.score_decisions(probe, processor, letters)
    delta = float((adapter_logits - merged_logits).abs().max())
    if not torch.isfinite(merged_logits).all() or not torch.equal(adapter_logits.argmax(-1), merged_logits.argmax(-1)):
        raise ValueError("merged model changed probe decisions")
    log("export_stage", stage="saving_merged_weights", out=str(out))
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(out, processor)
    shutil.copyfile(Path(config["base"]) / "LICENSE", out / "LICENSE")
    (out / "training_config.json").write_text(json.dumps(config, indent=2))
    summary = {"source_run": str(run), "base_model": base_model_id, "model_name": model_name,
               "dataset": "bbldCVer-hf/CJ-Dataset", "dataset_revision": config.get("data_revision"),
               "method_reference": "n4ze3m/hmm", "prompt_version": config["prompt_version"],
               "selected_checkpoint": json.loads((run / "best/selection.json").read_text()),
               "training_progress": json.loads((run / "progress.json").read_text())}
    summary["training_algorithm"] = config.get("training_algorithm")
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    # Evaluate the artifact reloaded from disk; calibrate its actual BF16 logits.
    del model
    torch.cuda.empty_cache()
    model = DecisionModel.from_pretrained(out).eval()
    with torch.inference_mode():
        reloaded_logits, _ = model.score_decisions(probe, processor, letters)
    reload_delta = float((merged_logits - reloaded_logits).abs().max())
    if reload_delta > 1e-5:
        raise ValueError(f"save/reload logit mismatch: {reload_delta}")
    log("export_stage", stage="calibration", batch_size=args.batch_size, token_budget=args.token_budget)
    calibration, audit = validation_items(config["data"], config["domains"], processor,
                                           args.calibration_per_stratum, subset=(0,), seed=config["seed"])
    _, rows = evaluate_decisions(model, processor, calibration, args.batch_size, args.token_budget)
    temperature, calibration_hard_n = fit_temperature(rows)
    raw_metrics, val_rows = evaluate_decisions(model, processor, validation, args.batch_size, args.token_budget)
    report = {"temperature": temperature, "calibration_hard_n": calibration_hard_n, "calibration_partition": [0],
              "evaluation_partition": [1, 2, 3, 4], "before": raw_metrics,
              "after": summarize(val_rows, temperature), "sampling": audit,
              "evaluated_artifact": str(out), "save_reload_probe_max_delta": reload_delta,
              "evaluation_batch_size": args.batch_size, "evaluation_token_budget": args.token_budget,
              "note": "Released validation subset; used for model selection. Not a CJ-Bench test score."}
    (run / "calibration.json").write_text(json.dumps(report, indent=2))
    (out / "decision_config.json").write_text(json.dumps({"model_name": model_name, "temperature": temperature, "prompt_version": config["prompt_version"],
                                                           "max_length": config["max_length"], "max_options": len(LETTERS),
                                                           "source_run": str(run), "merge_probe_max_logit_delta": delta}, indent=2))
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"merged": str(out), "temperature": temperature, "parity_max_delta": delta,
                      "raw": raw_metrics["all"], "calibrated": report["after"]["all"]}), flush=True)
    # Release the calibration model before testing the actual exported HTTP artifact.
    del model
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(Path(__file__).with_name("verify_export.py")), "--model", str(out),
                    "--report", str(run / "export_verification.json")], check=True)


if __name__ == "__main__":
    main()
