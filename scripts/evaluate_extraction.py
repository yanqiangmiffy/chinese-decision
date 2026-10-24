"""Score a multi-task run's span head on extraction files, optionally across decision thresholds.

    python scripts/evaluate_extraction.py --run runs/multitask-0.8b \\
        --files datasets/multitask-mini/extraction/resume_zero_shot.test.jsonl

Writes `extraction_eval.json` in the run directory; files keep their stem (`resume_zero_shot.test`) as key.
"""
import argparse
import json
from pathlib import Path

from decision.metrics import entity_metrics
from decision.pipeline import DecisionPipeline
from decision.utils import log, write_json


def load_pipeline(base, run, device):
    """A training run's adapter and span head on `base`, or an exported model directory."""
    if (Path(run) / "adapter").exists():
        return DecisionPipeline.from_run(base, run, device)
    return DecisionPipeline.from_pretrained(run, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="models/chinese-decision-0.8b")
    parser.add_argument("--run", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--thresholds", default="0", help="comma-separated minimum entity probabilities")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pipe = load_pipeline(args.base, args.run, args.device)
    result = {}
    for path in args.files:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        for threshold in map(float, args.thresholds.split(",")):
            items, predictions = pipe.predict_spans(rows, threshold)
            metrics = entity_metrics(items, predictions)
            result.setdefault(Path(path).stem, {})[f"threshold={threshold:g}"] = metrics
            log("extraction_eval", file=path, threshold=threshold, **metrics)
    output = Path(args.run) / "extraction_eval.json"
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    write_json(output, previous | result)


if __name__ == "__main__":
    main()
