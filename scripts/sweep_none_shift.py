"""Sweep the extraction operating point (none_shift, flat/nested decoding) of an exported unified model.

    python scripts/sweep_none_shift.py --model runs/cj-ie-4b-from-base/model --data datasets/cj-ie \\
        --shifts=-1,-0.5,0,0.5,1,1.609 --out runs/cj-ie-4b-from-base/none_shift_sweep.json

One forward pass per row; every setting is decoded from the same utilities. Settings are chosen on the
dev split only (global best micro F1, and best per source) and then reported once on the test split.
The released none_shift (log 1/none_weight) restores the trained none probability for calibration;
it is not an F1-optimal operating point.
"""
import argparse
import json
from pathlib import Path

from decision.metrics import entity_metrics
from decision.pipeline import DecisionPipeline
from decision.utils import log, write_json


def score(items, predictions):
    metrics = entity_metrics(items, predictions)
    return {key: metrics[key] for key in ("gold", "predicted", "precision", "recall", "f1", "entity_ece")}


def micro(per_source):
    gold = sum(m["gold"] for m in per_source.values())
    predicted = sum(m["predicted"] for m in per_source.values())
    tp = sum(m["recall"] * m["gold"] for m in per_source.values())
    precision, recall = (tp / predicted if predicted else 0.0), (tp / gold if gold else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro = sum(m["f1"] for m in per_source.values()) / len(per_source)
    return {"precision": precision, "recall": recall, "f1": f1, "macro_f1": macro}


def evaluate(pipe, files, settings, token_budget):
    """{setting key: {source: metrics}} for every file."""
    result = {key: {} for key, _ in settings}
    for path in files:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        items = [pipe.processor.encode_extraction(row) for row in rows]
        items = [item for item in items if item is not None]
        grid = pipe.model.predict_spans_grid(items, pipe.processor, [s for _, s in settings], token_budget)
        source = Path(path).name.split(".")[0]
        for (key, _), predictions in zip(settings, grid):
            result[key][source] = score(items, predictions)
        best = max(result, key=lambda k: result[k][source]["f1"])
        log("sweep", file=Path(path).name, rows=len(items), released=round(result[settings[0][0]][source]["f1"], 4),
            best=best, best_f1=round(result[best][source]["f1"], 4))
    return result


def combine(result, choice):
    """Per-source metrics taking, for each source, the setting `choice[source]`."""
    return {source: result[key][source] for source, key in choice.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="exported unified model directory")
    parser.add_argument("--data", default="datasets/cj-ie")
    parser.add_argument("--shifts", default="-1,-0.5,0,0.25,0.5,0.75,1,1.25,1.609")
    parser.add_argument("--nested", action="store_true", help="also decode every shift without the flat constraint")
    parser.add_argument("--sources", nargs="*", help="limit to these sources (file stems before the split)")
    parser.add_argument("--token-budget", type=int, default=12288)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pipe = DecisionPipeline.from_pretrained(args.model, device=args.device)
    released = pipe.model.config.extraction.none_shift
    shifts = [released] + [s for s in map(float, args.shifts.split(",")) if abs(s - released) > 1e-3]
    settings = [(f"shift={s:g},flat", (released, 0.0, True, s)) for s in shifts]
    if args.nested:
        settings += [(f"shift={s:g},nested", (released, 0.0, False, s)) for s in shifts]
    released_key = settings[0][0]

    def files(split):
        found = sorted((Path(args.data) / "extraction").glob(f"*.{split}.jsonl"))
        return [p for p in found if not args.sources or p.name.split(".")[0] in args.sources]

    dev = evaluate(pipe, files("dev"), settings, args.token_budget)
    dev_summary = {key: micro(per_source) for key, per_source in dev.items()}
    global_best = max(dev_summary, key=lambda k: dev_summary[k]["f1"])
    per_source_best = {source: max(dev, key=lambda k: dev[k][source]["f1"]) for source in dev[released_key]}
    log("dev_selected", released=dev_summary[released_key], global_best=global_best, **dev_summary[global_best])

    keys = sorted({released_key, global_best, *per_source_best.values()})
    test = evaluate(pipe, files("test"), [(k, s) for k, s in settings if k in keys], args.token_budget)
    report = {
        "model": args.model, "released_none_shift": released, "settings": [k for k, _ in settings],
        "dev": {"summary": dev_summary, "per_source": dev},
        "selection": {"global": global_best, "per_source": per_source_best},
        "test": {
            "released": micro(test[released_key]),
            "global": micro(test[global_best]),
            "per_source": micro(combine(test, per_source_best)),
            "per_source_metrics": {"released": test[released_key], "global": test[global_best],
                                   "per_source": combine(test, per_source_best)},
        },
    }
    write_json(args.out, report)
    log("test", released=report["test"]["released"], global_best=report["test"]["global"],
        per_source_best=report["test"]["per_source"])


if __name__ == "__main__":
    main()
