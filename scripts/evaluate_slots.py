"""End-to-end intent + slot evaluation of a multi-task run: query -> {intent: {slot: value}}.

    python scripts/evaluate_slots.py --run runs/mt-slots-500

Writes `slots_eval.json` (metrics and one prediction per query) in the run directory. Slots are
scored as (slot, value) pairs, so a slot predicted under a wrong intent counts as a false positive.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from decision.data.slots import load_schema
from decision.pipeline import SlotFillingPipeline
from decision.utils import log, write_json
from evaluate_extraction import load_pipeline


def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def score(rows, predictions):
    counts, per_slot, per_intent = Counter(), {}, {}
    for row, pred in zip(rows, predictions):
        gold, found = set(row["slots"].items()), set(pred["slots"].items())
        for kind, pairs in (("tp", gold & found), ("fp", found - gold), ("fn", gold - found)):
            counts[kind] += len(pairs)
            for key, _ in pairs:
                per_slot.setdefault(key, Counter())[kind] += 1
        intent_ok = pred["intent"] == row["intent"]
        exact = intent_ok and pred["slots"] == row["slots"]
        counts["intent"] += intent_ok
        counts["exact"] += exact
        bucket = per_intent.setdefault(row["intent"], Counter())
        bucket["n"] += 1
        bucket["intent"] += intent_ok
        bucket["exact"] += exact
    return {"n": len(rows), "intent_accuracy": counts["intent"] / len(rows), "exact_match": counts["exact"] / len(rows),
            "slots": f1(counts["tp"], counts["fp"], counts["fn"]),
            "per_slot": {k: dict(f1(c["tp"], c["fp"], c["fn"]), gold=c["tp"] + c["fn"]) for k, c in sorted(per_slot.items())},
            "per_intent": {k: {"n": c["n"], "intent_accuracy": c["intent"] / c["n"], "exact_match": c["exact"] / c["n"]}
                           for k, c in sorted(per_intent.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="models/chinese-decision-0.8b")
    parser.add_argument("--run", required=True)
    parser.add_argument("--schema", default="examples/slot_filling/schema.json")
    parser.add_argument("--test", default="examples/slot_filling/test_handwritten.jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    filler = SlotFillingPipeline(load_pipeline(args.base, args.run, args.device), load_schema(args.schema))
    rows = [json.loads(line) for line in open(args.test, encoding="utf-8") if line.strip()]
    predictions = [filler(row["query"]) for row in rows]
    metrics = score(rows, predictions)
    examples = [{"query": row["query"], "response": {pred["intent"]: pred["slots"]},
                 "gold": {row["intent"]: row["slots"]}, "correct": pred["intent"] == row["intent"] and pred["slots"] == row["slots"],
                 "intent_probability": round(pred["intent_probability"], 4), "spans": pred["spans"]}
                for row, pred in zip(rows, predictions)]
    write_json(Path(args.run) / "slots_eval.json", {"test": args.test, "metrics": metrics, "predictions": examples})
    log("slots_eval", run=args.run, **{k: metrics[k] for k in ("n", "intent_accuracy", "exact_match", "slots")})


if __name__ == "__main__":
    main()
