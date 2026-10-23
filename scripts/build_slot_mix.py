"""Add N synthesized intent + slot records to a multi-task mix, with a synthesized dev split and a test file.

Every record becomes an intent decision (decision head) and, when its intent has slots, a span
extraction row (span head). The pool is shuffled with a fixed seed and dev comes first, so
training sets of different N are nested prefixes of one order.

    python scripts/build_slot_mix.py --n 500 --out datasets/multitask-slots-500
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from decision.slots import decision_case, extraction_row, load_schema


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def append(path, rows):
    with open(path, "a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="datasets/multitask-mini")
    parser.add_argument("--pool", default="datasets/slots/pool.jsonl")
    parser.add_argument("--test", default="examples/slot_filling/test_handwritten.jsonl")
    parser.add_argument("--schema", default="examples/slot_filling/schema.json")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dev", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    schema, out = load_schema(args.schema), Path(args.out)
    pool = read(args.pool)
    random.Random(args.seed).shuffle(pool)
    if args.dev + args.n > len(pool):
        raise ValueError(f"pool has {len(pool)} records; need {args.dev + args.n}")
    splits = {"train": pool[args.dev:args.dev + args.n], "dev": pool[:args.dev], "test": read(args.test)}
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(args.base, out)
    counts = {}
    for split, records in splits.items():
        rows = [row for row in (extraction_row(r, schema, split) for r in records) if row is not None]
        append(out / "extraction" / f"slots.{split}.jsonl", rows)
        append(out / "decision" / f"{split}.cases.jsonl", [decision_case(r, schema, split) for r in records])
        counts[split] = {"records": len(records), "extraction_rows": len(rows),
                         "intents": dict(Counter(r["intent"] for r in records))}
    (out / "slot_mix.json").write_text(json.dumps({"arguments": vars(args), "counts": counts}, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
