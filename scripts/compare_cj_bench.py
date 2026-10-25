"""Side-by-side CJ-Bench results of several full.json reports.

    python scripts/compare_cj_bench.py original=<full.json> v1=<full.json> v2=<full.json> v3=<full.json> \\
        --out runs/cj-bench/comparison.json

Prints accuracy / hard NLL / ECE per domain and per question type, plus the change from the first report.
"""
import argparse
import json
from pathlib import Path

ROWS = ["all", "domain:general", "domain:medical", "domain:legal", "domain:finance",
        "type:choice", "type:noul", "type:score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", help="name=path/to/full.json; the first is the reference")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    reports = {}
    for item in args.reports:
        name, _, path = item.partition("=")
        reports[name] = json.loads(Path(path).read_text(encoding="utf-8"))["metrics"]["groups"]
    names = list(reports)
    table = {key: {name: {m: reports[name][key].get(m) for m in ("n", "accuracy", "hard_nll", "ece")}
                   for name in names if key in reports[name]} for key in ROWS}
    width = 24
    for metric, scale in (("accuracy", 100), ("hard_nll", 1), ("ece", 1)):
        print(f"\n{metric}" + (" (%)" if scale == 100 else ""))
        print(f"{'':20s}" + "".join(f"{name:>{width // 2 + 2}s}" for name in names)
              + ("".join(f"{'Δ ' + name:>{width // 2 + 2}s}" for name in names[1:]) if scale == 100 else ""))
        for key in ROWS:
            values = [table[key].get(name, {}).get(metric) for name in names]
            cells = "".join(f"{(v * scale if v is not None else float('nan')):{width // 2 + 2}.{2 if scale == 100 else 3}f}" for v in values)
            deltas = ""
            if scale == 100:
                deltas = "".join(f"{(v - values[0]) * scale:+{width // 2 + 2}.2f}" if v is not None and values[0] is not None
                                 else f"{'-':>{width // 2 + 2}s}" for v in values[1:])
            print(f"{key:20s}{cells}{deltas}")
    if args.out:
        Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
