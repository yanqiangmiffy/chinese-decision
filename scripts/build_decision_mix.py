"""Build the decision training mix from the downloaded catalog, then shard it for scripts/train.py.

    python scripts/build_decision_mix.py --catalog G:/pretrained_models/datasets/decision-training-catalog \\
        --out datasets/decision-mix-v1/raw --prepared datasets/decision-mix-v1/prepared

Every case is checked with the training reader and against the Decision Index 0.3 suite (13-unit spans,
see decision/data/mix.py); `raw/build_report.json` lists what each source kept and dropped.
Train with `--readout pointer --data datasets/decision-mix-v1/raw --prepared-data datasets/decision-mix-v1/prepared
--domains <domains in the report>` (add CJ corpora and their domains to mix in Chinese decisions).
"""
import argparse
import json
from pathlib import Path

from decision.data.mix import Decontaminator, build_mix, prepare_mix
from decision.data.mix_sources import SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="G:/pretrained_models/datasets/decision-training-catalog")
    parser.add_argument("--out", default="datasets/decision-mix-v1/raw")
    parser.add_argument("--prepared", default="datasets/decision-mix-v1/prepared",
                        help="sharded copy for training ('' to skip)")
    parser.add_argument("--suite-dir", default="datasets/decision-index/suite-0.3")
    parser.add_argument("--decontamination-cache", default="datasets/decision-mix-v1/suite-0.3-spans.npy")
    parser.add_argument("--no-decontamination", action="store_true")
    parser.add_argument("--sources", nargs="*", help="only these source names")
    parser.add_argument("--limit", type=int, help="read at most N cases per source (smoke test)")
    parser.add_argument("--validation-cap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--revision", help="manifest source_revision (default: the folder holding --out)")
    args = parser.parse_args()
    sources = [s for s in SOURCES if not args.sources or s.name in args.sources]
    unknown = set(args.sources or []) - {s.name for s in SOURCES}
    if unknown:
        parser.error(f"unknown sources: {sorted(unknown)}")
    out = Path(args.out)
    # A repeated run reuses every source already staged in out/_staging (resume after an interruption).
    decontaminator = None if args.no_decontamination else Decontaminator.from_suite(args.suite_dir, args.decontamination_cache)
    summary = build_mix(sources, out, args.catalog, decontaminator, args.seed, args.validation_cap, args.limit)
    print(json.dumps({"event": "built", "domains": summary["domains"]}, ensure_ascii=False), flush=True)
    if args.prepared:
        manifest = prepare_mix(out, args.prepared, args.workers, args.revision or out.resolve().parent.name)
        print(json.dumps({"event": "prepared", "decisions": manifest["decisions"], "strata": len(manifest["strata"])}), flush=True)


if __name__ == "__main__":
    main()
