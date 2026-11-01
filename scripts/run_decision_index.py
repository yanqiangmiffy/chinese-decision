"""Run the official Decision Index suite in shards (one process per GPU), then merge and score.

Each shard is the official runner (`decision_index.runner.run`, checkpoint/resume included) over the
requests whose run_id hashes to it, so shards can run on separate GPUs at the same time:

    python scripts/run_decision_index.py run --suite-dir S --model M --out runs/di --shards 4 --shard 0
    ...
    python scripts/run_decision_index.py merge --suite-dir S --out runs/di --shards 4

`merge` concatenates the shards' results.jsonl into <out>/results.jsonl, keeps shard 0's
environment.json, and writes scores.json / benchmark-summary.json / index.json with the official scorer.
"""
import argparse
import hashlib
import json
from pathlib import Path

from decision_index.pipeline import score_run
from decision_index.runner import run
from decision_index.suite.io import Suite

ENGINE = "decision.decision_index_engine:ChineseDecisionEngine"


def shard_of(run_id, shards):
    return int(hashlib.sha256(run_id.encode("utf-8")).hexdigest(), 16) % shards


def cmd_run(args):
    suite = Suite(args.suite_dir, args.edition)
    corpus = suite.verify(strict=True)["sha256"]
    keep = lambda e: suite.in_edition(e) and shard_of(e["run_id"], args.shards) == args.shard
    options = {"model": args.model, "device": args.device, **{k: json.loads(v) for k, v in
                                                              (o.split("=", 1) for o in args.option)}}
    out = Path(args.out) / f"shard-{args.shard}-of-{args.shards}"
    print(json.dumps(run(ENGINE, options, suite.row_paths, out, limit=args.limit, compact=True,
                         corpus_sha256=corpus, keep=keep)), flush=True)


def cmd_merge(args):
    suite = Suite(args.suite_dir, args.edition)
    out = Path(args.out)
    shards = [out / f"shard-{i}-of-{args.shards}" for i in range(args.shards)]
    seen, lines = {}, []
    for shard in shards:
        for line in (shard / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["status"] == "error" and row["run_id"] in seen:
                continue
            if row["run_id"] in seen:  # a resumed retry of an error replaces it
                lines[seen[row["run_id"]]] = line
            else:
                seen[row["run_id"]] = len(lines)
                lines.append(line)
    (out / "results.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    environment = json.loads((shards[0] / "environment.json").read_text(encoding="utf-8"))
    environment["shards"] = [{"dir": s.name, **{k: v for k, v in json.loads(
        (s / "environment.json").read_text(encoding="utf-8")).items() if k in ("gpu", "device", "loaded_seconds")}}
        for s in shards]
    (out / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    scores = score_run(suite, out / "results.jsonl", args.engine_name, out)
    print(json.dumps({k: scores[k] for k in ("decision_index", "completed", "complete", "counts", "latency_ms")},
                     indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    for name, func in (("run", cmd_run), ("merge", cmd_merge)):
        p = sub.add_parser(name)
        p.add_argument("--suite-dir", required=True)
        p.add_argument("--edition", default="0.3")
        p.add_argument("--out", required=True)
        p.add_argument("--shards", type=int, default=1)
        p.set_defaults(func=func)
        if name == "run":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--model", required=True)
            p.add_argument("--device", default="cuda")
            p.add_argument("--option", action="append", default=[], help="engine option key=JSON value")
            p.add_argument("--limit", type=int)
        else:
            p.add_argument("--engine-name", default="chinese-decision-4b")
    args = parser.parse_args()
    if args.func is cmd_run and not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    args.func(args)


if __name__ == "__main__":
    main()
