"""Replay and independent validation sets that keep the original decision abilities in view.

Joint training on CJ-IE alone erased much of what chinese-decision-0.8b learned from CJ-Dataset
(CJ-Bench 83.1% -> 71.0%; legal -27.9, finance -20.2 points). This builds, from CJ-Dataset only:

    train.cases.jsonl        replay decisions mixed into joint training (scripts/train_multitask_ddp.py --replay)
    dev.cases.jsonl          checkpoint selection: validation partitions 1-4 (--cj-dev)
    calibration.cases.jsonl  temperature fitting: validation partition 0 (--cj-calibration)
    manifest.json

Replay is drawn from the training split (datasets/CJ-mixed shards one domain/source/type each).
Every domain gets a budget (--budget); within a domain each shard gets a share proportional to the
square root of its size, so small sources are not drowned by the largest ones, capped by its size,
with leftovers handed on. Each shard is sampled uniformly in one pass (selection sampling).
Validation follows decision.data.validation_items: a seeded reservoir per domain/type over the
released validation split, restricted to the partitions above, so selection never sees the
calibration cases and CJ-Bench is never read. `_meta.source_family` becomes `cj:<domain>` (the
original stays in `_meta.cj_source_family`) so metrics group by domain.

    python scripts/build_replay_data.py --out datasets/cj-replay
"""
import argparse
import gzip
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from decision.data import DOMAINS
from decision.schema import partition

DEFAULT_BUDGET = {"general": 200000, "medical": 100000, "legal": 0, "finance": 0}   # 0 = the whole domain


def allocate(sizes, budget):
    """Shard -> count: square-root shares of `budget`, capped by shard size, leftovers redistributed."""
    if budget <= 0 or budget >= sum(sizes.values()):
        return dict(sizes)
    counts, open_ = {key: 0 for key in sizes}, set(sizes)
    remaining = budget
    while remaining > 0 and open_:
        weights = {key: math.sqrt(sizes[key]) for key in open_}
        total = sum(weights.values())
        handed = 0
        for key in sorted(open_):
            share = min(sizes[key] - counts[key], max(1, int(remaining * weights[key] / total)))
            counts[key] += share
            handed += share
        remaining -= handed
        open_ = {key for key in open_ if counts[key] < sizes[key]}
        if handed == 0:
            break
    while remaining < 0:                                   # rounding up with max(1, ...) can overshoot
        key = max(counts, key=counts.get)
        counts[key] -= 1
        remaining += 1
    return counts


def sample_shard(spec):
    """Uniform `k` of a shard's `n` lines in one pass (Knuth's selection sampling)."""
    path, domain, n, k, seed = spec
    rng, chosen, seen = random.Random(seed), [], 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if rng.random() * (n - seen) < k - len(chosen):
                chosen.append(retag(json.loads(line), domain))
            seen += 1
            if len(chosen) == k:
                break
    if len(chosen) != k:
        raise ValueError(f"{path}: sampled {len(chosen)} of {k} (manifest says {n} lines, read {seen})")
    return chosen


def retag(case, domain):
    meta = dict(case.get("_meta") or {})
    meta["cj_source_family"] = meta.get("source_family")
    meta["source_family"] = f"cj:{domain}"
    meta["cj_domain"] = domain
    return dict(case, _meta=meta)


def validation_cases(data, per_stratum, subset, seed):
    """Seeded reservoir per (domain, question type) over the validation split, restricted to `subset` partitions."""
    rng, reservoirs, seen = random.Random(seed), {}, {}
    for domain in DOMAINS:
        with gzip.open(Path(data) / domain / "validation.cases.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                case = json.loads(line)
                if partition(case) not in subset:
                    continue
                kind, = {q["type"] for q in case["questions"].values()}
                key = (domain, kind)
                seen[key] = seen.get(key, 0) + 1
                bucket = reservoirs.setdefault(key, [])
                if len(bucket) < per_stratum:
                    bucket.append(retag(case, domain))
                else:
                    i = rng.randrange(seen[key])
                    if i < per_stratum:
                        bucket[i] = retag(case, domain)
    return [case for key in sorted(reservoirs) for case in reservoirs[key]], {f"{d}:{k}": n for (d, k), n in seen.items()}


def write(path, cases):
    with open(path, "w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixed", default="/data/users/yanqiang/chinese-decision/datasets/CJ-mixed")
    parser.add_argument("--data", default="/data/users/yanqiang/chinese-decision/datasets/CJ-Dataset")
    parser.add_argument("--out", default="datasets/cj-replay")
    parser.add_argument("--budget", default=",".join(f"{d}={n}" for d, n in DEFAULT_BUDGET.items()),
                        help="domain=count pairs; 0 keeps the whole domain")
    parser.add_argument("--dev-per-stratum", type=int, default=300)
    parser.add_argument("--calibration-per-stratum", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    budget = {d: int(n) for d, n in (item.split("=") for item in args.budget.split(","))}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(args.mixed) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("validation_included") is not False:
        raise ValueError("CJ-mixed must hold training data only")
    specs, plan = [], {}
    for domain in DOMAINS:
        strata = [s for s in manifest["strata"] if s["domain"] == domain]
        counts = allocate({s["path"]: s["decisions"] for s in strata}, budget.get(domain, 0))
        for k, s in enumerate(strata):
            if counts[s["path"]]:
                specs.append((str(Path(args.mixed) / s["path"]), domain, s["decisions"], counts[s["path"]],
                              f"{args.seed}:{s['path']}"))
                plan[s["path"]] = {"domain": domain, "source": s["source"], "kind": s["kind"],
                                   "available": s["decisions"], "replayed": counts[s["path"]]}
    with ProcessPoolExecutor(args.workers) as pool:
        replay = [case for chosen in pool.map(sample_shard, specs) for case in chosen]
    random.Random(args.seed).shuffle(replay)
    dev, dev_seen = validation_cases(args.data, args.dev_per_stratum, (1, 2, 3, 4), args.seed)
    calibration, _ = validation_cases(args.data, args.calibration_per_stratum, (0,), args.seed + 1)
    counts = {"train": write(out / "train.cases.jsonl", replay), "dev": write(out / "dev.cases.jsonl", dev),
              "calibration": write(out / "calibration.cases.jsonl", calibration)}
    by_domain = {d: sum(p["replayed"] for p in plan.values() if p["domain"] == d) for d in DOMAINS}
    (out / "manifest.json").write_text(json.dumps({
        "source_training": args.mixed, "source_validation": args.data, "seed": args.seed, "budget": budget,
        "files": counts, "replay_by_domain": by_domain, "validation_rows_seen": dev_seen,
        "dev_partitions": [1, 2, 3, 4], "calibration_partitions": [0], "cj_bench_read": False,
        "strata": plan}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": counts, "replay_by_domain": by_domain}, ensure_ascii=False))


if __name__ == "__main__":
    main()
