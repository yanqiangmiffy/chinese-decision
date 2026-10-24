"""Bounded-memory CJ-Dataset reading, re-sharding, mixing, and validation sampling.

Original train/validation files stay separate; validation is never mixed into training.
"""
import gzip
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from ..processing import examples_for, partition

DOMAINS = ("general", "medical", "legal", "finance")
FULL_TRAIN_DECISIONS = 13_048_000


def read_examples(root, domain, split, stats=None, subset=None):
    path = Path(root) / domain / f"{split}.cases.jsonl.gz"
    stats = stats if stats is not None else Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            stats[f"{domain}:rows"] += 1
            if subset is not None and partition(case) not in subset:
                continue
            try:
                examples = list(examples_for(case, domain))
            except (ValueError, KeyError, TypeError) as exc:
                # Invalid labels must never silently become class zero.
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            for example in examples:
                stats[f"{domain}:{example.kind}"] += 1
                yield example


def prepare_domain(spec):
    """Split one domain's training file into per-(source, type) shards for `prepared_examples`.

    `spec` is `(data, destination, domain, expected_decisions)`, a single tuple so the
    function can be mapped over a process pool.
    """
    data, destination, domain, expected = spec
    data, destination = Path(data), Path(destination)
    marker = destination / domain / "manifest.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["decisions"] == expected and all((destination / x["path"]).exists() for x in result["strata"]):
            return result
        raise ValueError(f"incomplete existing preprocessing output: {marker}")
    directory = destination / domain
    directory.mkdir(parents=True, exist_ok=True)
    files, descriptions = {}, {}
    started, count = time.time(), 0
    try:
        with gzip.open(data / domain / "train.cases.jsonl.gz", "rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                case = json.loads(line)
                if len(case["questions"]) != 1:
                    raise ValueError("preprocessing expects CJ's one-decision-per-line contract")
                question, = case["questions"].values()
                meta = case.get("_meta") or {}
                source = str(meta.get("source_family") or meta.get("source_id") or meta.get("source") or "unknown")
                kind = question["type"]
                if kind not in ("choice", "noul", "score"):
                    raise ValueError(f"unknown decision type {kind}")
                key = (source, kind)
                if key not in files:
                    name = hashlib.sha256(source.encode()).hexdigest()[:16] + "-" + kind + ".jsonl.gz"
                    path = directory / name
                    files[key] = gzip.open(path.with_suffix(path.suffix + ".partial"), "wb", compresslevel=1)
                    descriptions[key] = {"domain": domain, "source": source, "kind": kind,
                                         "path": str(path.relative_to(destination)), "decisions": 0}
                files[key].write(line)
                descriptions[key]["decisions"] += 1
                count += 1
                if count % 250000 == 0:
                    print(json.dumps({"event": "prepare", "domain": domain, "decisions": count,
                                      "seconds": round(time.time() - started)}), flush=True)
    finally:
        for stream in files.values():
            stream.close()
    if count != expected:
        raise ValueError(f"{domain}: expected {expected} decisions; read {count}")
    for entry in descriptions.values():
        path = destination / entry["path"]
        path.with_suffix(path.suffix + ".partial").replace(path)
    result = {"domain": domain, "decisions": count, "strata": list(descriptions.values())}
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"event": "prepared", "domain": domain, "decisions": count,
                      "strata": len(descriptions), "seconds": round(time.time() - started)}), flush=True)
    return result


def shuffled(stream, rng, buffer_size):
    buffer = []
    for item in stream:
        if len(buffer) < buffer_size:
            buffer.append(item)
        else:
            i = rng.randrange(len(buffer))
            yield buffer[i]
            buffer[i] = item
    rng.shuffle(buffer)
    yield from buffer


def prepared_examples(root, domains, seed, stats, buffer_size=256):
    """Interleave shards in proportion to their remaining decisions; each one is visited once."""
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("validation_included") is not False:
        raise ValueError("prepared corpus must contain training data only")
    entries = [entry for entry in manifest["strata"] if entry["domain"] in domains]
    def read(entry):
        with gzip.open(Path(root) / entry["path"], "rt", encoding="utf-8") as stream:
            for line in stream:
                case = json.loads(line)
                stats[entry["domain"] + ":rows"] += 1
                for example in examples_for(case, entry["domain"]):
                    stats[entry["domain"] + ":" + example.kind] += 1
                    yield example
    streams = {i: iter(shuffled(read(entry), random.Random(seed + i), buffer_size)) for i, entry in enumerate(entries)}
    remaining = {i: entry["decisions"] for i, entry in enumerate(entries)}
    rng = random.Random(seed)
    while streams:
        i = rng.choices(list(streams), [remaining[j] for j in streams], k=1)[0]
        try:
            yield next(streams[i])
            remaining[i] -= 1
            if remaining[i] == 0:
                del streams[i]
        except StopIteration:
            raise ValueError(f"prepared stratum has fewer rows than its manifest: {entries[i]['path']}")


def validation_items(root, domains, processor, per_stratum=100, subset=(1, 2, 3, 4), seed=42):
    """Reservoir samples every domain/type over the entire released validation split."""
    rng, reservoirs, seen, stats = random.Random(seed), {}, Counter(), Counter()
    for domain in domains:
        for example in read_examples(root, domain, "validation", stats, subset):
            key = (domain, example.kind)
            seen[key] += 1
            bucket = reservoirs.setdefault(key, [])
            if len(bucket) < per_stratum:
                bucket.append(example)
            else:
                i = rng.randrange(seen[key])
                if i < per_stratum:
                    bucket[i] = example
    chosen = [processor.encode_example(example, stats=stats) for bucket in reservoirs.values() for example in bucket]
    return [item for item in chosen if item is not None], dict(stats)
