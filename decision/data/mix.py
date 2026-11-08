"""The decision training mix: catalog datasets as CJ cases, deduplicated against the Decision Index.

Every source adapter (decision/data/mix_sources.py) yields CJ cases with one question each, the
`prepare_domain` contract:

    {"id", "state", "questions": {"q": {"type", "instructions", "criteria"}}, "gold": {"q": ...},
     "_meta": {"source", "source_family", "group_key", "lang", "license", "label_origin"}}

`build_mix` validates every case with the training reader, drops cases whose text shares a
13-unit span with a Decision Index 0.3 row (`Decontaminator`), keeps at most `cap` cases per
source (a seeded reservoir), and splits train/validation by group, so a group never straddles both.
"""
import gzip
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from ..configuration import POINTER_MAX_OPTIONS
from ..processing import examples_for, group_key

UNIT = re.compile(r"[a-z0-9]+|[㐀-鿿豈-﫿]")
SHINGLE = 13
# Spans in this many suite rows or more are boilerplate (templates, schema keys), not leaked items.
COMMON_ROWS = 20


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _options(options):
    """`{key: description}` from `[(key, description)]` or `[key]`; descriptions default to the key."""
    criteria = {}
    for option in options:
        key, description = option if isinstance(option, (tuple, list)) else (option, None)
        key = str(key).strip()
        if not key or key in criteria:
            raise ValueError(f"empty or duplicate candidate key {key!r}")
        criteria[key] = _text(description).strip() if description not in (None, "") else key
    return criteria


def case(source, rid, state, kind, instructions, criteria, gold, *, group=None, lang="en", license=None,
         family=None, label_origin="gold"):
    return {"id": f"{source}:{rid}", "state": state,
            "questions": {"q": {"type": kind, "instructions": instructions, "criteria": criteria}},
            "gold": {"q": gold},
            "_meta": {"source": source, "source_family": family or source, "group_key": f"{source}:{group or rid}",
                      "lang": lang, "license": license, "label_origin": label_origin}}


def choice(source, rid, state, instructions, options, gold, probabilities=None, **meta):
    """A choice case; `gold` is the answer key, or `probabilities` a distribution aligned with `options`."""
    criteria = _options(options)
    keys = list(criteria)
    if probabilities is not None:
        total = sum(probabilities)
        if len(probabilities) != len(keys) or total <= 0:
            raise ValueError("probabilities must cover every candidate")
        answer = {"type": "choice", "probabilities": {k: p / total for k, p in zip(keys, probabilities)}}
    else:
        answer = str(gold).strip()
        if answer not in criteria:
            raise ValueError(f"gold {answer!r} is not a candidate")
    return case(source, rid, state, "choice", instructions, criteria, answer, **meta)


def noul(source, rid, state, instructions, answer, true=None, false=None, **meta):
    """A yes/no case; `answer` is a bool or P(true)."""
    criteria = {"false": false or "No", "true": true or "Yes"}
    if isinstance(answer, bool):
        gold = "true" if answer else "false"
    else:
        p = float(answer)
        if not 0 <= p <= 1:
            raise ValueError("noul probability out of range")
        gold = {"type": "noul", "probabilities": {"false": 1 - p, "true": p}}
    return case(source, rid, state, "noul", instructions, criteria, gold, **meta)


def score(source, rid, state, instructions, levels, gold, probabilities=None, **meta):
    """An ordered-levels case; `gold` is a level index or `probabilities` over the levels."""
    if probabilities is not None:
        total = sum(probabilities)
        answer = {"type": "score", "probabilities": {str(i): p / total for i, p in enumerate(probabilities)}}
    else:
        answer = str(int(gold))
    return case(source, rid, state, "score", instructions, [str(level) for level in levels], answer, **meta)


# --- deduplication ---------------------------------------------------------------------------------

def _hash(span):
    return int.from_bytes(hashlib.blake2b(span.encode("utf-8"), digest_size=8).digest(), "little")


def shingles(text):
    """Hashes of every 13-unit span of `text` (lower-cased words and numbers; one CJK character per unit)."""
    units = UNIT.findall(text.lower())
    return {_hash(" ".join(units[i:i + SHINGLE])) for i in range(len(units) - SHINGLE + 1)}


class Decontaminator:
    """Sorted span hashes of the Decision Index rows; `hits(text)` counts shared spans."""

    def __init__(self, index):
        import numpy as np
        self.np, self.index = np, index

    @classmethod
    def from_suite(cls, suite_dir, cache=None, log=print):
        import numpy as np
        if cache and Path(cache).exists():
            return cls(np.load(cache))
        started, rows = time.time(), []
        for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz", "gsm8k-rows.jsonl.gz"):
            path = Path(suite_dir) / name
            if not path.exists():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    text = _text(row["state"]) + " " + " ".join(
                        _text(q.get("criteria")) for q in row.get("questions", {}).values())
                    rows.append(np.fromiter(shingles(text), dtype=np.uint64))
        values, counts = np.unique(np.concatenate(rows), return_counts=True)
        index = values[counts < COMMON_ROWS]
        log(json.dumps({"event": "decontamination_index", "rows": len(rows), "spans": int(len(values)),
                        "kept": int(len(index)), "seconds": round(time.time() - started)}))
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, index)
        return cls(index)

    def hits(self, text):
        found = self.np.fromiter(shingles(text), dtype=self.np.uint64)
        if not len(found):
            return 0
        position = self.np.searchsorted(self.index, found).clip(max=len(self.index) - 1)
        return int((self.index[position] == found).sum())


def validation_bucket(group, rate=0.03):
    """About `rate` of groups, independent of `partition`."""
    return int(hashlib.sha256(("validation:" + group).encode()).hexdigest()[:8], 16) / 2 ** 32 < rate


def gold_label(case):
    """The gold answer key of a one-question case (the most probable key of a distribution)."""
    gold = case["gold"]["q"]
    if isinstance(gold, dict):
        probabilities = gold["probabilities"]
        return max(probabilities, key=probabilities.get)
    return str(gold)


def case_text(case):
    question = case["questions"]["q"]
    return _text(case["state"]) + " " + _text(question.get("criteria"))


# --- building ----------------------------------------------------------------------------------------

def pack(cases, size):
    """Merge sampled cases about one state (same group and state) into multi-question cases of at most `size`.

    The pointer readout answers them in one sequence; reading each alone is unchanged for other readouts.
    A case with `_meta.pack = False` stays alone: versions of one question with different candidate lists
    (GSM8K with 4 and 10 options) would otherwise give the answer away as the one candidate they share.
    """
    by_state = {}
    for case in cases:
        alone = case["_meta"].get("pack") is False
        by_state.setdefault((group_key(case), _text(case["state"]), case["id"] if alone else None), []).append(case)
    for members in by_state.values():
        for start in range(0, len(members), size):
            chunk = members[start:start + size]
            if len(chunk) == 1:
                yield chunk[0]
                continue
            merged = {"id": chunk[0]["id"] + f"+{len(chunk)}", "state": chunk[0]["state"], "questions": {}, "gold": {},
                      "_meta": dict(chunk[0]["_meta"], packed_ids=[c["id"] for c in chunk])}
            for n, case in enumerate(chunk):
                merged["questions"][f"q{n}"] = case["questions"]["q"]
                merged["gold"][f"q{n}"] = case["gold"]["q"]
            yield merged


def build_source(source, staging, catalog, decontaminator, seed, validation_cap, limit, pack_size, log):
    """Sample, pack and split one source into `staging/<name>.{train,validation}.jsonl.gz` plus `<name>.json`.

    The stats file is written last and only when the adapter finished, so an interrupted or failed
    source is built again by the next run.
    """
    started, stats = time.time(), Counter()
    rng, kept = random.Random(f"{seed}:{source.name}"), []
    balance = getattr(source, "balance", False)
    by_label, seen_label = {}, Counter()
    dedup_text = getattr(source, "dedup", None) or case_text
    try:
        for case in source.read(Path(catalog)):
            stats["seen"] += 1
            if limit and stats["seen"] > limit:
                break
            try:
                list(examples_for(case, source.domain, POINTER_MAX_OPTIONS))
            except (ValueError, KeyError, TypeError):
                stats["invalid"] += 1
                continue
            if decontaminator is not None and decontaminator.hits(dedup_text(case)):
                stats["overlap_dropped"] += 1
                continue
            stats["valid"] += 1
            if balance:                       # one reservoir per gold label, trimmed to the rarest label below
                label = gold_label(case)
                seen_label[label] += 1
                bucket = by_label.setdefault(label, [])
                if len(bucket) < source.cap:
                    bucket.append(case)
                else:
                    j = rng.randrange(seen_label[label])
                    if j < source.cap:
                        bucket[j] = case
                continue
            if len(kept) < source.cap:
                kept.append(case)
            else:
                j = rng.randrange(stats["valid"])
                if j < source.cap:
                    kept[j] = case
    except FileNotFoundError as exc:
        log(json.dumps({"event": "source_missing", "source": source.name, "error": str(exc)}))
        return {"domain": source.domain, "missing": str(exc)}
    except Exception as exc:  # one broken adapter must not stop the build; it is reported and retried next run
        error = f"{type(exc).__name__}: {exc}"
        log(json.dumps({"event": "source_error", "source": source.name, "error": error}, ensure_ascii=False))
        return {"domain": source.domain, "error": error}
    if balance and by_label:
        each = min(min(len(b) for b in by_label.values()), source.cap // len(by_label))
        kept = [case for label in sorted(by_label) for case in by_label[label][:each]]
        stats.update({f"label:{label}": each for label in by_label})
        stats["balanced_from"] = sum(seen_label.values())
    for case in kept:
        stats["kind:" + case["questions"]["q"]["type"]] += 1
    # Training reads each stratum in file order through a small shuffle buffer, so the file itself must be
    # shuffled: balanced sources come out grouped by label, and sources under their cap keep the raw file's
    # order (often sorted by label), which made the last part of training see one label only.
    packed = list(pack(kept, getattr(source, "pack_size", None) or pack_size))
    rng.shuffle(packed)
    held, files = set(), {}
    try:
        for case in packed:
            case["_meta"].update(domain=source.domain)
            # Validation draws whole groups by a hash of their own, so it holds every partition (training
            # selects on partitions 1-4 and calibrates on 0, like CJ's validation split).
            group = group_key(case)
            if group not in held and stats["validation"] < validation_cap and validation_bucket(group):
                held.add(group)
            split = "validation" if group in held else "train"
            stats[split] += len(case["questions"])
            stats[split + "_lines"] += 1
            if split not in files:
                files[split] = gzip.open(staging / f"{source.name}.{split}.jsonl.gz", "wt", encoding="utf-8")
            files[split].write(json.dumps(case, ensure_ascii=False) + "\n")
    finally:
        for stream in files.values():
            stream.close()
    result = {"domain": source.domain, "lang": source.lang, "license": source.license, "cap": source.cap, **stats,
              "seconds": round(time.time() - started, 1)}
    (staging / f"{source.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def build_mix(sources, out, catalog, decontaminator=None, seed=42, validation_cap=300, limit=None, log=print,
              pack_size=8):
    """Write `out/<domain>/{train,validation}.cases.jsonl.gz` and `out/build_report.json`.

    Sources are built one by one into `out/_staging` and reused when a run is repeated (resume, or a
    rebuilt source after deleting its `_staging/<name>.json`); the domain files are assembled from them.
    `limit` reads at most that many cases per source (smoke tests). Counts in the report are questions
    (`train`, `validation`) and lines (`*_lines`).
    """
    out = Path(out)
    staging = out / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    report = {}
    for source in sources:
        done = staging / f"{source.name}.json"
        if done.exists():
            report[source.name] = json.loads(done.read_text(encoding="utf-8"))
            log(json.dumps({"event": "source_reused", "source": source.name}))
            continue
        report[source.name] = build_source(source, staging, catalog, decontaminator, seed, validation_cap, limit,
                                           pack_size, log)
        log(json.dumps({"event": "source_built", "source": source.name, **report[source.name]}, ensure_ascii=False))
    built = {name: r for name, r in report.items() if "missing" not in r and "error" not in r}
    domains = sorted({r["domain"] for r in built.values()})
    for domain in domains:
        for split in ("train", "validation"):
            path = out / domain / f"{split}.cases.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as target:          # gzip members concatenate into one valid gzip file
                for name, r in built.items():
                    part = staging / f"{name}.{split}.jsonl.gz"
                    if r["domain"] == domain and part.exists():
                        target.write(part.read_bytes())
            if path.stat().st_size == 0:
                gzip.open(path, "wt", encoding="utf-8").close()
    summary = {"domains": {d: {s: sum(r.get(s, 0) for r in built.values() if r["domain"] == d)
                               for s in ("train", "validation", "train_lines", "validation_lines")} for d in domains},
               "sources": report, "seed": seed, "validation_cap": validation_cap, "pack_size": pack_size,
               "decontaminated": decontaminator is not None,
               "validation_groups": "about 3% of groups by their own hash (all partitions), at most validation_cap per source",
               "incomplete": sorted(set(report) - set(built))}
    (out / "build_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def prepare_mix(raw, prepared, workers=4, revision="decision-mix-v1"):
    """`prepare_domain` for every domain of a built mix, plus the manifest `scripts/train.py` reads."""
    from concurrent.futures import ProcessPoolExecutor
    from .cj import prepare_domain
    report = json.loads((Path(raw) / "build_report.json").read_text(encoding="utf-8"))
    specs = [(str(raw), str(prepared), domain, counts["train_lines"], True) for domain, counts in report["domains"].items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(prepare_domain, specs))
    manifest = {"source_revision": revision, "decisions": sum(r["decisions"] for r in results),
                "strata": [entry for r in results for entry in r["strata"]], "validation_included": False}
    Path(prepared, "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


__all__ = ["Decontaminator", "build_mix", "case", "choice", "group_key", "noul", "prepare_mix", "score", "shingles"]
