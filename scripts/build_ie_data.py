"""Build CJ-IE: a large Chinese information-extraction set for the decision and span heads.

Reads the local copies in G:/pretrained_models/datasets/ie_collections (see its README_数据集总索引.md):
32 NER sources (10 standalone corpora and 22 B2NERD_all subsets), DuIE, CMeIE, FinRE, SanWen,
DuEE, FewFC and ChFinAnn. decision/iedata/ turns each record into views:

    extraction/{task}_{source}.{split}.jsonl  span rows (decision/extraction.py), one file per source
    decision/{split}.cases.jsonl             CJ cases; `_meta.source_family` is "<task>:<source>"
    structured/{source}.{split}.jsonl        entities + relations / events, for end-to-end scoring
    manifest.json, README.md

The layout is the one scripts/train_multitask.py reads (`--data <out>`).

Splits. Official splits are kept. Where a test file has no labels, the labelled dev file becomes the
test set; where a corpus has no dev (or no split at all) one is carved from train by a stable hash of
the text, so reruns and every view of a text agree. Within a split, repeated texts keep one copy.

Leakage. Texts are compared in normalized form, whole and by sentences of 12+ characters. A dev
record overlapping any test record, and a train record overlapping any dev or test record, of any
source, is dropped. Overlap means the same text, a one-sentence held-out text inside the record, or
half of the record's sentences held out; sentences found in more than 20 records (boilerplate such
as announcement disclaimers) do not count. Cross-source duplicates inside train are kept: the same
sentence under two annotation standards teaches that type descriptions matter.

Sampling. Per source and split at most --max-train / --max-dev / --max-test records (overrides in
CAPS), chosen at random with --seed. Entity-typing decisions are further capped per source.

    python scripts/build_ie_data.py --out datasets/cj-ie
    python scripts/build_ie_data.py --out datasets/cj-ie-smoke --limit 200      # every source, quickly
"""
import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision.iedata import event, ner, relation  # noqa: E402
from decision.iedata.common import SPLITS, norm, pieces, stable_hash  # noqa: E402
from decision.schema import examples_for  # noqa: E402

# Per-source record caps that differ from the defaults.
CAPS = {"duie": {"train": 100000}, "chfinann": {"train": 8000, "dev": 300, "test": 1000}}
# Decision caps per task and split (records still give all their span rows).
DECISION_CAPS = {"ner_typing": {"train": 10000, "dev": 200, "test": 1000}}
TRAIN_ONLY = {"microbiota"}       # 99 sentences: too few for a held-out split
# PeopleDaily-NER's texts are all in MSRA: its dev/test stay as the benchmark used so far, its train adds nothing.
EVAL_ONLY = {"people_daily"}
COMMON_PIECE = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ie-root", default="G:/pretrained_models/datasets/ie_collections")
    parser.add_argument("--out", default="datasets/cj-ie")
    parser.add_argument("--sources", default="", help="comma-separated source names (default: all)")
    parser.add_argument("--max-train", type=int, default=50000, help="records per source")
    parser.add_argument("--max-dev", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--max-chars", type=int, default=300, help="longer NER texts are cut into windows")
    parser.add_argument("--max-subject-chars", type=int, default=40)
    parser.add_argument("--negative-pairs", type=float, default=0.5, help="unrelated pairs per related pair")
    parser.add_argument("--negative-event-rows", type=float, default=0.25, help="share of sentences with an absent-event row")
    parser.add_argument("--max-doc-chars", type=int, default=3000)
    parser.add_argument("--fields-per-event", type=int, default=3)
    parser.add_argument("--not-mentioned-rate", type=float, default=0.2, help="share of empty ChFinAnn fields asked")
    parser.add_argument("--limit", type=int, default=0, help="records per source and split, for smoke runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def all_sources(root):
    return ner.sources(root) + relation.sources(root) + event.sources(root)


def keys_of(text):
    """Full-text key and sentence keys of a record."""
    return stable_hash(norm(text)), {stable_hash(p) for p in pieces(text)}


def load(source, args, stats):
    """The source's records per split: de-duplicated, shuffled, capped (train keeps a margin for leakage drops)."""
    caps = {"train": args.max_train, "dev": args.max_dev, "test": args.max_test, **CAPS.get(source.name, {})}
    by_split, seen = defaultdict(list), defaultdict(set)
    for raw in source.records():
        if source.name in TRAIN_ONLY:
            raw.split = "train"
        if source.name in EVAL_ONLY and raw.split == "train":
            continue
        stats[f"{source.name}:{raw.split}:read"] += 1
        key = norm(raw.text)
        if not key or key in seen[raw.split]:
            stats[f"{source.name}:{raw.split}:duplicate"] += 1
            continue
        seen[raw.split].add(key)
        by_split[raw.split].append(raw)
    for split, raws in by_split.items():
        random.Random(f"{args.seed}:{source.name}:{split}").shuffle(raws)
        if args.limit:
            raws[args.limit:] = []
        margin = int(caps[split] * 1.2) + 100 if split == "train" else caps[split]
        raws[margin:] = []
    return by_split, caps


def leakage_filter(loaded, stats):
    """Drop dev records that overlap a test record, and train records that overlap a dev or test record.

    Overlap: the same text; a held-out text of one sentence found among the record's sentences; or at
    least half of the record's sentences found in held-out records. Sentences in more than COMMON_PIECE
    records (boilerplate) are ignored, so two announcements sharing a disclaimer do not overlap.
    """
    keyed, df = {}, Counter()
    for by_split, _ in loaded.values():
        for raws in by_split.values():
            for raw in raws:
                keyed[id(raw)] = keys_of(raw.text)
                df.update(keyed[id(raw)][1])
    common = {key for key, count in df.items() if count > COMMON_PIECE}
    stats["leakage:common_pieces"] = len(common)
    for split, against in (("dev", ("test",)), ("train", ("dev", "test"))):
        full, parts, single = set(), set(), set()
        for by_split, _ in loaded.values():
            for held in against:
                for raw in by_split.get(held, []):
                    f, p = keyed[id(raw)]
                    p = p - common
                    full.add(f)
                    parts |= p
                    if len(p) == 1:
                        single |= p
        for name, (by_split, _) in loaded.items():
            kept = []
            for raw in by_split.get(split, []):
                f, p = keyed[id(raw)]
                p = p - common
                if f in full or p & single or (p and 2 * len(p & parts) >= len(p)):
                    stats[f"{name}:{split}:leak_dropped"] += 1
                else:
                    kept.append(raw)
            by_split[split] = kept


class Writers:
    """Lazily opened JSONL files under `out`."""

    def __init__(self, out):
        self.out, self.files, self.counts = Path(out), {}, Counter()

    def write(self, relative, row):
        if relative not in self.files:
            path = self.out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files[relative] = open(path, "w", encoding="utf-8")
        self.files[relative].write(json.dumps(row, ensure_ascii=False) + "\n")
        self.counts[relative] += 1

    def close(self):
        for stream in self.files.values():
            stream.close()


def check_span_row(row):
    keys = {t["key"] for t in row["types"]}
    assert len(keys) == len(row["types"]), row["id"]
    for e in row["entities"]:
        assert 0 <= e["start"] < e["end"] <= len(row["text"]) and e["label"] in keys, (row["id"], e)


def merge_decisions(out, parts, counts, seed):
    """Interleave the per-source decision files at random (each is already in random order) into one per split."""
    rng = random.Random(f"{seed}:merge")
    for split in SPLITS:
        names = [name for name in parts if name.endswith(f".{split}.jsonl")]
        if not names:
            continue
        streams = [open(out / name, encoding="utf-8") for name in names]
        remaining = [counts[name] for name in names]
        with open(out / "decision" / f"{split}.cases.jsonl", "w", encoding="utf-8") as target:
            while any(remaining):
                i = rng.choices(range(len(streams)), weights=remaining)[0]
                target.write(streams[i].readline())
                remaining[i] -= 1
        for stream in streams:
            stream.close()
    shutil.rmtree(out / "decision" / "_parts")


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    stats, started = Counter(), time.time()
    sources = all_sources(args.ie_root)
    if args.sources:
        wanted = set(args.sources.split(","))
        sources = [s for s in sources if s.name in wanted]
        missing = wanted - {s.name for s in sources}
        if missing:
            raise SystemExit(f"unknown sources: {sorted(missing)}")

    loaded = {}
    for source in sources:
        loaded[source.name] = load(source, args, stats)
        sizes = {split: len(raws) for split, raws in loaded[source.name][0].items()}
        print(f"[{time.time() - started:6.0f}s] read {source.name}: {sizes}", flush=True)
    leakage_filter(loaded, stats)

    options = {"max_chars": args.max_chars, "max_subject_chars": args.max_subject_chars,
               "negative_pairs": args.negative_pairs, "negative_event_rows": args.negative_event_rows,
               "max_doc_chars": args.max_doc_chars, "fields_per_event": args.fields_per_event,
               "not_mentioned_rate": args.not_mentioned_rate, "stats": stats}
    writers, decision_counts, span_stats = Writers(out), Counter(), defaultdict(Counter)
    parts, held_out = set(), set()
    # Held-out splits first: a window of a long record can repeat a window of another record (shared
    # boilerplate sections), so span rows whose text is already in a later-held-out split are dropped.
    for split in ("test", "dev", "train"):
        seen_here = set()
        for source in sources:
            by_split, caps = loaded[source.name]
            raws = by_split.pop(split, [])[:caps[split]]
            stats[f"{source.name}:{split}:records"] = len(raws)
            rng = random.Random(f"{args.seed}:{source.name}:{split}:views")
            for raw in raws:
                views = source.views(raw, rng, options)
                for stem, row in views.spans:
                    check_span_row(row)
                    key = norm(row["text"])
                    if key in held_out:
                        stats[f"{source.name}:{split}:span_rows_held_out_text"] += 1
                        continue
                    seen_here.add(key)
                    writers.write(f"extraction/{stem}.{split}.jsonl", row)
                    span_stats[f"{stem}.{split}"]["rows"] += 1
                    span_stats[f"{stem}.{split}"]["entities"] += len(row["entities"])
                for case in views.decisions:
                    family = case["_meta"]["source_family"]
                    cap = DECISION_CAPS.get(case["_meta"]["task_id"], {}).get(split)
                    if cap is not None and decision_counts[(family, split)] >= cap:
                        continue
                    list(examples_for(case, "general"))          # raises on any schema error
                    name = f"decision/_parts/{source.name}.{split}.jsonl"
                    writers.write(name, case)
                    parts.add(name)
                    decision_counts[(family, split)] += 1
                if views.structured is not None:
                    writers.write(f"structured/{source.name}.{split}.jsonl", views.structured)
        held_out |= seen_here
        print(f"[{time.time() - started:6.0f}s] wrote {split}", flush=True)
    writers.close()
    merge_decisions(out, parts, writers.counts, args.seed)

    manifest = {
        "name": "CJ-IE", "seed": args.seed, "arguments": vars(args), "caps": CAPS, "decision_caps": DECISION_CAPS,
        "sources": {s.name: {"dataset": s.dataset, "quality": s.quality, "domain": s.domain, "license": s.license,
                             **({"types": [[t["key"], t["name"]] for t in s.types]} if hasattr(s, "types") else {})}
                    for s in sources},
        "extraction": {name: dict(counts) for name, counts in sorted(span_stats.items())},
        "decision": {f"{family}.{split}": n for (family, split), n in sorted(decision_counts.items())},
        "files": {name: n for name, n in sorted(writers.counts.items()) if not name.startswith("decision/_parts")},
        "stats": dict(sorted(stats.items())),
    }
    for split in SPLITS:
        manifest["files"][f"decision/{split}.cases.jsonl"] = sum(
            n for (_, s), n in decision_counts.items() if s == split)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(readme(manifest), encoding="utf-8")
    totals = {split: {"span_rows": sum(c["rows"] for k, c in span_stats.items() if k.endswith("." + split)),
                      "decisions": manifest["files"][f"decision/{split}.cases.jsonl"]} for split in SPLITS}
    print(json.dumps(totals, ensure_ascii=False), f"{time.time() - started:.0f}s")


def readme(manifest):
    """Counts tables for the generated README; the design is documented in docs/IE_DATASET.md."""
    lines = ["# CJ-IE（中文信息抽取决策数据）", "",
             "由 `scripts/build_ie_data.py` 生成，设计与格式见仓库 `docs/IE_DATASET.md`。各数字来自本次构建的 `manifest.json`。", "",
             "## 跨度抽取（span head）", "", "| 文件 | train 行 / 实体 | dev 行 / 实体 | test 行 / 实体 |", "|---|---:|---:|---:|"]
    stems = sorted({name.rsplit(".", 1)[0] for name in manifest["extraction"]})
    for stem in stems:
        cells = []
        for split in SPLITS:
            c = manifest["extraction"].get(f"{stem}.{split}", {})
            cells.append(f"{c.get('rows', 0):,} / {c.get('entities', 0):,}" if c else "—")
        lines.append(f"| `{stem}` | " + " | ".join(cells) + " |")
    lines += ["", "## 决策（decision head）", "", "| source_family | train | dev | test |", "|---|---:|---:|---:|"]
    families = sorted({name.rsplit(".", 1)[0] for name in manifest["decision"]})
    for family in families:
        cells = [f"{manifest['decision'].get(f'{family}.{split}', 0):,}" for split in SPLITS]
        lines.append(f"| `{family}` | " + " | ".join(cells) + " |")
    lines += ["", "## 来源", "", "| source | 数据目录 | 质量 | 领域 | 许可 |", "|---|---|---|---|---|"]
    for name, info in manifest["sources"].items():
        lines.append(f"| `{name}` | {info['dataset']} | {info['quality']} | {info['domain']} | {info['license'] or '见来源目录'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
