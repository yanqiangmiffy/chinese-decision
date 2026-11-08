"""Shuffle the line order of gzip JSONL training files (seeded per file), in place or into a copy.

Training reads every prepared stratum in file order through a small buffer, so a file sorted by label
trains one label at the end of the run (decision-mix-v1: RAGTruth, ESCI, ...). New mixes are shuffled when
built; this script fixes files written before that:

    python scripts/shuffle_corpus.py --files datasets/decision-mix-v2/raw/_staging/*.train.jsonl.gz
    python scripts/shuffle_corpus.py --copy datasets/CJ-mixed datasets/CJ-mixed-shuffled

`--copy` reads a prepared corpus (manifest.json + strata) and writes the same strata with shuffled lines
plus the manifest; line counts do not change, so `--domain-fraction` (which keeps the first fraction of a
stratum) then keeps a random subset instead of the head of the file.
"""
import argparse
import gzip
import json
import random
import shutil
from pathlib import Path


def shuffle_file(source, target, seed):
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        lines = [line if line.endswith("\n") else line + "\n" for line in stream if line.strip()]
    random.Random(seed).shuffle(lines)
    partial = Path(str(target) + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8", compresslevel=6) as stream:
        stream.writelines(lines)
    partial.replace(target)
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", nargs="+", help="shuffle these files in place")
    group.add_argument("--copy", nargs=2, metavar=("SRC", "DST"), help="shuffled copy of a prepared corpus")
    parser.add_argument("--seed", default="order-v2")
    args = parser.parse_args()
    if args.files:
        for path in args.files:
            n = shuffle_file(path, path, f"{args.seed}:{Path(path).name}")
            print(json.dumps({"event": "shuffled", "path": path, "lines": n}), flush=True)
        return
    source, target = map(Path, args.copy)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    for entry in manifest["strata"]:
        out = target / entry["path"]
        if out.exists():                                   # resumable
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        n = shuffle_file(source / entry["path"], out, f"{args.seed}:{entry['path']}")
        print(json.dumps({"event": "shuffled", "path": entry["path"], "lines": n}), flush=True)
    for extra in source.iterdir():                          # manifest and per-domain markers
        if extra.is_file():
            shutil.copy2(extra, target / extra.name)
    for domain_dir in (p for p in source.iterdir() if p.is_dir()):
        for marker in domain_dir.glob("*.json"):
            shutil.copy2(marker, target / domain_dir.name / marker.name)
    manifest["shuffled_from"] = str(source)
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "copied", "target": str(target), "strata": len(manifest["strata"])}), flush=True)


if __name__ == "__main__":
    main()
