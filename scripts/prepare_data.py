"""Verify CJ-Dataset against its MANIFEST, then re-shard training data by domain/source/type.

Ordered gzip files would otherwise feed whole task families to training in blocks.
Every training line is kept once; validation files are never included.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from decision.data import prepare_domain
from decision.utils import log, sha256


def verify_files(data, manifest):
    for domain, spec in manifest["domains"].items():
        for split, entry in spec["files"].items():
            path = data / entry["path"]
            if not path.exists() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"missing or corrupt dataset file: {path}")
            log("verified", domain=domain, split=split, bytes=entry["bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="datasets/CJ-Dataset")
    parser.add_argument("--out", default="datasets/CJ-mixed")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data = Path(args.data)
    manifest = json.loads((data / "MANIFEST.json").read_text())
    verify_files(data, manifest)
    specs = [(args.data, args.out, domain, entry["files"]["train"]["decisions"]) for domain, entry in manifest["domains"].items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(prepare_domain, specs))
    output = {"source_revision": manifest["source_revision"], "decisions": sum(r["decisions"] for r in results),
              "strata": [entry for r in results for entry in r["strata"]], "validation_included": False}
    Path(args.out, "manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    log("all_prepared", decisions=output["decisions"], strata=len(output["strata"]))


if __name__ == "__main__":
    main()
