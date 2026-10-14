"""Copy a run's validation-selected best adapter into a fresh directory for evaluation.

Training keeps replacing `best/`; evaluating a frozen copy keeps reports tied to one checkpoint.
"""
import argparse
import json
import shutil
from pathlib import Path

from decision.utils import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-run", default="runs/cj-ddp")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.source_run), Path(args.out)
    if not (source / "best/selection.json").exists():
        raise ValueError("source run has no validation-selected best checkpoint")
    selection = json.loads((source / "best/selection.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "best", out / "adapter")
    shutil.copyfile(source / "config.json", out / "training_config.json")
    copied = json.loads((out / "adapter/selection.json").read_text())
    current = json.loads((source / "best/selection.json").read_text())
    digest = sha256(out / "adapter/adapter_model.safetensors")
    if selection != copied or copied != current or digest != sha256(source / "best/adapter_model.safetensors"):
        raise ValueError("best changed during copy; retry with a new output directory")
    setup = {"run": str(out), "source_run": str(source), "selection": selection, "adapter_sha256": digest}
    (out / "setup.json").write_text(json.dumps(setup, indent=2))
    print(json.dumps(setup))


if __name__ == "__main__":
    main()
