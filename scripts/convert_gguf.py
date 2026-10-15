"""Export the merged decision model with a local llama.cpp checkout (CPU only)."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "models/chinese-decision-0.8b")
    parser.add_argument("--out", type=Path, default=ROOT / "models/chinese-decision-0.8b-gguf")
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=["bf16", "q8_0"], default=["bf16", "q8_0"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    base, out, checkout = args.base.resolve(), args.out.resolve(), args.llama_cpp.resolve()
    converter = checkout / "convert_hf_to_gguf.py"
    if not converter.is_file() or not (base / "model.safetensors").is_file():
        parser.error("llama.cpp converter or source model.safetensors is missing")
    if base == out:
        parser.error("output must be separate from the source model")
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for fmt in dict.fromkeys(args.formats):
        path = out / f"chinese-decision-0.8b-{fmt.upper()}.gguf"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path}; use a new output directory or --overwrite")
        command = [sys.executable, str(converter), str(base), "--outfile", str(path),
                   "--outtype", fmt, "--no-mtp"]
        subprocess.run(command, check=True)
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError(f"invalid GGUF header: {path}")
        files.append({"file": path.name, "bytes": path.stat().st_size,
                      "sha256": digest(path), "command": command})
    # The export can be served as --base without retaining the source weights; the training summary
    # and validation report let the server describe its checkpoint (/v1/info, /v1/training).
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json", "added_tokens.json", "decision_config.json", "LICENSE",
                 "training_summary.json", "validation_report.json"):
        source = base / name
        if source.is_file():
            shutil.copy2(source, out / name)
    revision = None
    if (checkout / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    report = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "source": str(base), "source_sha256": digest(base / "model.safetensors"),
              "llama_cpp": str(checkout), "llama_cpp_revision": revision,
              "converter_sha256": digest(converter),
              "no_mtp": True, "files": files}
    (out / "conversion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
