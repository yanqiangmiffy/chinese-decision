"""Export the merged decision model with a local llama.cpp checkout (CPU only)."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from decision.utils import model_name_for

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkout_revision(checkout):
    """Source distributions may keep a .git pointer without the parent repository."""
    if not (checkout / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def conversion_report(base, checkout, files, model_name, decision, source_hash):
    return {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "source": str(base), "source_sha256": source_hash, "model_name": model_name,
            "temperature": decision.get("temperature", 1.0),
            "llama_cpp": str(checkout), "llama_cpp_revision": checkout_revision(checkout),
            "converter_sha256": digest(checkout / "convert_hf_to_gguf.py"),
            "no_mtp": True, "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "models/chinese-decision-0.8b")
    parser.add_argument("--out", type=Path, help="defaults to <base directory>/<model name>-gguf")
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=["bf16", "q8_0"], default=["bf16", "q8_0"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    base, checkout = args.base.resolve(), args.llama_cpp.resolve()
    decision_path = base / "decision_config.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    model_name = model_name_for(base, decision)
    if "/" in model_name or "\\" in model_name or model_name in (".", ".."):
        parser.error("model_name must be a single filename component")
    out = (args.out or base.parent / (model_name + "-gguf")).resolve()
    converter = checkout / "convert_hf_to_gguf.py"
    if not converter.is_file() or not (base / "model.safetensors").is_file():
        parser.error("llama.cpp converter or source model.safetensors is missing")
    if base == out:
        parser.error("output must be separate from the source model")
    outputs = [(fmt, out / f"{model_name}-{fmt.upper()}.gguf") for fmt in dict.fromkeys(args.formats)]
    for _, path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path}; use a new output directory or --overwrite")
    source_hash = digest(base / "model.safetensors")
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for fmt, path in outputs:
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
                 "training_summary.json", "training_config.json", "validation_report.json", "THIRD_PARTY_NOTICES.md"):
        source = base / name
        if source.is_file():
            shutil.copy2(source, out / name)
    if (base / "LICENSES").is_dir():
        shutil.copytree(base / "LICENSES", out / "LICENSES", dirs_exist_ok=True)
    if digest(base / "model.safetensors") != source_hash:
        raise ValueError("source weights changed during conversion")
    report = conversion_report(base, checkout, files, model_name, decision, source_hash)
    (out / "conversion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
