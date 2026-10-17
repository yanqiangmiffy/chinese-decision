"""Small I/O helpers shared by the command-line scripts."""
import hashlib
import json
import time
from pathlib import Path


def model_name_for(base, config=None):
    """Read an export's identity, with compatible names for Qwen3.5 adapters."""
    if config and config.get("model_name"):
        return config["model_name"]
    name = Path(base).name
    if name.startswith("Qwen3.5-"):
        return "chinese-decision-" + name.removeprefix("Qwen3.5-").lower()
    if name.startswith("chinese-decision-"):
        return name.removesuffix("-gguf")
    return "chinese-decision-0.8b"


def log(event, **fields):
    """Print one structured JSON log line."""
    print(json.dumps({"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}, ensure_ascii=False),
          flush=True)


def write_json(path, value):
    """Replace a JSON file atomically, so readers never see a partial write."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
