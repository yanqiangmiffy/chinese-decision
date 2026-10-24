"""Configuration of a decision model beyond its backbone, stored as `decision_config.json`.

Released decision-only models (chinese-decision-0.8b/2b/4b) carry the decision settings; unified
models add an `extraction` section for the span head. Unknown keys are kept in `metadata`, so a
file round-trips unchanged.
"""
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from .utils import resolve_pretrained

CONFIG_NAME = "decision_config.json"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT_VERSION = "hmm-zh-v1"
# Longer inputs are rejected everywhere; context and questions are never truncated.
MAX_LENGTH = 16384


@dataclass
class ExtractionConfig:
    """Span head shape and the settings its predictions need."""

    hidden_size: int
    max_span_units: int = 20
    inner: int = 512
    out: int = 256
    width_dim: int = 32
    dropout: float = 0.1
    none_shift: float = 0.0          # log(1 / none_weight) added to the none utility at inference
    echo: bool = True                # spans are read from a one-unit-per-character copy of the text

    HEAD_FIELDS = ("hidden_size", "max_span_units", "inner", "out", "width_dim", "dropout")

    def head_kwargs(self):
        return {name: getattr(self, name) for name in self.HEAD_FIELDS}

    def to_dict(self):
        return {"head": self.head_kwargs(), "none_shift": self.none_shift, "echo": self.echo}

    @classmethod
    def from_dict(cls, data):
        return cls(**data["head"], none_shift=float(data.get("none_shift", 0.0)), echo=bool(data.get("echo", True)))


@dataclass
class DecisionConfig:
    """Decision readout settings, plus the span head's when the model also extracts."""

    temperature: float = 1.0
    prompt_version: str = PROMPT_VERSION
    max_length: int = MAX_LENGTH
    max_options: int = len(LETTERS)
    model_name: str = None
    extraction: ExtractionConfig = None
    metadata: dict = field(default_factory=dict)

    @property
    def can_extract(self):
        return self.extraction is not None

    def to_dict(self):
        data = {"model_name": self.model_name, "temperature": self.temperature, "prompt_version": self.prompt_version,
                "max_length": self.max_length, "max_options": self.max_options, **self.metadata}
        if self.model_name is None:
            del data["model_name"]
        if self.extraction is not None:
            data["extraction"] = self.extraction.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        known = {f.name for f in fields(cls)} - {"metadata", "extraction"}
        values = {name: data.pop(name) for name in list(data) if name in known}
        extraction = data.pop("extraction", None)
        return cls(**values, extraction=ExtractionConfig.from_dict(extraction) if extraction else None,
                   metadata=data)

    @classmethod
    def from_pretrained(cls, name_or_path, revision=None):
        """The model's config; a directory without `decision_config.json` (a raw base model) gets the defaults."""
        path = Path(resolve_pretrained(name_or_path, revision)) / CONFIG_NAME
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_NAME).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
