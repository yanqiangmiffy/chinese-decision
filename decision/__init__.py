"""Chinese Decision: calibrated decisions and typed span extraction from one model.

    from decision import DecisionPipeline
    pipe = DecisionPipeline.from_pretrained("models/chinese-decision-0.8b")

Public classes load lazily, so `import decision` (and the GGUF serving path) works without PyTorch.

    DecisionConfig, ExtractionConfig   decision_config.json             decision.configuration
    DecisionProcessor                  prompts, encoding, decoding      decision.processing
    DecisionModel, SpanHead            backbone + span head             decision.modeling
    DecisionPipeline, SlotFillingPipeline  request-level inference      decision.pipeline
    RLCDLoss, RLCDLossConfig, SpanLoss training losses                  decision.losses
"""
import importlib

__version__ = "0.2.0"

_EXPORTS = {
    "DecisionConfig": "configuration", "ExtractionConfig": "configuration",
    "DecisionProcessor": "processing",
    "DecisionModel": "modeling", "SpanHead": "modeling",
    "DecisionPipeline": "pipeline", "SlotFillingPipeline": "pipeline",
    "RLCDLoss": "losses", "RLCDLossConfig": "losses", "SpanLoss": "losses",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
