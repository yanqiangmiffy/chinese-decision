"""One model, one weight file, two readouts: decisions and span extraction.

A unified model directory is a standard causal-LM export (`AutoModelForCausalLM` loads it) whose
safetensors also hold the span head under `span_head.*`; `decision_config.json` carries both heads'
settings. Training keeps a LoRA adapter and the head apart; `save_unified` merges the adapter
into the backbone and writes everything once:

    model.safetensors          backbone (LoRA merged) + span_head.* tensors
    decision_config.json       {"temperature", "prompt_version", "max_length", "max_options",
                                "extraction": {"head": {...}, "none_shift", "echo"}}
    tokenizer files, training_summary.json

Both readouts share one forward pass type and one set of weights:
- decision: the prompt lists candidates as letters; the option distribution is read from the LM
  head's letter logits at the last position;
- extraction: the prompt lists entity types and echoes the text one unit per character; the span
  head scores every candidate span against [none, types] from the last hidden states.

Which readout answers a request is fixed by the request, not guessed from outputs: a decision
request carries questions with candidates, an extraction request carries types (`predict`).
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .extraction import SpanExtractor, SpanHead
from .inference import candidate_probabilities
from .model import forward_batch, letter_tokens, load_model, load_tokenizer
from .schema import MAX_LENGTH, compile_prompt, options_for

SPAN_PREFIX = "span_head."


def save_unified(model, head, tokenizer, out, decision_config, extraction, summary=None):
    """Write `model` (a plain causal LM, adapters already merged) and `head` as one model directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(model.state_dict())
    state.update({SPAN_PREFIX + key: value.detach().contiguous() for key, value in head.state_dict().items()})
    model.save_pretrained(out, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(out)
    config = dict(decision_config, extraction=dict(extraction, head=head.config))
    (out / "decision_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary is not None:
        (out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def read_config(path):
    file = Path(path) / "decision_config.json"
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def load_span_head(path, device="cuda"):
    """The span head stored in a unified model directory and its settings; (None, {}) for a decision-only model."""
    extraction = read_config(path).get("extraction")
    if not extraction:
        return None, {}
    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith(SPAN_PREFIX):
                    tensors[key[len(SPAN_PREFIX):]] = stream.get_tensor(key)
    if not tensors:
        raise ValueError(f"{path}: decision_config.json describes a span head but no span_head.* tensors were found")
    head = SpanHead(**extraction["head"])
    head.load_state_dict(tensors)
    return head.to(device).eval(), extraction


class UnifiedModel:
    """Loads a unified directory once and answers decision and extraction requests from the same weights.

        model = UnifiedModel("models/chinese-decision-ie-0.8b")
        model.predict({"state": "...", "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {...}}}})
        model.predict({"text": "...", "types": {"人名": "人物的姓名", "公司": "企业名称"}})
    """

    def __init__(self, path, device="cuda"):
        self.path, self.device = Path(path), device
        self.config = read_config(path)
        self.temperature = float(self.config.get("temperature", 1.0))
        self.tokenizer = load_tokenizer(str(path))
        self.letters = letter_tokens(self.tokenizer, device)
        self.model = load_model(str(path), device=device).eval()
        head, extraction = load_span_head(path, device)
        self.extractor = (SpanExtractor(self.model, head, self.tokenizer, extraction["none_shift"], extraction["echo"])
                          if head is not None else None)

    @torch.inference_mode()
    def decide(self, state, questions):
        """`{name: {candidate key: probability}}` for CJ-format questions about one state."""
        result = {}
        for name, question in questions.items():
            options = options_for(question)
            prompt = compile_prompt(self.tokenizer, state, question["instructions"], options)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > int(self.config.get("max_length", MAX_LENGTH)):
                raise ValueError(f"question {name!r} is longer than the model's max_length")
            logits, _ = forward_batch(self.model, self.tokenizer, self.letters,
                                      [{"input_ids": ids, "target": [0.0] * len(options)}])
            probabilities = candidate_probabilities(logits[0, :len(options)].tolist(), self.temperature)
            result[name] = {key: p for (key, _), p in zip(options, probabilities)}
        return result

    def extract(self, text, types, threshold=0.0):
        """Typed character spans; `types` is a list of names or a {name: description} mapping."""
        if self.extractor is None:
            raise ValueError(f"{self.path} has no span head; it answers decisions only")
        return self.extractor.extract(text, types, threshold)

    def predict(self, request):
        """Route by request shape: `questions` (or one `question`) -> decide, `types` -> extract."""
        if "types" in request and ("questions" in request or "question" in request):
            raise ValueError("a request asks for either decisions (questions) or extraction (types), not both")
        if "types" in request:
            return {"task": "extraction", "entities": self.extract(request["text"], request["types"],
                                                                   request.get("threshold", 0.0))}
        if "questions" in request or "question" in request:
            questions = request.get("questions") or {"q": request["question"]}
            return {"task": "decision", "answers": self.decide(request["state"], questions)}
        raise ValueError("request needs `questions`/`question` (decision) or `types` (extraction)")
