"""High-level inference: load a model directory once, answer decision and extraction requests.

Which readout answers a request is fixed by the request, not guessed from outputs: a decision
request carries questions with candidates, an extraction request carries entity types. Any input
could be pushed through both readouts, but the other readout's numbers mean nothing for it and
cannot be compared. Cascades read one answer to choose the next request, as
`SlotFillingPipeline` does: the intent decides which slot types to extract.

    pipe = DecisionPipeline.from_pretrained("models/chinese-decision-ie-0.8b")
    pipe({"state": "...", "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {...}}}})
    pipe({"text": "...", "types": {"人名": "人物的姓名", "公司": "企业名称"}})
"""
from pathlib import Path

import torch

from .data.batching import shape_batches
from .data.slots import STATE_KEY
from .modeling import DecisionModel
from .processing import DecisionProcessor, Example
from .serving.backends import candidate_probabilities


class DecisionPipeline:
    """A `DecisionModel` and its `DecisionProcessor`, with request-level methods."""

    def __init__(self, model, processor):
        self.model, self.processor = model.eval(), processor
        self.letters = model.letter_tensor(processor)

    @classmethod
    def from_pretrained(cls, name_or_path, device="cuda", revision=None, **kwargs):
        model = DecisionModel.from_pretrained(name_or_path, device=device, revision=revision, **kwargs)
        return cls(model, DecisionProcessor.from_pretrained(name_or_path, revision=revision))

    @classmethod
    def from_run(cls, base, run, device="cuda"):
        """A training run's `adapter/` and `span_head.pt` on top of `base` (before export to one directory)."""
        model = DecisionModel.from_pretrained(base, adapter=Path(run) / "adapter", device=device)
        model.load_span_head(Path(run) / "span_head.pt")
        return cls(model, DecisionProcessor(DecisionProcessor.from_pretrained(base).tokenizer, model.config))

    @property
    def temperature(self):
        return float(self.model.config.temperature)

    @property
    def can_extract(self):
        return self.model.span_head is not None

    @torch.inference_mode()
    def decision_probabilities(self, items):
        """Calibrated candidate probabilities for encoded decisions (one list per item)."""
        logits, _ = self.model.score_decisions(items, self.processor, self.letters)
        return [candidate_probabilities(row[:len(item["target"])].tolist(), self.temperature)
                for item, row in zip(items, logits)]

    @torch.inference_mode()
    def candidate_logits(self, items, batch_size=128, token_budget=32768):
        """Raw (uncalibrated) candidate logits for encoded decisions, one list per item, in input order.

        Items are scored in length-sorted batches of few distinct shapes, the loop of
        scripts/evaluate_cj_bench.py, so a server and the benchmark read the same numbers.
        """
        rows = sorted((dict(item, index=i) for i, item in enumerate(items)), key=lambda row: len(row["input_ids"]))
        logits = [None] * len(items)
        for batch in shape_batches(rows, batch_size, token_budget):
            values, _ = self.model.score_decisions(batch, self.processor, self.letters)
            if not torch.isfinite(values).all().item():
                raise FloatingPointError("non-finite decision logits")
            for row, line in zip(batch, values.float().cpu().tolist()):
                logits[row["index"]] = line[:len(row["options"])]
        return logits

    def decide(self, state, questions):
        """`{name: {candidate key: probability}}` for CJ-format questions about one state."""
        result = {}
        for name, question in questions.items():
            item = self.processor.encode_question(state, question)
            probabilities, = self.decision_probabilities([item])
            result[name] = {key: p for (key, _), p in zip(item["options"], probabilities)}
        return result

    def predict_spans(self, rows, threshold=0.0):
        """Encoded rows and their `(start, end, type, probability)` predictions."""
        if not self.can_extract:
            raise ValueError("this model has no span head; it answers decisions only")
        items = [self.processor.encode_extraction(row) for row in rows]
        if any(item is None for item in items):
            raise ValueError("input longer than the extraction encoder's max length")
        return items, self.model.predict_spans(items, self.processor, threshold)

    def extract(self, text, types, threshold=0.0):
        """Typed character spans; `types` is a list of names or a `{name: description}` mapping."""
        _, (found,) = self.predict_spans([self.processor.extraction_row(text, types)], threshold)
        return [{"text": text[s:e], "label": label, "start": s, "end": e, "score": round(p, 4)}
                for s, e, label, p in sorted(found)]

    def __call__(self, request):
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

    predict = __call__


class SlotFillingPipeline:
    """Intent by the decision readout, then the slots of that intent by the span readout.

    `schema` is `{"instructions", "intents": {intent: {"description", "slots": {key: {"name", "description"}}}}}`
    (see examples/slot_filling/schema.json).
    """

    def __init__(self, pipeline, schema):
        self.pipeline, self.schema = pipeline, schema

    def intents(self, query):
        """`[(probability, intent)]`, most probable first (softmax at temperature 1, as trained)."""
        options = [(intent, spec["description"]) for intent, spec in self.schema["intents"].items()]
        example = Example({STATE_KEY: query}, self.schema["instructions"], options,
                          [1 / len(options)] * len(options), "choice", "general", "slot_intent", "q")
        item = self.pipeline.processor.encode_example(example)
        with torch.inference_mode():
            logits, _ = self.pipeline.model.score_decisions([item], self.pipeline.processor, self.pipeline.letters)
        probabilities = logits[0, :len(options)].float().softmax(-1).tolist()
        return sorted(zip(probabilities, (intent for intent, _ in options)), reverse=True)

    def __call__(self, query, threshold=0.0):
        """`{"intent", "intent_probability", "slots": {key: value}, "spans": [...]}`.

        A slot found more than once keeps its most probable span.
        """
        probability, intent = self.intents(query)[0]
        spans = []
        types = [{"key": key, "name": slot["name"], "description": slot["description"]}
                 for key, slot in self.schema["intents"][intent]["slots"].items()]
        if types:
            _, (found,) = self.pipeline.predict_spans([{"text": query, "types": types}], threshold)
            spans = sorted(found, key=lambda span: -span[3])
        slots = {}
        for start, end, key, _ in spans:
            slots.setdefault(key, query[start:end])
        return {"intent": intent, "intent_probability": probability, "slots": slots,
                "spans": [{"key": k, "value": query[s:e], "start": s, "end": e, "score": round(p, 4)}
                          for s, e, k, p in spans]}
