"""High-level inference: load a model directory once, answer decision and extraction requests.

Which readout answers a request is fixed by the request, not guessed from outputs: a decision
request carries questions with candidates, an extraction request carries entity types, and a
request that carries both gets both answers about the same input. Any input
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
from .processing import PAD_MULTIPLE, DecisionProcessor, Example
from .serving.backends import candidate_probabilities


def entities(text, found):
    """Decoded `(start, end, type, probability)` spans as response entities, in text order."""
    return [{"text": text[s:e], "label": label, "start": s, "end": e, "score": round(p, 4)}
            for s, e, label, p in sorted(found)]


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
        return entities(text, found)

    @torch.inference_mode()
    def decide_and_extract(self, state, questions, text, types, threshold=0.0, joint_tokens=3072, token_budget=32768):
        """`(decide(state, questions), extract(text, types, threshold))`, in one forward pass when it is short.

        The span row and as many questions as fit in `joint_tokens` padded tokens share one pass; further
        questions are scored in batches of their own. One pass saves a pass's fixed cost but pads every
        decision to the span row's length (the text twice, with the echo), so past `joint_tokens` the
        questions run batched in one pass and the span row in another. On an A800 with the 2B model a
        combined request takes ~50 ms instead of ~97 ms at 100 characters, and breaks even near 1,200.
        """
        if not self.can_extract:
            raise ValueError("this model has no span head; it answers decisions only")
        items = [self.processor.encode_question(state, question) for question in questions.values()]
        row = self.processor.encode_extraction(self.processor.extraction_row(text, types))
        if row is None:
            raise ValueError("input longer than the extraction encoder's max length")
        width = -(-max(len(item["input_ids"]) for item in [row, *items]) // PAD_MULTIPLE) * PAD_MULTIPLE
        joint = joint_tokens // width - 1
        if joint >= 1:
            logits, utilities = self.model.joint_readouts(items[:joint], [row], self.processor, self.letters)
            logits = logits.cpu().tolist() + self.candidate_logits(items[joint:], token_budget=token_budget)
            extraction = self.model.config.extraction
            found = self.processor.decode_spans(row, utilities, extraction.none_shift if extraction else 0.0,
                                                threshold, decision_shift=extraction.decision_shift if extraction else None)
        else:
            logits = self.candidate_logits(items, token_budget=token_budget)
            found, = self.model.predict_spans([row], self.processor, threshold)
        probabilities = {name: dict(zip((key for key, _ in item["options"]),
                                        candidate_probabilities(line[:len(item["options"])], self.temperature)))
                         for name, item, line in zip(questions, items, logits, strict=True)}
        return probabilities, entities(text, found)

    def __call__(self, request):
        """Route by request shape: `questions` (or one `question`) -> decide, `types` -> extract, both -> both.

        A request with both reads one input twice: `text` defaults to `state` and `state` to `text`.
        """
        decision, extraction = "questions" in request or "question" in request, "types" in request
        if not (decision or extraction):
            raise ValueError("request needs `questions`/`question` (decision), `types` (extraction) or both")
        state, text = request.get("state", request.get("text")), request.get("text", request.get("state"))
        threshold = request.get("threshold", 0.0)
        if decision:
            questions = request.get("questions") or ({"q": request["question"]} if "question" in request else None)
            if not questions:
                raise ValueError("`questions` must be a nonempty object")
        if decision and extraction:
            answers, found = self.decide_and_extract(state, questions, text, request["types"], threshold)
            return {"task": "decision+extraction", "answers": answers, "entities": found}
        if decision:
            return {"task": "decision", "answers": self.decide(state, questions)}
        return {"task": "extraction", "entities": self.extract(text, request["types"], threshold)}

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
