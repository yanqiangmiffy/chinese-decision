"""Intent + slot filling with the two heads: query -> {intent: {slot: value}}.

The decision head picks the intent among the schema's intents; the span head then extracts the
slots of that intent, each slot being a typed span of the query. Records look like

    {"query": "打开空调", "intent": "device_control", "slots": {"function": "打开", "device": "空调"}}

and slot values must be verbatim substrings of the query.
"""
import json
from pathlib import Path

import torch

from .data import encode
from .model import forward_batch, letter_tokens
from .schema import Example

STATE_KEY = "用户话语"


def load_schema(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def align(query, slots):
    """Character-offset entities for `slots`, or None when a value is missing or cannot avoid overlaps.

    Longer values are placed first, each at its first occurrence that overlaps nothing placed so far.
    """
    taken, entities = set(), []
    for key, value in sorted(slots.items(), key=lambda kv: -len(kv[1])):
        if not isinstance(value, str) or not value or len(value) > len(query):
            return None
        start = query.find(value)
        while start >= 0 and taken & set(range(start, start + len(value))):
            start = query.find(value, start + 1)
        if start < 0:
            return None
        taken.update(range(start, start + len(value)))
        entities.append({"start": start, "end": start + len(value), "label": key})
    return sorted(entities, key=lambda e: e["start"])


def slot_types(schema, intent):
    return [{"key": key, "name": slot["name"], "description": slot["description"]}
            for key, slot in schema["intents"][intent]["slots"].items()]


def decision_case(record, schema, split, source="slot_intent"):
    criteria = {intent: spec["description"] for intent, spec in schema["intents"].items()}
    return {"id": f"{source}-{split}-{record['query']}", "state": {STATE_KEY: record["query"]},
            "questions": {"q": {"type": "choice", "instructions": schema["instructions"], "criteria": criteria}},
            "gold": {"q": record["intent"]}, "_meta": {"source_family": source, "split": split}}


def extraction_row(record, schema, split, source="slots"):
    """Span-extraction row over the record's intent slots; None for intents without slots."""
    types = slot_types(schema, record["intent"])
    if not types:
        return None
    entities = align(record["query"], record["slots"])
    if entities is None:
        raise ValueError(f"slots do not align with the query: {record}")
    return {"id": f"{source}-{split}-{record['query']}", "source": source, "text": record["query"],
            "types": types, "entities": entities}


class SlotFiller:
    """End-to-end inference: intent by the decision head, slots of that intent by the span head."""

    def __init__(self, extractor, schema):
        self.extractor, self.schema = extractor, schema
        self.device = next(extractor.head.parameters()).device
        self.letters = letter_tokens(extractor.tokenizer, self.device)

    @torch.no_grad()
    def intents(self, query):
        """`[(probability, intent)]`, most probable first."""
        options = [(intent, spec["description"]) for intent, spec in self.schema["intents"].items()]
        example = Example({STATE_KEY: query}, self.schema["instructions"], options,
                          [1 / len(options)] * len(options), "choice", "general", "slot_intent", "q")
        item = encode(example, self.extractor.tokenizer, 16384)
        logits, _ = forward_batch(self.extractor.model, self.extractor.tokenizer, self.letters, [item])
        probs = logits[0, :len(options)].float().softmax(-1).tolist()
        return sorted(zip(probs, (intent for intent, _ in options)), reverse=True)

    def __call__(self, query, threshold=0.0):
        """`{"intent", "intent_probability", "slots": {key: value}, "spans": [...]}`.

        A slot found more than once keeps its most probable span.
        """
        probability, intent = self.intents(query)[0]
        spans = []
        types = slot_types(self.schema, intent)
        if types:
            row = {"text": query, "types": types}
            _, (found,) = self.extractor.predict([row], threshold)
            spans = sorted(found, key=lambda span: -span[3])
        slots = {}
        for start, end, key, _ in spans:
            slots.setdefault(key, query[start:end])
        return {"intent": intent, "intent_probability": probability, "slots": slots,
                "spans": [{"key": k, "value": query[s:e], "start": s, "end": e, "score": round(p, 4)}
                          for s, e, k, p in spans]}
