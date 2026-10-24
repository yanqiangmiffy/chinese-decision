"""Intent + slot records as training data for both heads.

A record is a query, its intent, and the slots of that intent as verbatim substrings of the query:

    {"query": "打开空调", "intent": "device_control", "slots": {"function": "打开", "device": "空调"}}

It becomes an intent decision (decision head) and, when the intent has slots, a span row over the
intent's slot types (span head). Inference is decision.pipeline.SlotFillingPipeline.
"""
import json
from pathlib import Path

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
