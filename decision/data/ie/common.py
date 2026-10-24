"""Shared pieces of the Chinese IE set built by scripts/build_ie_data.py.

A source yields `Raw` records: one sentence, paragraph or document with its split and its parsed
annotation. The builder samples, de-duplicates and checks records for leakage before their views
are made, so every view of a record (span rows, decision cases, the structured record) lands in
the record's split. Views use two formats:

- span rows for the span head, as read by decision/extraction.py:
  {"id", "source", "text", "types": [{"key", "name", "description"}], "entities": [{"start", "end", "label"}]}
- CJ decision cases for the decision head: {"id", "state", "questions", "gold", "_meta"}.
"""
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

SPLITS = ("train", "dev", "test")
SENTENCE_END = "。！？!?；;\n"
OTHER = "其它"
NONE_RELATION = "无关系"


@dataclass
class Raw:
    uid: str
    split: str
    text: str                     # the source text; leakage checks and dedup read it
    payload: object = None


@dataclass
class Views:
    spans: list = field(default_factory=list)        # (file stem, row)
    decisions: list = field(default_factory=list)    # CJ cases
    structured: dict = None


class Source:
    """One dataset: `records()` reads it with its splits, `views(raw, rng, options)` makes a record's views."""
    name = dataset = ""
    quality = "gold"              # gold (human), silver (automatic), distant (distant supervision)
    domain = "general"
    license = ""

    def records(self):
        raise NotImplementedError

    def views(self, raw, rng, options):
        raise NotImplementedError

    def meta(self, raw, task, **extra):
        return {"source_family": f"{task}:{self.name}", "task_id": task, "source": self.name,
                "dataset": self.dataset, "quality": self.quality, "domain": self.domain, "split": raw.split,
                "group_key": f"{self.name}:{raw.uid}", **extra}


def read_parquet(path):
    """A parquet table; pyarrow is needed only by the readers, not by the views."""
    import pyarrow.parquet as pq
    return pq.read_table(path)


def norm(text):
    """Comparison form: NFKC (full-width to half-width), no whitespace, lower case."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def stable_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def carve(key, dev=0.0, test=0.0):
    """A split from a stable hash of `key`: the same text always lands in the same split."""
    position = stable_hash(key) % 100000 / 100000
    return "test" if position < test else "dev" if position < test + dev else "train"


def pieces(text, min_length=12):
    """Normalized sentences long enough to identify a text when it reappears inside another one."""
    parts = re.split(f"[{re.escape(SENTENCE_END)}]", unicodedata.normalize("NFKC", text))
    return {norm(part) for part in parts if len(norm(part)) >= min_length}


def type_entry(key, name, description=""):
    return {"key": key, "name": name, "description": description}


def entity(start, end, label):
    return {"start": start, "end": end, "label": label}


def span_row(row_id, source, text, types, entities, meta):
    return {"id": row_id, "source": source, "text": text, "types": types,
            "entities": sorted(entities, key=lambda e: (e["start"], e["end"], e["label"])), "_meta": meta}


def choice_case(case_id, state, instructions, criteria, gold, meta):
    """`gold` is one candidate key, or a list of keys sharing the probability mass."""
    if isinstance(gold, (list, tuple, set)):
        gold = sorted(set(gold))
        target = {"type": "choice", "probabilities": {k: (1 / len(gold) if k in gold else 0.0) for k in criteria}}
        gold = gold[0] if len(gold) == 1 else target
    if not 2 <= len(criteria) <= 16:
        raise ValueError(f"{case_id}: {len(criteria)} candidates")
    return {"id": case_id, "state": state,
            "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}},
            "gold": {"q": gold}, "_meta": meta}


def noul_case(case_id, state, instructions, answer, meta):
    return {"id": case_id, "state": state, "questions": {"q": {"type": "noul", "instructions": instructions}},
            "gold": {"q": "true" if answer else "false"}, "_meta": meta}


def bio_entities(tags):
    """`(start, end, label)` token spans of a BIO sequence; an I- without its B- starts a new entity."""
    spans, current = [], None
    for position, tag in enumerate(list(tags) + ["O"]):
        prefix, _, label = tag.partition("-")
        if current and (prefix != "I" or label != current[2]):
            spans.append((current[0], position, current[2]))
            current = None
        if prefix == "B" or (prefix == "I" and current is None):
            current = (position, None, label)
    return spans


def join_tokens(tokens):
    """Text and token start offsets; consecutive ASCII words keep a space between them."""
    text, starts = "", []
    for token in tokens:
        if text and re.fullmatch(r"[A-Za-z0-9]", text[-1]) and re.match(r"[A-Za-z0-9]", token) and len(token) > 1:
            text += " "
        starts.append(len(text))
        text += token
    return text, starts + [len(text)]


def locate(text, mentions):
    """Character spans for `(label, surface)` pairs given as a set, as B2NERD and CMeIE list them.

    Every occurrence is a mention unless it overlaps a longer mention placed before it. A surface
    found only inside longer mentions is placed at its first occurrence (a nested entity), and one
    with two labels gets both. Returns the spans and the pairs that do not occur in the text.
    """
    spans, taken, missing = [], bytearray(len(text)), []
    placed = {}
    for label, surface in sorted(set(mentions), key=lambda m: (-len(m[1]), m[1], m[0])):
        if not surface:
            missing.append((label, surface))
            continue
        starts, start = [], text.find(surface)
        while start >= 0:
            starts.append(start)
            start = text.find(surface, start + len(surface))
        if not starts:
            missing.append((label, surface))
            continue
        free = [s for s in starts if (s, s + len(surface)) in placed or not any(taken[s:s + len(surface)])]
        for start in free or starts[:1]:
            end = start + len(surface)
            if label not in placed.setdefault((start, end), set()):
                placed[(start, end)].add(label)
                spans.append(entity(start, end, label))
            taken[start:end] = b"\x01" * (end - start)
    return sorted(spans, key=lambda e: (e["start"], e["end"], e["label"])), missing


def chunks(text, entities, max_chars):
    """`(start, end)` windows of at most `max_chars` that cut after sentence ends and never inside an entity.

    A window with no allowed cut is cut at its limit; entities crossing that cut are lost to both sides.
    """
    if len(text) <= max_chars:
        return [(0, len(text))]
    inside = bytearray(len(text) + 1)
    for e in entities:
        inside[e["start"] + 1:e["end"]] = b"\x01" * max(0, e["end"] - e["start"] - 1)
    windows, start = [], 0
    while len(text) - start > max_chars:
        limit = start + max_chars
        cut = next((p for p in range(limit, start, -1) if text[p - 1] in SENTENCE_END and not inside[p]), None)
        if cut is None:
            cut = next((p for p in range(limit, start + max_chars // 2, -1) if not inside[p]), limit)
        windows.append((start, cut))
        start = cut
    windows.append((start, len(text)))
    return [(s, e) for s, e in windows if text[s:e].strip()]


def window_rows(row_id, source, text, types, entities, meta, max_chars):
    """Span rows for `text` cut into windows; returns the rows and the number of entities lost at forced cuts."""
    rows, kept = [], 0
    for index, (start, end) in enumerate(chunks(text, entities, max_chars)):
        inner = [entity(e["start"] - start, e["end"] - start, e["label"]) for e in entities
                 if start <= e["start"] and e["end"] <= end]
        kept += len(inner)
        suffix = f"#{index}" if (start, end) != (0, len(text)) else ""
        rows.append(span_row(row_id + suffix, source, text[start:end], types, inner, meta))
    return rows, len(entities) - kept


def mark(text, start, end, left="【", right="】"):
    return text[:start] + left + text[start:end] + right + text[end:]


def sample_candidates(gold, pool, rng, limit=16, first=()):
    """`gold` plus distractors up to `limit`: the `first` keys (hard negatives) before random ones from `pool`."""
    chosen = list(dict.fromkeys(list(gold) + [k for k in first if k in pool]))
    rest = [k for k in pool if k not in chosen]
    rng.shuffle(rest)
    return (chosen + rest)[:max(limit, len(gold))]
