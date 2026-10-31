"""Inputs and outputs of a decision model: the CJ case format, both prompt layouts, and `DecisionProcessor`.

Decisions follow hmm's option-letter objective: a question's candidates are listed as letters and
the model's next-token distribution over those letters is the answer. CJ labels are distributions,
unlike hmm's original scalar gold labels.

Extraction follows CalOn's typed span decisions on a causal backbone: the prompt lists entity types,
then reads the text once with natural tokens and echoes it one unit per character; spans are read
from the echo (see decision.modeling.SpanHead).

    <|im_start|>user
    从文本中找出下列类型的实体。
    实体类型：
    - 人名：真实或虚构人物的姓名        <- each type's tokens pool into a type vector
    文本：
    {text, natural tokens}             <- omitted with echo=False
    逐字定位：
    {text, one unit per character}     <- span vectors

The module imports without torch, so the GGUF server can compile prompts on a CPU-only install.
"""
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

from .configuration import LETTERS, MAX_LENGTH, PROMPT_VERSION, DecisionConfig
from .utils import resolve_pretrained

if TYPE_CHECKING:
    import torch

__all__ = ["DecisionProcessor", "Example", "ExtractionBatch", "LETTERS", "MAX_LENGTH", "PROMPT_VERSION",
           "compile_prompt", "examples_for", "group_key", "options_for", "partition", "target_for", "text_units"]

EXTRACTION_INSTRUCTION = "从文本中找出下列类型的实体。\n实体类型：\n"
# One unit per CJK character or symbol, one per ASCII word or number, one per whitespace character.
UNIT_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|\s|.", re.S)
PAD_MULTIPLE = 64


# --- the CJ case format ---------------------------------------------------------------------------

def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def options_for(question):
    """`[(candidate key, description)]` of a CJ question; raises on an invalid question."""
    if not isinstance(question, dict) or question.get("instructions") is None:
        raise ValueError("question requires non-null instructions")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict):
        options = [(str(k), text(v)) for k, v in criteria.items()]
    elif kind == "score" and isinstance(criteria, list):
        options = [(str(i), text(v)) for i, v in enumerate(criteria)]
    elif kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object")
        criteria = criteria or {}
        options = [("false", text(criteria.get("false", "否"))), ("true", text(criteria.get("true", "是")))]
    else:
        raise ValueError(f"invalid question type/criteria: {kind}")
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"requires 2-{len(LETTERS)} candidates; got {len(options)}")
    return options


def target_for(gold, options, kind):
    """The target distribution over `options` for a gold answer key or probability object."""
    keys = [k for k, _ in options]
    if isinstance(gold, dict):
        if gold.get("type", kind) != kind:
            raise ValueError("gold type mismatch")
        probabilities = gold.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
            raise ValueError("gold probability keys must match all candidates")
        target = [float(probabilities[k]) for k in keys]
    else:
        answer = str(gold).lower() if kind == "noul" else str(gold)
        if answer not in keys:
            raise ValueError("gold option not found")
        target = [float(k == answer) for k in keys]
    if any(not math.isfinite(p) or p < 0 for p in target) or sum(target) <= 0:
        raise ValueError("invalid probability distribution")
    total = sum(target)
    if abs(total - 1) > 1e-4:
        raise ValueError(f"target mass is {total}, expected 1")
    return [p / total for p in target]


def group_key(case):
    meta = case.get("_meta") or {}
    return str(meta.get("parent_id") or meta.get("group_key") or meta.get("original_id") or case.get("id")
               or text(case["state"]))


def partition(case):
    """A stable 0-4 partition of a case's group, for calibration/selection splits."""
    return int(hashlib.sha256(group_key(case).encode()).hexdigest()[:8], 16) % 5


def user_prompt(state, instructions, options):
    return ("待判断的上下文：\n" + text(state)
            + "\n\n问题：\n" + text(instructions)
            + "\n\n候选项：\n" + "\n".join(f"{LETTERS[i]}: {k} — {v}" for i, (k, v) in enumerate(options))
            + "\n只返回候选项对应的字母。")


def compile_prompt(tokenizer, state, instructions, options):
    """The decision prompt in the tokenizer's chat template, thinking disabled."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt(state, instructions, options)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


@dataclass
class Example:
    """One decision: a state, a question and its candidates, and the target distribution."""

    state: object
    instructions: object
    options: list
    target: list
    kind: str
    domain: str
    source: str
    case_id: str

    @property
    def hard(self):
        return max(self.target) >= 1 - 1e-6

    def render(self, tokenizer, rng=None):
        """Prompt and target; `rng` shuffles `choice` candidates (training only)."""
        order = list(range(len(self.options)))
        if self.kind == "choice" and rng is not None:
            rng.shuffle(order)
        return (compile_prompt(tokenizer, self.state, self.instructions, [self.options[i] for i in order]),
                [self.target[i] for i in order])


def examples_for(case, domain):
    """The decisions of one CJ case (one per question)."""
    meta = case.get("_meta") or {}
    for name, question in case["questions"].items():
        options = options_for(question)
        target = target_for(case["gold"][name], options, question["type"])
        yield Example(case["state"], question["instructions"], options, target, question["type"], domain,
                      str(meta.get("source_family") or meta.get("source_id") or meta.get("source") or "unknown"),
                      str(case.get("id", "")) + ":" + name)


def text_units(text):
    """Character offsets `(start, end)` of the span units of `text`."""
    return [match.span() for match in UNIT_PATTERN.finditer(text)]


def type_entries(types):
    """Span-row types from a list of names or a `{name: description}` mapping."""
    described = types.items() if isinstance(types, dict) else ((name, "") for name in types)
    return [{"key": name, "name": name, "description": description} for name, description in described]


# --- batches -----------------------------------------------------------------------------------------

@dataclass
class ExtractionBatch:
    """Tensors of a span-extraction batch; built by `DecisionProcessor.collate_extraction`."""

    input_ids: "torch.Tensor"          # [B, T], left padded
    attention_mask: "torch.Tensor"     # [B, T]
    span_row: "torch.Tensor"           # [N] batch row of each candidate span
    span_first: "torch.Tensor"         # [N] first token of the span
    span_last: "torch.Tensor"          # [N] last token of the span
    span_next: "torch.Tensor"          # [N] first token after the span (the last token at the end of the text)
    span_width: "torch.Tensor"         # [N] width in units minus one
    type_row: "torch.Tensor"           # [L] batch row of each type
    type_start: "torch.Tensor"         # [L] first token of the type name
    type_end: "torch.Tensor"           # [L] one past its last token
    type_slot: "torch.Tensor"          # [L] position of the type in its row
    max_types: int
    targets: "torch.Tensor"            # [N, 1 + M] none first; uniform over gold types
    mask: "torch.Tensor"               # [N, 1 + M] valid choices

    def to(self, device):
        import torch
        return replace(self, **{k: v.to(device) for k, v in vars(self).items() if isinstance(v, torch.Tensor)})


# --- the processor ---------------------------------------------------------------------------------

class DecisionProcessor:
    """Turns requests into model inputs and model outputs into answers, for both readouts.

        processor = DecisionProcessor.from_pretrained("models/chinese-decision-0.8b")
        item = processor.encode_question(state, question)             # decision input
        row = processor.encode_extraction(processor.extraction_row(text, types))

    Encoded decisions are dicts `{"input_ids", "target", "kind", ...}`; encoded span rows also carry
    candidate spans, type positions and gold labels. `pad_decisions` / `collate_extraction` batch them.
    """

    def __init__(self, tokenizer, config=None, max_span_units=None, echo=None, max_extraction_length=4096):
        self.tokenizer = tokenizer
        self.config = config or DecisionConfig()
        extraction = self.config.extraction
        self.max_span_units = max_span_units or (extraction.max_span_units if extraction else 20)
        self.echo = echo if echo is not None else (extraction.echo if extraction else True)
        self.max_extraction_length = max_extraction_length
        self._prefix = None
        self._unit_ids = lru_cache(maxsize=65536)(self._ids)

    @classmethod
    def from_pretrained(cls, name_or_path, revision=None, **kwargs):
        from transformers import AutoTokenizer
        path = resolve_pretrained(name_or_path, revision)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(tokenizer, DecisionConfig.from_pretrained(path), **kwargs)

    def save_pretrained(self, directory):
        self.tokenizer.save_pretrained(str(directory))

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def max_length(self):
        return self.config.max_length

    @property
    def letter_ids(self):
        """Token id of each candidate letter A-Z; every letter must be one token."""
        ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(x) != 1 for x in ids):
            raise ValueError("all candidate letters must be single tokens")
        return [x[0] for x in ids]

    # decisions

    def build_prompt(self, state, instructions, options):
        return compile_prompt(self.tokenizer, state, instructions, options)

    def encode_example(self, example, rng=None, stats=None):
        """An encoded decision, or None when it is longer than `max_length` (counted in `stats`)."""
        prompt, target = example.render(self.tokenizer, rng)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_length:
            if stats is not None:
                stats[f"{example.domain}:overlength"] += 1
            return None
        return {"input_ids": ids, "target": target, "kind": example.kind, "domain": example.domain,
                "source": example.source, "case_id": example.case_id, "hard": example.hard}

    def encode_case(self, case, domain="general", rng=None, stats=None):
        """Encoded decisions of a CJ case; over-length ones are left out."""
        items = [self.encode_example(example, rng, stats) for example in examples_for(case, domain)]
        return [item for item in items if item is not None]

    def encode_question(self, state, question):
        """An inference input for one question (uniform placeholder target); raises when too long."""
        options = options_for(question)
        ids = self.tokenizer.encode(self.build_prompt(state, question["instructions"], options),
                                    add_special_tokens=False)
        if len(ids) > self.max_length:
            raise ValueError(f"the question is {len(ids)} tokens; the model accepts {self.max_length}")
        return {"input_ids": ids, "target": [1 / len(options)] * len(options), "kind": question["type"],
                "options": options}

    def pad_decisions(self, items, device=None):
        """Left-padded `input_ids` / `attention_mask`.

        Qwen3.5's linear-attention layers zero padded positions only in batches of two or more rows,
        so a lone input stays unpadded; left padding would otherwise leak into its recurrent state.
        """
        inputs = self.tokenizer.pad([{"input_ids": e["input_ids"], "attention_mask": [1] * len(e["input_ids"])}
                                     for e in items], padding=True,
                                    pad_to_multiple_of=PAD_MULTIPLE if len(items) > 1 else None, return_tensors="pt")
        return inputs.to(device) if device is not None else inputs

    # extraction

    def _ids(self, value):
        return tuple(self.tokenizer.encode(value, add_special_tokens=False))

    def _user_turn_prefix(self):
        """The chat template's text before a user message's content."""
        sentinel = "⁣SPAN⁣"
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": sentinel}], tokenize=False,
                                                      add_generation_prompt=False)
        return rendered[:rendered.index(sentinel)]

    @staticmethod
    def extraction_row(text, types):
        """A span row for inference; `types` is a list of names or a `{name: description}` mapping."""
        return {"text": text, "types": type_entries(types)}

    def encode_extraction(self, row, rng=None):
        """Token ids, type spans, candidate spans and gold labels of a `{"text", "types", "entities"}` row.

        None when the row is longer than `max_extraction_length`. `rng` shuffles the type list (training).
        """
        if self._prefix is None:
            self._prefix = self._ids(self._user_turn_prefix() + EXTRACTION_INSTRUCTION)
        types = list(row["types"])
        if rng is not None:
            rng.shuffle(types)          # type vectors must not depend on list position
        ids, type_spans = list(self._prefix), []
        for entry in types:
            ids += self._unit_ids("- ")
            start = len(ids)
            ids += self._ids(f"{entry['name']}：{entry['description']}" if entry.get("description") else entry["name"])
            type_spans.append((start, len(ids)))
            ids += self._unit_ids("\n")
        text = row["text"]
        if self.echo:
            ids += self._unit_ids("文本：\n") + self._ids(text) + self._unit_ids("\n逐字定位：\n")
        else:
            ids += self._unit_ids("文本：\n")
        units, unit_tokens = text_units(text), []
        for start, end in units:
            piece = self._unit_ids(text[start:end])
            unit_tokens.append((len(ids), len(ids) + len(piece) - 1))
            ids += piece
        if len(ids) > self.max_extraction_length:
            return None
        spaces = [text[start:end].isspace() for start, end in units]
        cands = [(i, j) for i in range(len(units)) if not spaces[i]
                 for j in range(i, min(len(units), i + self.max_span_units)) if not spaces[j]]
        index = {cand: k for k, cand in enumerate(cands)}
        start_unit = {start: i for i, (start, _) in enumerate(units)}
        end_unit = {end: i for i, (_, end) in enumerate(units)}
        type_index = {entry["key"]: m for m, entry in enumerate(types)}
        gold, unrepresented = {}, 0
        for entity in row.get("entities", []):
            k = index.get((start_unit.get(entity["start"]), end_unit.get(entity["end"])))
            if k is None or entity["label"] not in type_index:
                unrepresented += 1
                continue
            gold.setdefault(k, set()).add(type_index[entity["label"]])
        return {"id": row.get("id"), "source": row.get("source", "unknown"), "text": text,
                "input_ids": ids, "type_spans": type_spans, "types": [entry["key"] for entry in types],
                "units": units, "unit_tokens": unit_tokens, "cands": cands,
                "gold": {k: sorted(v) for k, v in gold.items()},
                "gold_entities": {(e["start"], e["end"], e["label"]) for e in row.get("entities", [])},
                "unrepresented": unrepresented}

    def collate_extraction(self, items, multiple=PAD_MULTIPLE, width=None):
        """Left-pad like `pad_decisions`: a lone row stays unpadded for the linear-attention layers.

        `width` pads to that length instead, for a batch that also holds other rows (`DecisionModel.joint_readouts`).
        """
        import torch
        longest = max(len(item["input_ids"]) for item in items)
        width = width or (-(-longest // multiple) * multiple if len(items) > 1 else longest)
        max_types = max(len(item["types"]) for item in items)
        input_ids = torch.full((len(items), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(items), width), dtype=torch.long)
        spans, types, targets, masks = [], [], [], []
        for b, item in enumerate(items):
            offset = width - len(item["input_ids"])
            input_ids[b, offset:] = torch.tensor(item["input_ids"])
            attention[b, offset:] = 1
            tokens, last = item["unit_tokens"], len(item["unit_tokens"]) - 1
            for i, j in item["cands"]:
                spans.append((b, tokens[i][0] + offset, tokens[j][1] + offset,
                              (tokens[j + 1][0] if j < last else tokens[j][1]) + offset, j - i))
            for slot, (start, end) in enumerate(item["type_spans"]):
                types.append((b, start + offset, end + offset, slot))
            count = len(item["types"])
            target = torch.zeros((len(item["cands"]), 1 + max_types))
            target[:, 0] = 1.0
            for k, labels in item["gold"].items():
                target[k, 0] = 0.0
                target[k, [1 + m for m in labels]] = 1.0 / len(labels)
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, :1 + count] = True
            targets.append(target)
            masks.append(mask)
        span = torch.tensor(spans, dtype=torch.long).reshape(-1, 5)
        kind = torch.tensor(types, dtype=torch.long).reshape(-1, 4)
        return ExtractionBatch(input_ids, attention, *span.unbind(1), *kind.unbind(1), max_types,
                               torch.cat(targets), torch.cat(masks))

    @staticmethod
    def decode_spans(item, utilities, none_shift=0.0, threshold=0.0, flat=True, decision_shift=None):
        """Predicted `(start, end, type, probability)` entities of one encoded row.

        Probabilities are the softmax with `none_shift` added to the none utility (the calibrated
        ones). Which spans are entities is decided with `decision_shift` instead when it is given:
        each span takes its best choice under that shift, non-none choices whose calibrated
        probability is at or above `threshold` are kept, and with `flat` overlapping spans are
        resolved greedily by probability.
        """
        utilities = utilities[:, :1 + len(item["types"])].float().clone()
        utilities[:, 0] += none_shift
        probabilities = utilities.softmax(-1)
        if decision_shift is None:
            choice = probabilities.argmax(-1)
        else:
            utilities[:, 0] += decision_shift - none_shift
            choice = utilities.argmax(-1)
        confidence = probabilities.gather(1, choice[:, None]).squeeze(1)
        picked = ((choice > 0) & (confidence >= threshold)).nonzero().flatten().tolist()
        found = sorted(((confidence[k].item(), k, choice[k].item() - 1) for k in picked), reverse=True)
        taken, result = set(), []
        for probability, k, m in found:
            i, j = item["cands"][k]
            if flat and any(u in taken for u in range(i, j + 1)):
                continue
            taken.update(range(i, j + 1))
            result.append((item["units"][i][0], item["units"][j][1], item["types"][m], probability))
        return result
