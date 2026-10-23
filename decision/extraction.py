"""Span extraction as typed decisions: every candidate span chooses one of [none, type_1..type_M].

The decision formulation follows CalOn (extraordinarylab/calon, arXiv 2610.17113): span vectors
from start/end/mean/width, type vectors pooled from the type names written in the input, a
scaled dot product per (span, type), and a learned "none" key, so deciding *whether* to extract
a span is one softmax with deciding *which* type it is. Its code is not copied here.

CalOn runs a bidirectional encoder. This backbone is causal with linear-attention layers, so the
input uses an echo layout instead: the sentence is read once with natural tokenization, then
repeated one unit per character, and spans are read from the repeat. Every repeated unit has
already seen the whole sentence, and every character boundary is a token boundary (natural Qwen
tokens split 9-14% of CLUENER/PeopleDaily entities). The sequence ends at the last unit; nothing
after it could influence the span vectors.

    <|im_start|>user
    从文本中找出下列类型的实体。
    实体类型：
    - 人名：真实或虚构人物的姓名        <- each type's tokens pool into a type vector
    - ...
    文本：
    {text, natural tokens}             <- omitted with echo=False
    逐字定位：
    {text, one unit per character}     <- span vectors
"""
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from .data import greedy_batches
from .rlcd_loss import RLCDLoss, RLCDLossConfig

INSTRUCTION = "从文本中找出下列类型的实体。\n实体类型：\n"
# One unit per CJK character or symbol, one per ASCII word or number, one per whitespace character.
UNIT_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|\s|.", re.S)


def text_units(text):
    """Character offsets `(start, end)` of the span units of `text`."""
    return [match.span() for match in UNIT_PATTERN.finditer(text)]


def user_turn_prefix(tokenizer):
    """The chat template's text before a user message's content."""
    sentinel = "⁣SPAN⁣"
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": sentinel}], tokenize=False,
                                             add_generation_prompt=False)
    return rendered[:rendered.index(sentinel)]


class ExtractionEncoder:
    """Turns `{"text", "types", "entities"}` rows into token ids, type spans, candidates and gold labels."""

    def __init__(self, tokenizer, max_span_units=20, echo=True, max_length=4096):
        self.tokenizer, self.max_span_units, self.echo, self.max_length = tokenizer, max_span_units, echo, max_length
        self.prefix = self.ids(user_turn_prefix(tokenizer) + INSTRUCTION)
        self.unit_ids = lru_cache(maxsize=65536)(self.ids)

    def ids(self, value):
        return tuple(self.tokenizer.encode(value, add_special_tokens=False))

    def encode(self, row, rng=None):
        types = list(row["types"])
        if rng is not None:
            rng.shuffle(types)          # type vectors must not depend on list position
        ids, type_spans = list(self.prefix), []
        for entry in types:
            ids += self.unit_ids("- ")
            start = len(ids)
            ids += self.ids(f"{entry['name']}：{entry['description']}" if entry.get("description") else entry["name"])
            type_spans.append((start, len(ids)))
            ids += self.unit_ids("\n")
        text = row["text"]
        if self.echo:
            ids += self.unit_ids("文本：\n") + self.ids(text) + self.unit_ids("\n逐字定位：\n")
        else:
            ids += self.unit_ids("文本：\n")
        units, unit_tokens = text_units(text), []
        for start, end in units:
            piece = self.unit_ids(text[start:end])
            unit_tokens.append((len(ids), len(ids) + len(piece) - 1))
            ids += piece
        if len(ids) > self.max_length:
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


@dataclass
class ExtractionBatch:
    input_ids: torch.Tensor          # [B, T], left padded
    attention_mask: torch.Tensor     # [B, T]
    span_row: torch.Tensor           # [N] batch row of each candidate span
    span_first: torch.Tensor         # [N] first token of the span
    span_last: torch.Tensor          # [N] last token of the span
    span_next: torch.Tensor          # [N] first token after the span (the last token at the end of the text)
    span_width: torch.Tensor         # [N] width in units minus one
    type_row: torch.Tensor           # [L] batch row of each type
    type_start: torch.Tensor         # [L] first token of the type name
    type_end: torch.Tensor           # [L] one past its last token
    type_slot: torch.Tensor          # [L] position of the type in its row
    max_types: int
    targets: torch.Tensor            # [N, 1 + M] none first; uniform over gold types
    mask: torch.Tensor               # [N, 1 + M] valid choices

    def to(self, device):
        return replace(self, **{k: v.to(device) for k, v in vars(self).items() if isinstance(v, torch.Tensor)})


def collate_extraction(items, pad_id, multiple=64):
    """Left-pad like `decision.model.forward_batch`: a lone row stays unpadded for the linear-attention layers."""
    longest = max(len(item["input_ids"]) for item in items)
    width = -(-longest // multiple) * multiple if len(items) > 1 else longest
    max_types = max(len(item["types"]) for item in items)
    input_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
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


def mlp(d_in, hidden, d_out, dropout):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))


class SpanHead(nn.Module):
    """Utilities `[N, 1 + M]` over [none, types] for every candidate span, in float32."""

    def __init__(self, hidden_size, max_span_units=20, inner=512, out=256, width_dim=32, dropout=0.1):
        super().__init__()
        self.config = dict(hidden_size=hidden_size, max_span_units=max_span_units, inner=inner, out=out,
                           width_dim=width_dim, dropout=dropout)
        self.width = nn.Embedding(max_span_units, width_dim)
        # span: first, last, next-after-span and mean token states, plus a width embedding
        self.span_mlp = mlp(4 * hidden_size + width_dim, inner, out, dropout)
        # type: mean and last token states of its name and description
        self.type_mlp = mlp(2 * hidden_size, inner, out, dropout)
        self.none_key = nn.Parameter(torch.randn(out) * 0.02)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = out ** -0.5

    def forward(self, hidden, batch):
        hidden = hidden.float()
        rows, dim = hidden.shape[0], hidden.shape[-1]
        prefix = torch.cat([hidden.new_zeros(rows, 1, dim), hidden.cumsum(1)], 1)    # [B, T + 1, D]

        def mean(row, first, stop):
            return (prefix[row, stop] - prefix[row, first]) / (stop - first).clamp_min(1)[:, None].to(hidden.dtype)

        r = batch.span_row
        span = torch.cat([hidden[r, batch.span_first], hidden[r, batch.span_last], hidden[r, batch.span_next],
                          mean(r, batch.span_first, batch.span_last + 1),
                          self.width(batch.span_width.clamp_max(self.width.num_embeddings - 1))], -1)
        span = self.span_mlp(span)                                                     # [N, O]
        t = batch.type_row
        kind = self.type_mlp(torch.cat([mean(t, batch.type_start, batch.type_end), hidden[t, batch.type_end - 1]], -1))
        table = kind.new_zeros(rows, batch.max_types, kind.shape[-1])
        table[t, batch.type_slot] = kind
        utilities = torch.einsum("nmo,no->nm", table[r], span) * self.scale + self.bias
        none = (span @ self.none_key)[:, None] * self.scale
        return torch.cat([none, utilities], 1)


def backbone_hidden(model, batch):
    """Last-layer hidden states of a causal LM (or a PEFT wrapper around one), without the LM head."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.get_decoder()(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                              use_cache=False).last_hidden_state


class SpanLoss:
    """`RLCDLoss` per span, with "none" spans down-weighted by `none_weight`.

    Down-weighting multiplies the trained none probability by `none_weight`; `none_shift` gives the
    log(1 / none_weight) added to the none utility at inference to undo it.
    """

    def __init__(self, config: RLCDLossConfig, none_weight=1.0):
        self.loss_fn = RLCDLoss(replace(config, reduction="none"))
        self.none_weight = none_weight

    @property
    def none_shift(self):
        return -torch.log(torch.tensor(self.none_weight)).item()

    def __call__(self, utilities, batch, sigma=None):
        out = self.loss_fn(utilities, batch.targets, batch.mask, sigma=sigma)
        weight = torch.where(batch.targets[:, 0] >= 1.0, self.none_weight, 1.0)
        return (out.loss * weight).sum() / weight.sum(), out


@torch.no_grad()
def decode(item, utilities, none_shift=0.0, threshold=0.0, flat=True):
    """Predicted `(start, end, type, probability)` entities of one encoded row.

    Each span takes its most probable choice; non-none choices at or above `threshold` are kept,
    and with `flat` overlapping spans are resolved greedily by probability.
    """
    utilities = utilities[:, :1 + len(item["types"])].float().clone()
    utilities[:, 0] += none_shift
    confidence, choice = utilities.softmax(-1).max(-1)
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


@torch.no_grad()
def predict(model, head, items, pad_id, none_shift=0.0, threshold=0.0, token_budget=12288, batch_size=16):
    """Decoded entities for encoded rows, in input order; batches are cut by length and token budget."""
    was_training = model.training, head.training
    model.eval()
    head.eval()
    device = next(head.parameters()).device
    predictions = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    for batch_items in greedy_batches([dict(items[i], index=i) for i in order], batch_size, token_budget):
        batch = collate_extraction(batch_items, pad_id).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = backbone_hidden(model, batch)
        utilities, start = head(hidden, batch), 0
        for item in batch_items:
            count = len(item["cands"])
            predictions[item["index"]] = decode(item, utilities[start:start + count], none_shift, threshold)
            start += count
    model.train(was_training[0])
    head.train(was_training[1])
    return predictions


class SpanExtractor:
    """A trained run for inference: `extract(text, types)` returns typed character spans with probabilities.

    `types` is a list of type names or a `{name: description}` mapping; any names work, including
    ones never seen in training.
    """

    def __init__(self, model, head, tokenizer, none_shift=0.0, echo=True):
        self.model, self.head, self.tokenizer, self.none_shift = model.eval(), head.eval(), tokenizer, none_shift
        self.encoder = ExtractionEncoder(tokenizer, head.config["max_span_units"], echo)

    @classmethod
    def from_run(cls, base, run, device="cuda"):
        from .model import load_model, load_tokenizer
        state = torch.load(Path(run) / "span_head.pt", map_location=device)
        head = SpanHead(**state["config"]).to(device)
        head.load_state_dict(state["state_dict"])
        model = load_model(base, str(Path(run) / "adapter"), device=device)
        return cls(model, head, load_tokenizer(base), state["none_shift"], state["echo"])

    def predict(self, rows, threshold=0.0):
        items = [self.encoder.encode(row) for row in rows]
        if any(item is None for item in items):
            raise ValueError("input longer than the encoder's max_length")
        return items, predict(self.model, self.head, items, self.tokenizer.pad_token_id, self.none_shift, threshold)

    def extract(self, text, types, threshold=0.0):
        described = types.items() if isinstance(types, dict) else ((name, "") for name in types)
        row = {"text": text, "types": [{"key": n, "name": n, "description": d} for n, d in described]}
        _, (found,) = self.predict([row], threshold)
        return [{"text": text[s:e], "label": label, "start": s, "end": e, "score": round(p, 4)}
                for s, e, label, p in sorted(found)]


def entity_metrics(items, predictions, bins=10):
    """Micro precision/recall/F1 over exact (start, end, type) matches, per type as well, plus ECE of predictions."""
    tp = fp = fn = unrepresented = 0
    confidences, hits, per_type = [], [], {}
    for item, predicted in zip(items, predictions):
        gold = item["gold_entities"]
        keys = {(s, e, t) for s, e, t, _ in predicted}
        tp += len(keys & gold)
        fp += len(keys - gold)
        fn += len(gold - keys)
        unrepresented += item["unrepresented"]
        for kind, found in (("tp", keys & gold), ("fp", keys - gold), ("fn", gold - keys)):
            for _, _, label in found:
                per_type.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})[kind] += 1
        for s, e, t, probability in predicted:
            confidences.append(probability)
            hits.append(float((s, e, t) in gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ece = 0.0
    for b in range(bins):
        chosen = [i for i, c in enumerate(confidences) if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1)]
        if chosen:
            ece += len(chosen) / len(confidences) * abs(sum(hits[i] for i in chosen) / len(chosen)
                                                        - sum(confidences[i] for i in chosen) / len(chosen))
    types = {label: {"gold": c["tp"] + c["fn"], "f1": 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"])}
             for label, c in sorted(per_type.items())}
    return {"rows": len(items), "gold": tp + fn, "predicted": tp + fp, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "entity_ece": ece if confidences else None, "unrepresented_gold": unrepresented, "per_type": types}
