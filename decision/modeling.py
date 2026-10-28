"""`DecisionModel`: one backbone, two readouts, one weight file.

    decision    the backbone's next-token logits over the candidate letters at the last position
    extraction  `SpanHead` over the last hidden states: each candidate span chooses [none, type_1..type_M]

The span head follows CalOn's typed span decisions (extraordinarylab/calon, arXiv 2610.17113): span
vectors from first/last/next/mean token states and a width embedding, type vectors pooled from the
type text in the prompt, a scaled dot product per (span, type) and a learned "none" key. Its code is
not copied here.

On disk a model is a standard causal-LM directory (llama.cpp, vLLM and `AutoModelForCausalLM` read
it as usual) whose safetensors may also hold `span_head.*`, described in `decision_config.json`:

    model.safetensors       backbone (+ span_head.* tensors)
    config.json             backbone config
    decision_config.json    temperature, prompt version, and the span head's settings
    tokenizer files

`DecisionModel` is a plain `nn.Module` around the backbone rather than a `PreTrainedModel` subclass,
so that layout stays the backbone's own.
"""
import contextlib
from pathlib import Path

import torch
from torch import nn

from .configuration import DecisionConfig, ExtractionConfig
from .data.batching import greedy_batches
from .processing import PAD_MULTIPLE
from .utils import resolve_pretrained

SPAN_PREFIX = "span_head."


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


def load_backbone(path, device="cuda", dtype=None, ignore_span_head=False):
    """The causal LM of `path` (the text backbone of a multimodal Qwen3.5 checkpoint), in eval mode."""
    from transformers import AutoConfig, AutoModelForCausalLM
    dtype = dtype or (torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16)
    model_class, previous = None, None
    if ignore_span_head:
        # span_head.* tensors share the weight file; tell the backbone they are expected.
        model_class = AutoModelForCausalLM._model_mapping[type(AutoConfig.from_pretrained(str(path),
                                                                                        local_files_only=True))]
        previous = model_class._keys_to_ignore_on_load_unexpected
        model_class._keys_to_ignore_on_load_unexpected = list(previous or []) + [r"^span_head\."]
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, attn_implementation="sdpa",
                                                     local_files_only=True)
    finally:
        if model_class is not None:
            model_class._keys_to_ignore_on_load_unexpected = previous
    model.config.use_cache = False
    return model.to(device).eval()


def read_span_head(path, config, device):
    """The span head stored in a model directory's safetensors."""
    from safetensors import safe_open
    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith(SPAN_PREFIX):
                    tensors[key[len(SPAN_PREFIX):]] = stream.get_tensor(key)
    if not tensors:
        raise ValueError(f"{path}: decision_config.json describes a span head but no span_head.* tensors were found")
    head = SpanHead(**config.head_kwargs())
    head.load_state_dict(tensors)
    return head.to(device).eval()


class DecisionModel(nn.Module):
    """A causal-LM backbone that answers decisions, and, with a span head, extracts typed spans.

        model = DecisionModel.from_pretrained("models/chinese-decision-0.8b")       # decisions only
        model = DecisionModel.from_pretrained("Qwen/Qwen3.5-0.8B")                  # a raw base, to train
        model.add_span_head()                                                       # add extraction
        model.enable_lora(r=32)                                                     # train an adapter
        model.save_pretrained("out/")                                               # one directory, one weight file
    """

    def __init__(self, backbone, config=None, span_head=None):
        super().__init__()
        self.backbone = backbone
        self.config = config or DecisionConfig()
        self.span_head = span_head

    @classmethod
    def from_pretrained(cls, name_or_path, *, adapter=None, trainable=False, device="cuda", dtype=None,
                        revision=None):
        """Load a released decision model, a unified model, or a raw base; `adapter` adds a PEFT adapter on top."""
        path = resolve_pretrained(name_or_path, revision)
        config = DecisionConfig.from_pretrained(path)
        backbone = load_backbone(path, device, dtype, ignore_span_head=config.can_extract)
        if adapter:
            from peft import PeftModel
            backbone = PeftModel.from_pretrained(backbone, str(adapter), is_trainable=trainable)
        head = read_span_head(path, config.extraction, device) if config.can_extract else None
        return cls(backbone, config, head)

    # structure

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @property
    def hidden_size(self):
        return self.backbone.config.hidden_size

    @property
    def is_peft(self):
        return hasattr(self.backbone, "get_base_model")

    @property
    def base_model(self):
        return self.backbone.get_base_model() if self.is_peft else self.backbone

    def add_span_head(self, max_span_units=20, none_shift=0.0, echo=True, **kwargs):
        """A new, randomly initialized span head (replacing any existing one)."""
        extraction = ExtractionConfig(hidden_size=self.hidden_size, max_span_units=max_span_units,
                                      none_shift=none_shift, echo=echo, **kwargs)
        self.span_head = SpanHead(**extraction.head_kwargs()).to(self.device)
        self.config.extraction = extraction
        return self.span_head

    def load_span_head(self, path):
        """A span head saved by a training run (`span_head.pt`: state_dict, config, none_shift, echo)."""
        state = torch.load(path, map_location=self.device)
        self.span_head = SpanHead(**state["config"]).to(self.device)
        self.span_head.load_state_dict(state["state_dict"])
        self.config.extraction = ExtractionConfig(**state["config"], none_shift=state.get("none_shift", 0.0),
                                                  echo=state.get("echo", True))
        return self.span_head

    def enable_lora(self, r=32, alpha=None, dropout=0.05, targets=None):
        """Wrap the backbone in a trainable LoRA adapter; returns its parameters."""
        from peft import LoraConfig, TaskType, get_peft_model
        from .training.common import LORA_TARGETS
        self.backbone = get_peft_model(self.backbone, LoraConfig(
            r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, task_type=TaskType.CAUSAL_LM,
            target_modules=targets or LORA_TARGETS))
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def adapter_disabled(self):
        """A context in which the backbone runs without its adapter (the starting model)."""
        return self.backbone.disable_adapter() if self.is_peft else contextlib.nullcontext()

    def merge_adapter(self, safe_merge=True):
        """Fold a LoRA adapter into the backbone weights."""
        if self.is_peft:
            self.backbone = self.backbone.merge_and_unload(safe_merge=safe_merge)
        return self

    # readouts

    def decision_logits(self, input_ids, attention_mask, letter_ids, forward=None):
        """Next-token logits over the candidate letters at the last position, `[B, len(letter_ids)]`, float32.

        `forward` runs instead of the backbone, e.g. a DistributedDataParallel wrapper around it.
        """
        output = (forward or self.backbone)(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1,
                                            use_cache=False)
        return output.logits[:, -1, letter_ids].float()

    def letter_tensor(self, processor):
        """The candidate letters' token ids on the model's device."""
        return torch.tensor(processor.letter_ids, device=self.device)

    def score_decisions(self, items, processor, letter_ids=None, forward=None):
        """Logits (invalid candidates at -1e4) and target distributions for encoded decisions."""
        letters = letter_ids if letter_ids is not None else self.letter_tensor(processor)
        inputs = processor.pad_decisions(items, self.device)
        logits = self.decision_logits(inputs["input_ids"], inputs["attention_mask"], letters, forward)
        sizes = torch.tensor([len(e["target"]) for e in items], device=self.device)
        valid = torch.arange(len(letters), device=self.device)[None, :] < sizes[:, None]
        targets = torch.tensor([e["target"] + [0.] * (len(letters) - len(e["target"])) for e in items],
                               device=self.device)
        return logits.masked_fill(~valid, -1e4), targets

    def hidden_states(self, input_ids, attention_mask):
        """Last-layer hidden states, without the LM head."""
        return self.base_model.get_decoder()(input_ids=input_ids, attention_mask=attention_mask,
                                             use_cache=False).last_hidden_state

    def span_utilities(self, batch):
        """`[N, 1 + M]` utilities over [none, types] for every candidate span of an `ExtractionBatch`."""
        if self.span_head is None:
            raise ValueError("this model has no span head; it answers decisions only")
        return self.span_head(self.hidden_states(batch.input_ids, batch.attention_mask), batch)

    @torch.no_grad()
    def joint_readouts(self, decisions, rows, processor, letter_ids=None):
        """Decision logits `[D, len(letter_ids)]` and span utilities `[N, 1 + M]` from one forward pass.

        Encoded decisions and span rows are left-padded to one width and run through the backbone together;
        the LM head reads each decision's last position and the span head the span rows' hidden states.
        """
        if self.span_head is None:
            raise ValueError("this model has no span head; it answers decisions only")
        letters = letter_ids if letter_ids is not None else self.letter_tensor(processor)
        longest = max(len(item["input_ids"]) for item in [*decisions, *rows])
        width = -(-longest // PAD_MULTIPLE) * PAD_MULTIPLE if len(decisions) + len(rows) > 1 else longest
        batch = processor.collate_extraction(rows, width=width).to(self.device)
        ids = torch.full((len(decisions), width), processor.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for b, item in enumerate(decisions):
            ids[b, width - len(item["input_ids"]):] = torch.tensor(item["input_ids"])
            mask[b, width - len(item["input_ids"]):] = 1
        input_ids = torch.cat([batch.input_ids, ids.to(self.device)])
        attention_mask = torch.cat([batch.attention_mask, mask.to(self.device)])
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            hidden = self.hidden_states(input_ids, attention_mask)
            logits = self.base_model.get_output_embeddings()(hidden[len(rows):, -1])[:, letters].float()
        return logits, self.span_head(hidden[:len(rows)], batch)

    @torch.no_grad()
    def predict_spans(self, items, processor, threshold=0.0, none_shift=None, token_budget=12288, batch_size=16):
        """Decoded `(start, end, type, probability)` entities for encoded span rows, in input order."""
        if none_shift is None:
            none_shift = self.config.extraction.none_shift if self.config.extraction else 0.0
        was_training = self.training
        self.eval()
        device = self.device
        predictions = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
        for batch_items in greedy_batches([dict(items[i], index=i) for i in order], batch_size, token_budget):
            batch = processor.collate_extraction(batch_items).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = self.hidden_states(batch.input_ids, batch.attention_mask)
            utilities, start = self.span_head(hidden, batch), 0
            for item in batch_items:
                count = len(item["cands"])
                predictions[item["index"]] = processor.decode_spans(item, utilities[start:start + count],
                                                                    none_shift, threshold)
                start += count
        self.train(was_training)
        return predictions

    # saving

    def save_pretrained(self, directory, processor=None, safe_merge=True):
        """Write one model directory: the backbone (any adapter merged) and the span head in one weight set."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.merge_adapter(safe_merge)
        state = dict(self.backbone.state_dict())
        if self.span_head is not None:
            state.update({SPAN_PREFIX + k: v.detach().contiguous() for k, v in self.span_head.state_dict().items()})
        self.backbone.save_pretrained(str(directory), state_dict=state, safe_serialization=True)
        self.config.save_pretrained(directory)
        if processor is not None:
            processor.save_pretrained(directory)
        return directory
