"""Shared option-logit inference and distribution cross entropy."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import LETTERS


def load_tokenizer(base):
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(base, adapter=None, trainable=False, device="cuda"):
    # AutoModelForCausalLM extracts the Qwen3.5 text backbone from multimodal weights.
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                               attn_implementation="sdpa", local_files_only=True).to(device)
    model.config.use_cache = False
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model


def letter_tokens(tokenizer, device):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(x) != 1 for x in ids):
        raise ValueError("all candidate letters must be single tokens")
    return torch.tensor([x[0] for x in ids], device=device)


def forward_batch(model, tokenizer, letter_ids, batch):
    device = next(model.parameters()).device
    inputs = tokenizer.pad([{"input_ids": e["input_ids"], "attention_mask": [1] * len(e["input_ids"])}
                            for e in batch], padding=True, pad_to_multiple_of=64, return_tensors="pt").to(device)
    logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, letter_ids].float()
    sizes = torch.tensor([len(e["target"]) for e in batch], device=device)
    valid = torch.arange(len(letter_ids), device=device)[None, :] < sizes[:, None]
    logits = logits.masked_fill(~valid, -1e4)
    targets = torch.tensor([e["target"] + [0.] * (len(letter_ids) - len(e["target"])) for e in batch], device=device)
    return logits, targets
