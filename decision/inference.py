"""Serving backends. GGUF inference does not import PyTorch or require a GPU."""
import math
from pathlib import Path

from transformers import AutoTokenizer

from .schema import LETTERS


def candidate_probabilities(logits, temperature):
    """Normalize only the current question's candidate logits, without sampling."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = [float(x) / temperature for x in logits]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("candidate logits must be nonempty and finite")
    offset = max(values)
    weights = [math.exp(x - offset) for x in values]
    total = sum(weights)
    return [x / total for x in weights]


class TransformersBackend:
    name = "transformers"

    def __init__(self, base, adapter=None, device="cuda", threads=8):
        import torch
        from .model import letter_tokens, load_model, load_tokenizer

        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.tokenizer = load_tokenizer(base)
        self.model = load_model(base, adapter, device=device).eval()
        self.letters = letter_tokens(self.tokenizer, self.model.device)
        self.device = str(self.model.device)
        self.max_length = None

    def logits(self, ids, count):
        import torch
        from .model import forward_batch

        with torch.inference_mode():
            values, _ = forward_batch(self.model, self.tokenizer, self.letters,
                                      [{"input_ids": ids, "target": [0.] * count}])
        return values[0, :count].cpu().tolist()

    def close(self):
        pass


class GGUFBackend:
    name = "gguf"
    device = "cpu"

    def __init__(self, base, gguf, threads=8, context_size=16384):
        from llama_cpp import Llama

        if threads < 1 or context_size < 1:
            raise ValueError("threads and context_size must be positive")
        if not Path(gguf).is_file():
            raise FileNotFoundError(gguf)
        # Keep exactly the tokenizer and prompt compiler used for training. GGUF
        # tokenizer equivalence is checked for every request before evaluation.
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side="left")
        self.llm = Llama(model_path=str(gguf), n_gpu_layers=0, offload_kqv=False, n_ctx=context_size,
                         n_threads=threads, n_threads_batch=threads, n_batch=512,
                         logits_all=False, embedding=False, verbose=False)
        self.max_length = self.llm.n_ctx()
        self.letters = []
        for letter in LETTERS:
            hf_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            gguf_ids = self.llm.tokenize(letter.encode("utf-8"), add_bos=False, special=True)
            if len(hf_ids) != 1 or hf_ids != gguf_ids:
                self.close()
                raise ValueError(f"GGUF candidate token mismatch for {letter}")
            self.letters.append(hf_ids[0])

    def validate_tokens(self, prompt, ids):
        actual = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if ids != actual:
            raise ValueError("GGUF tokenizer does not match the source tokenizer for this prompt")

    def logits(self, ids, count):
        from llama_cpp import llama_get_logits_ith

        if not ids or len(ids) > self.max_length or not 2 <= count <= len(self.letters):
            raise ValueError("invalid input length or candidate count for GGUF evaluation")
        # In llama-cpp-python >=0.3.36 reset() clears both the attention cache and
        # recurrent state of hybrid architectures such as Qwen3.5.
        self.llm.reset()
        self.llm.eval(ids)
        values = llama_get_logits_ith(self.llm._ctx.ctx, -1)
        if not values:
            raise RuntimeError("llama.cpp returned no final-position logits")
        return [float(values[i]) for i in self.letters[:count]]

    def close(self):
        self.llm.close()
