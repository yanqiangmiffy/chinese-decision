"""Bounded, padding-free llama.cpp batches for hybrid Qwen3.5 evaluation.

All rows in a batch have the same true token length. Each row has a distinct
sequence ID; only its final input token requests logits. No tokens are generated.
"""
import math
import sys

from .schema import LETTERS


class GGUFSequenceBatch:
    _initialized = False
    _log_callback = None

    def __init__(self, path, tokenizer, *, parallel=8, context_size=16384, token_budget=512,
                 threads=4, gpu_layers=999, flash_attention=False):
        import llama_cpp as api

        if min(parallel, context_size, token_budget, threads) < 1 or token_budget < parallel:
            raise ValueError("positive limits and token_budget >= parallel are required")
        self.api, self.model, self.context, self.batch = api, None, None, None
        self.parallel, self.context_size, self.token_budget = parallel, context_size, token_budget
        if not self._initialized:
            # Keep warnings/errors, suppress per-graph debug output during a long run.
            @api.llama_log_callback
            def logger(level, message, _):
                if level >= 3:
                    sys.stderr.write(message.decode("utf-8", errors="replace"))
            type(self)._log_callback = logger
            api.llama_log_set(logger, None)
            api.llama_backend_init()
            type(self)._initialized = True
        if gpu_layers and not api.llama_supports_gpu_offload():
            raise RuntimeError("this llama.cpp runtime was built without GPU offload")
        parameters = api.llama_model_default_params()
        parameters.n_gpu_layers = gpu_layers
        self.model = api.llama_model_load_from_file(str(path).encode(), parameters)
        if not self.model:
            raise RuntimeError(f"could not load GGUF weights: {path}")
        try:
            self.vocab = api.llama_model_get_vocab(self.model)
            self.letters = []
            for letter in LETTERS:
                ids = tokenizer.encode(letter, add_special_tokens=False)
                if len(ids) != 1 or ids != self.tokenize(letter):
                    raise ValueError(f"GGUF candidate-token mismatch for {letter}")
                self.letters.append(ids[0])
            config = api.llama_context_default_params()
            config.n_ctx = context_size * parallel
            config.n_seq_max = parallel
            config.n_batch = config.n_ubatch = token_budget
            config.n_threads = config.n_threads_batch = threads
            config.n_outputs_max = parallel
            config.n_outputs_max_per_seq = 1
            config.flash_attn_type = (api.LLAMA_FLASH_ATTN_TYPE_ENABLED if flash_attention
                                      else api.LLAMA_FLASH_ATTN_TYPE_DISABLED)
            config.offload_kqv = bool(gpu_layers)
            config.op_offload = bool(gpu_layers)
            config.no_perf = True
            self.context = api.llama_init_from_model(self.model, config)
            if not self.context:
                raise RuntimeError("could not create the GGUF evaluation context")
            # One sequence ID per token; there are no shared prefixes or padding.
            self.batch = api.llama_batch_init(token_budget, 0, 1)
        except BaseException:
            self.close()
            raise

    def tokenize(self, prompt):
        raw = prompt.encode("utf-8")
        tokens = (self.api.llama_token * (len(raw) + 16))()
        count = self.api.llama_tokenize(self.vocab, raw, len(raw), tokens, len(tokens), False, True)
        if count < 0:
            tokens = (self.api.llama_token * -count)()
            count = self.api.llama_tokenize(self.vocab, raw, len(raw), tokens, len(tokens), False, True)
        if count < 0:
            raise RuntimeError("GGUF tokenization failed")
        return list(tokens[:count])

    def logits(self, rows):
        if not rows or len(rows) > self.parallel:
            raise ValueError("batch size is outside the configured sequence capacity")
        lengths = {len(row["input_ids"]) for row in rows}
        if len(lengths) != 1:
            raise ValueError("hybrid GGUF batches require equal true input lengths; padding is forbidden")
        length = lengths.pop()
        if not 0 < length <= self.context_size:
            raise ValueError("input exceeds the GGUF per-sequence context capacity")
        memory = self.api.llama_get_memory(self.context)
        if not memory:
            raise RuntimeError("hybrid sequence memory is unavailable")
        self.api.llama_memory_clear(memory, True)
        step = self.token_budget // len(rows)
        result = [None] * len(rows)
        for start in range(0, length, step):
            end = min(start + step, length)
            outputs, index = [], 0
            for sequence, row in enumerate(rows):
                for position in range(start, end):
                    self.batch.token[index] = row["input_ids"][position]
                    self.batch.pos[index] = position
                    self.batch.n_seq_id[index] = 1
                    self.batch.seq_id[index][0] = sequence
                    final = position == length - 1
                    self.batch.logits[index] = final
                    if final:
                        outputs.append((sequence, index))
                    index += 1
            self.batch.n_tokens = index
            status = self.api.llama_decode(self.context, self.batch)
            if status != 0:
                raise RuntimeError(f"llama_decode failed with status {status}")
            for sequence, index in outputs:
                pointer = self.api.llama_get_logits_ith(self.context, index)
                if not pointer:
                    raise RuntimeError("missing final-position logits")
                count = len(rows[sequence]["target"])
                values = [float(pointer[i]) for i in self.letters[:count]]
                if not all(math.isfinite(x) for x in values):
                    raise FloatingPointError("non-finite GGUF candidate logits")
                result[sequence] = values
        if any(row is None for row in result):
            raise RuntimeError("incomplete GGUF batch output")
        return result

    def close(self):
        if self.batch is not None:
            self.api.llama_batch_free(self.batch)
            self.batch = None
        if self.context:
            self.api.llama_free(self.context)
            self.context = None
        if self.model:
            self.api.llama_model_free(self.model)
            self.model = None
