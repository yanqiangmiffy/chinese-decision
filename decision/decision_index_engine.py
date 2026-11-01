"""In-process adapter for https://github.com/apolinario/decision-index.

Install that optional evaluation kit separately, then use
``--engine decision.decision_index_engine:ChineseDecisionEngine --model <path-or-Hub-id>``.
The official runner owns sampling, scoring, resume and result files.

A `choice` question with more candidates than one prompt holds (A-Z), or one whose prompt is longer
than the context window, goes through a candidate tournament (decision/serving/tournament.py): groups
of at most `group_size` candidates, each group's top `keep` advance until the finalists fit into one
question. Every candidate keeps a probability (see `spread`), so the answer covers all options.
"""
import math
import platform
from pathlib import Path

import torch
import transformers
from decision_index.engines import Engine, Unsupported

from .configuration import CONFIG_NAME, LETTERS, PROMPT_VERSION
from .pipeline import DecisionPipeline
from .processing import options_for
from .serving.api import build_answer
from .serving.backends import candidate_probabilities
from .serving.tournament import DEFAULT_GROUP_SIZE, DEFAULT_KEEP, MIN_GROUP_SIZE, Tournament
from .utils import resolve_pretrained, sha256


def spread(match, finalists):
    """Probabilities for every candidate of a finished tournament, in `match.criteria` order.

    The finalists keep the final question's probabilities. Walking back through the rounds, the
    candidates a group eliminated get that group's own probabilities, scaled so the group's
    advancers carry the mass the later rounds gave them; each is capped at the smallest surviving
    value, so a later round always outranks an earlier one and the winner stays the most probable.
    """
    mass = dict(finalists)
    for round_ in reversed(match.rounds):
        floor = min(mass.values())
        eliminated = {}
        for group in round_["groups"]:
            local = group["probabilities"]
            advanced = sum(local[key] for key in group["advanced"])
            later = sum(mass[key] for key in group["advanced"])
            for key, p in local.items():
                if key not in mass:
                    eliminated[key] = min(later * p / advanced, floor)
        mass.update(eliminated)
    total = sum(mass.values())
    return [mass[key] / total for key in match.criteria]


class ChineseDecisionEngine(Engine):
    name = "chinese-decision"
    latency = ("In-process request wall time including prompt construction, tokenization, all questions' "
               "forward passes (batched per request and tournament round) and probability formatting; "
               "synchronized CUDA; excludes model loading.")

    def __init__(self, model, revision=None, device="cuda", threads=8, group_size=DEFAULT_GROUP_SIZE,
                 keep=DEFAULT_KEEP, seed=0, batch_size=64, token_budget=65536):
        super().__init__(model=model, revision=revision, device=device, threads=threads, group_size=group_size,
                         keep=keep, seed=seed, batch_size=batch_size, token_budget=token_budget)
        if threads < 1 or batch_size < 1 or token_budget < 1:
            raise ValueError("threads, batch_size and token_budget must be positive")
        if not MIN_GROUP_SIZE <= group_size <= len(LETTERS) or not 1 <= keep <= group_size // 2:
            raise ValueError(f"group_size must be in [{MIN_GROUP_SIZE}, {len(LETTERS)}] and keep in [1, group_size // 2]")
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment. Install a CUDA-enabled "
                               "PyTorch build, or use --option device=cpu for a small smoke test.")
        torch.set_num_threads(threads)
        path = resolve_pretrained(model, revision).resolve()
        if not (path / CONFIG_NAME).is_file():
            raise ValueError("use an exported decision model containing decision_config.json")
        self.pipe = DecisionPipeline.from_pretrained(path, device=device)
        config = self.pipe.model.config
        if config.prompt_version != PROMPT_VERSION:
            raise ValueError(f"unsupported prompt version: {config.prompt_version}")
        if not math.isfinite(self.pipe.temperature) or self.pipe.temperature <= 0:
            raise ValueError("model temperature must be finite and positive")
        if not 2 <= config.max_options <= len(LETTERS) or config.max_length < 1:
            raise ValueError("invalid model capacity settings")
        self.max_options, self.max_length = config.max_options, config.max_length
        self.group_size, self.keep, self.seed = group_size, keep, seed
        self.batch_size, self.token_budget = batch_size, token_budget
        self.model_name = config.model_name or str(model)
        root = Path(__file__).resolve().parent
        sources = ("decision_index_engine.py", "configuration.py", "modeling.py", "processing.py",
                   "pipeline.py", "serving/api.py", "serving/backends.py", "serving/tournament.py")
        artifacts = sorted({*path.glob("*.safetensors"), *path.glob("*.json"), *path.glob("*.jinja")})
        self.provenance = {
            "kind": "in-process", "model": str(model), "requested_revision": revision,
            "resolved_path": str(path),
            "hub_revision": path.name if path.parent.name == "snapshots" else None,
            "artifact_sha256": {p.name: sha256(p) for p in artifacts},
            "source_sha256": {name: sha256(root / name) for name in sources},
            "prompt_version": config.prompt_version, "temperature": self.pipe.temperature,
            "max_context_tokens": self.max_length, "max_options_per_prompt": self.max_options,
            "question_types": ["choice", "noul"],
            "tournament": {"group_size": group_size, "keep": keep, "seed": seed},
            "policy": "Original state, instructions and all candidates, in order; no truncation, filtering, "
                      "translation or benchmark-specific prompt changes. A choice question with more than "
                      f"{self.max_options} candidates, or whose prompt exceeds the context window, is split "
                      f"into shuffled groups of at most {group_size} (seeded); each group's top {keep} "
                      "advance until the finalists fit one prompt. Eliminated candidates keep probabilities "
                      "below every survivor. A request is rejected whole if any prompt still does not fit.",
        }

    def _encode(self, state, instructions, options):
        processor = self.pipe.processor
        ids = processor.tokenizer.encode(processor.build_prompt(state, instructions, options),
                                         add_special_tokens=False)
        return {"input_ids": ids, "options": options, "target": [1 / len(options)] * len(options)}

    def _round(self, state, match, name):
        """Encode the current round of a tournament; Unsupported if a group prompt is too long."""
        items = []
        for label, group in match.questions().items():
            item = self._encode(state, group["instructions"], options_for(group))
            if len(item["input_ids"]) > self.max_length:
                raise Unsupported(f"{name} ({label}): {len(item['input_ids'])} tokens exceed context window "
                                  f"{self.max_length}")
            items.append(item)
        return items

    def _tournament(self, state, name, question):
        """A tournament whose first round fits, shrinking groups for long prompts; else Unsupported."""
        size = self.group_size
        while True:
            match = Tournament(state, question["instructions"], question["criteria"], size,
                               min(self.keep, size // 2), self.seed)
            try:
                return match, self._round(state, match, name)
            except Unsupported:
                if size <= MIN_GROUP_SIZE:
                    raise
                size = max(MIN_GROUP_SIZE, size // 2)

    def _probabilities(self, items):
        logits = self.pipe.candidate_logits(items, batch_size=self.batch_size, token_budget=self.token_budget)
        return [candidate_probabilities(row, self.pipe.temperature) for row in logits]

    def __call__(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        single, matches = {}, {}
        # Preflight every question: never answer only the supported part of a request.
        for name, question in questions.items():
            kind = question["type"]
            if kind not in ("choice", "noul"):
                raise Unsupported(f"unsupported question type: {kind}")
            criteria = question.get("criteria")
            if question.get("instructions") is None:
                raise ValueError("question requires non-null instructions")
            if kind == "choice" and isinstance(criteria, dict) and len(criteria) > self.max_options:
                matches[name] = self._tournament(state, name, question)
                continue
            options = options_for(question)  # Malformed requests remain errors, not capacity refusals.
            item = self._encode(state, question["instructions"], options)
            if len(item["input_ids"]) <= self.max_length:
                single[name] = (kind, options, item)
            elif kind == "choice" and len(options) > MIN_GROUP_SIZE:
                matches[name] = self._tournament(state, name, question)
            else:
                raise Unsupported(f"{name}: {len(item['input_ids'])} tokens exceed context window {self.max_length}")
        tokens = sum(len(item["input_ids"]) for _, _, item in single.values())
        answers, pending = {}, list(single.items())
        rounds = {name: items for name, (_, items) in matches.items()}
        # One batched forward per step: every plain question, plus each open tournament's current round.
        while pending or rounds:
            batch = [item for _, (_, _, item) in pending] + [item for items in rounds.values() for item in items]
            tokens += sum(len(item["input_ids"]) for items in rounds.values() for item in items)
            values = self._probabilities(batch)
            for name, (kind, options, _) in pending:
                answers[name] = build_answer(kind, options, values.pop(0))
            pending, following = [], {}
            for name, items in rounds.items():
                match = matches[name][0]
                match.record([values.pop(0) for _ in items])
                if match.done:
                    final = match.answer["probabilities"]
                    answer = build_answer("choice", list(match.criteria.items()), spread(match, final))
                    answer["choice"] = match.answer["choice"]
                    answers[name] = answer
                else:
                    following[name] = self._round(state, match, name)
            rounds = following
        response = {"model": self.model_name, "answers": {name: answers[name] for name in questions},
                    "usage": {"input_tokens": tokens, "output_tokens": 0}}
        return response, None

    def synchronize(self):
        if self.pipe.model.device.type == "cuda":
            torch.cuda.synchronize(self.pipe.model.device)

    def runtime(self):
        device = self.pipe.model.device
        return {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "dtype": str(next(self.pipe.model.parameters()).dtype),
                "threads": torch.get_num_threads()}

    def close(self):
        device = self.pipe.model.device
        del self.pipe
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
