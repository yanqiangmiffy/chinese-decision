"""In-process adapter for https://github.com/apolinario/decision-index.

Install that optional evaluation kit separately, then use
``--engine decision.decision_index_engine:ChineseDecisionEngine --model <path-or-Hub-id>``.
The official runner owns sampling, scoring, resume and result files.

A `choice` question with more candidates than one prompt holds (A-Z), or one whose prompt is longer
than the context window, goes through a candidate tournament (decision/serving/tournament.py): groups
of at most `group_size` candidates, each group's top `keep` advance until the finalists fit into one
question. Every candidate keeps a probability (see `spread`), so the answer covers all options.

With `--option debias=reverse` every prompt (plain questions and tournament groups alike) is also read
with its candidates in reverse order, and each candidate's probability is the mean over both orders.
This cancels the model's preference for a candidate position (on two-candidate questions exactly).

A pointer model reads all plain questions of a request in one sequence (one per candidate order), as
it was trained; requests too long for that fall back to one sequence per question.
"""
import math
import platform
from pathlib import Path

import torch
import transformers
from decision_index.engines import Engine, Unsupported

from .configuration import CONFIG_NAME, LETTERS, POINTER_MAX_OPTIONS, POINTER_PROMPT_VERSION, PROMPT_VERSION
from .data.batching import shape_batches
from .pipeline import DecisionPipeline
from .processing import options_for
from .serving.api import build_answer
from .serving.backends import candidate_probabilities
from .serving.tournament import DEFAULT_GROUP_SIZE, DEFAULT_KEEP, MIN_GROUP_SIZE, Tournament
from .utils import resolve_pretrained, sha256

# Candidate orders each prompt is read in; probabilities are averaged over them.
DEBIAS = {"none": ["original"], "reverse": ["original", "reversed"]}


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
                 keep=DEFAULT_KEEP, seed=0, batch_size=64, token_budget=65536, debias="none"):
        super().__init__(model=model, revision=revision, device=device, threads=threads, group_size=group_size,
                         keep=keep, seed=seed, batch_size=batch_size, token_budget=token_budget, debias=debias)
        if debias not in DEBIAS:
            raise ValueError(f"debias must be one of {sorted(DEBIAS)}")
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
        if config.prompt_version != (POINTER_PROMPT_VERSION if config.pointer is not None else PROMPT_VERSION):
            raise ValueError(f"unsupported prompt version: {config.prompt_version}")
        if not math.isfinite(self.pipe.temperature) or self.pipe.temperature <= 0:
            raise ValueError("model temperature must be finite and positive")
        # The pointer head has no alphabet: only the context window bounds a question.
        limit = POINTER_MAX_OPTIONS if config.pointer is not None else len(LETTERS)
        if not 2 <= config.max_options <= limit or config.max_length < 1:
            raise ValueError("invalid model capacity settings")
        self.max_options, self.max_length = config.max_options, config.max_length
        self.group_size, self.keep, self.seed = group_size, keep, seed
        self.batch_size, self.token_budget, self.debias = batch_size, token_budget, debias
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
            "candidate_orders": DEBIAS[debias],
            "policy": "Original state, instructions and all candidates, in order; no truncation, filtering, "
                      "translation or benchmark-specific prompt changes. A choice question with more than "
                      f"{self.max_options} candidates, or whose prompt exceeds the context window, is split "
                      f"into shuffled groups of at most {group_size} (seeded); each group's top {keep} "
                      "advance until the finalists fit one prompt. Eliminated candidates keep probabilities "
                      "below every survivor. A request is rejected whole if any prompt still does not fit."
                      + (" Every prompt is also read with its candidates reversed; probabilities are the mean "
                         "over both orders." if debias == "reverse" else ""),
        }

    @property
    def _pointer(self):
        return getattr(getattr(self.pipe, "model", None), "pointer_head", None) is not None

    def _orders(self, count):
        orders = [list(range(count))]
        if self.debias == "reverse":
            orders.append(orders[0][::-1])
        return orders

    def _encode(self, state, instructions, options):
        """One prompt per candidate order; `orders[k][j]` is the candidate shown at position j of prompt k."""
        processor = self.pipe.processor
        orders = self._orders(len(options))
        prompts = []
        for order in orders:
            shown = [options[i] for i in order]
            prompts.append({**processor.encode_options(state, instructions, shown), "options": shown,
                            "target": [1 / len(shown)] * len(shown)})
        return {"members": [{"options": options, "orders": orders}], "prompts": prompts,
                "length": max(len(x["input_ids"]) for x in prompts), "tokens": sum(len(x["input_ids"]) for x in prompts)}

    def _joint(self, state, questions):
        """All `[(instructions, options)]` in one pointer sequence per candidate order (orders[k] per question)."""
        processor = self.pipe.processor
        members = [{"options": options, "orders": self._orders(len(options))} for _, options in questions]
        prompts = []
        for k in range(len(members[0]["orders"])):
            shown = [(instructions, [options[i] for i in member["orders"][k]])
                     for (instructions, options), member in zip(questions, members)]
            prompt = processor.pointer_ids(state, shown)
            for entry, (_, options) in zip(prompt["questions"], shown):
                entry["target"] = [1 / len(options)] * len(options)
            prompts.append(prompt)
        return {"members": members, "prompts": prompts, "length": max(len(x["input_ids"]) for x in prompts),
                "tokens": sum(len(x["input_ids"]) for x in prompts)}

    def _round(self, state, match, name):
        """Encode the current round of a tournament; Unsupported if a group prompt is too long."""
        items = []
        for label, group in match.questions().items():
            item = self._encode(state, group["instructions"], options_for(group, self.max_options))
            if item["length"] > self.max_length:
                raise Unsupported(f"{name} ({label}): {item['length']} tokens exceed context window "
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

    def _rows(self, prompts):
        """Per prompt, the candidate logits of each question it holds (a letter or single prompt holds one)."""
        if not self._pointer:
            return [[row] for row in self.pipe.candidate_logits(prompts, batch_size=self.batch_size,
                                                                 token_budget=self.token_budget)]
        rows = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]["input_ids"]))
        with torch.inference_mode():
            for batch in shape_batches([dict(prompts[i], index=i) for i in order], self.batch_size, self.token_budget):
                logits, _ = self.pipe.model.score_decisions(batch, self.pipe.processor)
                if not torch.isfinite(logits).all().item():
                    raise FloatingPointError("non-finite decision logits")
                values, r = logits.float().cpu().tolist(), 0
                for item in batch:
                    rows[item["index"]] = []
                    for question in item["questions"]:
                        rows[item["index"]].append(values[r][:len(question["marks"])])
                        r += 1
        return rows

    def _probabilities(self, items):
        """Per item, each member question's probabilities averaged over its candidate orders; one batched pass."""
        rows = iter(self._rows([prompt for item in items for prompt in item["prompts"]]))
        result = []
        for item in items:
            per_prompt = [next(rows) for _ in item["prompts"]]
            for m, member in enumerate(item["members"]):
                mean = [0.] * len(member["options"])
                for k, order in enumerate(member["orders"]):
                    for position, p in zip(order, candidate_probabilities(per_prompt[k][m], self.pipe.temperature)):
                        mean[position] += p / len(member["orders"])
                result.append(mean)
        return result

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
            options = options_for(question, self.max_options)  # Malformed requests stay errors, not refusals.
            item = self._encode(state, question["instructions"], options)
            if item["length"] <= self.max_length:
                single[name] = (kind, options, item)
            elif kind == "choice" and len(options) > MIN_GROUP_SIZE:
                matches[name] = self._tournament(state, name, question)
            else:
                raise Unsupported(f"{name}: {item['length']} tokens exceed context window {self.max_length}")
        joint = None
        if self._pointer and len(single) > 1:
            joint = self._joint(state, [(questions[name]["instructions"], options) for name, (_, options, _) in single.items()])
            if joint["length"] > self.max_length:
                joint = None
        if joint is not None:
            tokens = joint["tokens"]
            pending_items, pending = [joint], list(single.items())
        else:
            tokens = sum(item["tokens"] for _, _, item in single.values())
            pending = list(single.items())
            pending_items = [item for _, (_, _, item) in pending]
        answers = {}
        rounds = {name: items for name, (_, items) in matches.items()}
        # One batched forward per step: every plain question, plus each open tournament's current round.
        while pending or rounds:
            batch = pending_items + [item for items in rounds.values() for item in items]
            tokens += sum(item["tokens"] for items in rounds.values() for item in items)
            values = self._probabilities(batch)
            for name, (kind, options, _) in pending:
                answers[name] = build_answer(kind, options, values.pop(0))
            pending, pending_items, following = [], [], {}
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
