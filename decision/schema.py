"""One shared input compiler for training, evaluation, and serving.

The option-letter objective and prompt layout derive from hmm/training/train.py.
CJ labels are distributions, unlike hmm's original scalar gold labels.
"""
import hashlib
import json
import math
from dataclasses import dataclass

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT_VERSION = "hmm-zh-v1"
# Longer inputs are rejected everywhere; context and questions are never truncated.
MAX_LENGTH = 16384


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def options_for(question):
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
    return str(meta.get("parent_id") or meta.get("group_key") or meta.get("original_id") or case.get("id") or text(case["state"]))


def partition(case):
    return int(hashlib.sha256(group_key(case).encode()).hexdigest()[:8], 16) % 5


def user_prompt(state, instructions, options):
    return ("待判断的上下文：\n" + text(state)
            + "\n\n问题：\n" + text(instructions)
            + "\n\n候选项：\n" + "\n".join(f"{LETTERS[i]}: {k} — {v}" for i, (k, v) in enumerate(options))
            + "\n只返回候选项对应的字母。")


def compile_prompt(tokenizer, state, instructions, options):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt(state, instructions, options)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


@dataclass
class Example:
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
        order = list(range(len(self.options)))
        if self.kind == "choice" and rng is not None:
            rng.shuffle(order)
        return (compile_prompt(tokenizer, self.state, self.instructions, [self.options[i] for i in order]),
                [self.target[i] for i in order])


def examples_for(case, domain):
    meta = case.get("_meta") or {}
    for name, question in case["questions"].items():
        options = options_for(question)
        target = target_for(case["gold"][name], options, question["type"])
        yield Example(case["state"], question["instructions"], options, target, question["type"], domain,
                      str(meta.get("source_family") or meta.get("source_id") or meta.get("source") or "unknown"),
                      str(case.get("id", "")) + ":" + name)
