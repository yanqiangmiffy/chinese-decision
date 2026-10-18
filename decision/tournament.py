"""Candidate tournaments: `choice` questions with more candidates than one prompt holds.

The model reads one letter per candidate (A-Z) and was trained on at most 16 candidates, so a larger
question is split into groups that each go through the unchanged prompt compiler. The leaders of every
group advance until the finalists fit into one question, whose probabilities are the answer.

Nothing here runs the model, so both servers share it:

    match = from_request(body)
    while not match.done:
        match.record([probabilities for each question in match.questions().values()])
    match.result()
"""
import random

from .api import build_answer
from .schema import LETTERS

# Largest candidate count in CJ-Dataset; letters beyond it were never trained as answers.
DEFAULT_GROUP_SIZE = 16
DEFAULT_KEEP = 2
MIN_GROUP_SIZE = 3
MAX_CANDIDATES = 1024


def _integer(body, name, default, low, high):
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def from_request(body):
    """Validate a /v1/tournament body: {state, question, group_size?, keep?, seed?}."""
    if not isinstance(body, dict) or body.get("state") is None:
        raise ValueError("state is required")
    question = body.get("question")
    if not isinstance(question, dict) or question.get("instructions") is None:
        raise ValueError("question requires non-null instructions")
    if question.get("type") != "choice":
        raise ValueError("tournaments split choice questions only; score levels are ordered and noul has two")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_CANDIDATES:
        raise ValueError(f"choice criteria must be an object with 2-{MAX_CANDIDATES} candidates")
    group_size = _integer(body, "group_size", DEFAULT_GROUP_SIZE, MIN_GROUP_SIZE, len(LETTERS))
    keep = _integer(body, "keep", DEFAULT_KEEP, 1, group_size // 2)
    seed = _integer(body, "seed", 0, 0, 2**32 - 1)
    return Tournament(body["state"], question["instructions"], criteria, group_size, keep, seed)


class Tournament:
    """Rounds of grouped `choice` questions; `keep <= group_size // 2` at least halves the field each round."""

    def __init__(self, state, instructions, criteria, group_size, keep, seed):
        self.state, self.instructions = state, instructions
        self.criteria, self.group_size, self.keep = criteria, group_size, keep
        self.rng = random.Random(seed)
        self.rounds, self.answer, self.decisions = [], None, 0
        self.groups = self._split(list(criteria))

    @property
    def done(self):
        return self.answer is not None

    def _split(self, keys):
        """One group in the given order when the field fits; otherwise shuffled, balanced groups of 2+."""
        if len(keys) <= self.group_size:
            return [keys]
        keys = keys[:]
        self.rng.shuffle(keys)
        count = -(-len(keys) // self.group_size)
        return [keys[i::count] for i in range(count)]

    def questions(self):
        """The current round, one `choice` question per group, named for error messages."""
        number = len(self.rounds) + 1
        return {f"round {number} group {g + 1}": {"type": "choice", "instructions": self.instructions,
                                                  "criteria": {key: self.criteria[key] for key in group}}
                for g, group in enumerate(self.groups)}

    def record(self, probabilities):
        """Take each current group's candidate probabilities (in group order) and advance."""
        if self.done or len(probabilities) != len(self.groups) \
                or any(len(p) != len(g) for p, g in zip(probabilities, self.groups)):
            raise ValueError("probabilities must match the current round's groups")
        self.decisions += len(self.groups)
        if len(self.groups) == 1:
            options = [(key, self.criteria[key]) for key in self.groups[0]]
            self.answer, self.groups = build_answer("choice", options, probabilities[0]), []
            return
        groups, advanced = [], []
        for group, values in zip(self.groups, probabilities):
            leaders = sorted(range(len(group)), key=values.__getitem__, reverse=True)[:self.keep]
            groups.append({"probabilities": dict(zip(group, values)), "advanced": [group[i] for i in leaders]})
            advanced += groups[-1]["advanced"]
        self.rounds.append({"groups": groups})
        self.groups = self._split(advanced)

    def result(self):
        """The final question's answer (probabilities cover the finalists only) and every elimination round."""
        return {"answer": self.answer, "candidates": len(self.criteria), "group_size": self.group_size,
                "keep": self.keep, "rounds": self.rounds}
