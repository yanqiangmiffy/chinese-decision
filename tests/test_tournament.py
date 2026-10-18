import math
import re
import unittest
from argparse import Namespace
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from decision.tournament import MAX_CANDIDATES, from_request
from scripts import serve, serve_gpu

TARGET = "k137"
CANDIDATES = {f"k{i}": f"第{i}号产品" for i in range(300)}


def softmax(values):
    offset = max(values)
    weights = [math.exp(v - offset) for v in values]
    return [w / sum(weights) for w in weights]


def logits_for(keys):
    """Prefers TARGET wherever it appears, then lower-numbered keys, so every group has a strict order."""
    return [10. if key == TARGET else -int(re.sub(r"\D", "", key) or 0) / 100 for key in keys]


def play(match):
    asked = []
    while not match.done:
        questions = list(match.questions().values())
        asked.append([list(q["criteria"]) for q in questions])
        match.record([softmax(logits_for(q["criteria"])) for q in questions])
    return asked


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, prompt, **kwargs):
        return list(prompt.encode("utf-8"))


def prompt_logits(ids, count):
    """Reads the candidate keys back out of a compiled prompt (one `X: key — description` line each)."""
    keys = re.findall(r"^[A-Z]: (\S+) — ", bytes(ids).decode("utf-8"), re.M)
    assert len(keys) == count
    return logits_for(keys)


class PromptBackend:
    name, device, max_length = "gguf", "cpu", 4096
    tokenizer = Tokenizer()

    def __init__(self, *args, **kwargs):
        pass

    def logits(self, ids, count):
        return prompt_logits(ids, count)

    def close(self):
        pass


class InlineEngine(serve_gpu.Engine):
    """The GPU engine without its batching thread: every submission is answered immediately."""

    def _loop(self):
        pass

    def submit(self, items):
        future = Future()
        future.set_result([prompt_logits(item["input_ids"], item["k"]) for item in items])
        return future


def body(criteria=CANDIDATES, **options):
    return {"state": "用户希望了解第137号产品。",
            "question": {"type": "choice", "instructions": "用户想了解哪个产品？", "criteria": criteria}, **options}


class TournamentTests(unittest.TestCase):
    def test_a_field_that_fits_is_one_question_in_request_order(self):
        criteria = {key: CANDIDATES[key] for key in ("k5", TARGET, "k1")}
        match = from_request(body(criteria))
        self.assertEqual(play(match), [[["k5", TARGET, "k1"]]])
        self.assertEqual((match.rounds, match.decisions), ([], 1))
        self.assertEqual(match.answer["choice"], TARGET)
        self.assertEqual(list(match.answer["probabilities"]), ["k5", TARGET, "k1"])

    def test_large_field_keeps_groups_within_trained_size_and_finds_the_target(self):
        match = from_request(body())
        asked = play(match)
        self.assertEqual([len(r) for r in asked], [19, 3, 1])
        for groups in asked:
            self.assertTrue(all(2 <= len(g) <= 16 for g in groups))
        self.assertEqual(sorted(k for g in asked[0] for k in g), sorted(CANDIDATES))
        for previous, rounds in zip(asked, asked[1:]):
            self.assertEqual(sum(map(len, rounds)), 2 * len(previous))
        self.assertEqual(match.answer["choice"], TARGET)
        self.assertEqual(set(match.answer["probabilities"]), set(asked[-1][0]))
        self.assertEqual(match.decisions, 23)
        result = match.result()
        self.assertEqual((result["candidates"], result["group_size"], result["keep"]), (300, 16, 2))
        first = result["rounds"][0]["groups"]
        self.assertEqual([list(g["probabilities"]) for g in first], asked[0])
        self.assertTrue(all(len(g["advanced"]) == 2 for g in first))

    def test_smallest_groups_still_shrink_the_field_to_a_final(self):
        criteria = {f"k{i}": str(i) for i in range(MAX_CANDIDATES)}
        match = from_request(body(criteria, group_size=3, keep=1))
        asked = play(match)
        self.assertTrue(all(2 <= len(g) <= 3 for groups in asked for g in groups))
        self.assertEqual(len(asked[-1]), 1)
        self.assertEqual(match.answer["choice"], TARGET)

    def test_seed_makes_grouping_reproducible(self):
        first, again, other = (play(from_request(body(seed=seed))) for seed in (7, 7, 8))
        self.assertEqual(first, again)
        self.assertNotEqual(first[0], other[0])

    def test_invalid_requests(self):
        cases = [
            ({"question": body()["question"]}, "state"),
            (body(group_size=2), "group_size"),
            (body(group_size=27), "group_size"),
            (body(group_size=True), "group_size"),
            (body(keep=0), "keep"),
            (body(keep=9), "keep"),
            (body(seed=-1), "seed"),
            (body({"only": "一个"}), "2-1024"),
            (body({str(i): "x" for i in range(MAX_CANDIDATES + 1)}), "2-1024"),
            (body(["a", "b"]), "object"),
            ({**body(), "question": {"type": "score", "instructions": "几级？", "criteria": ["低", "高"]}}, "choice"),
            ({**body(), "question": {"type": "choice", "criteria": CANDIDATES}}, "instructions"),
        ]
        for request, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                from_request(request)

    def test_record_requires_the_current_round(self):
        match = from_request(body())
        with self.assertRaises(ValueError):
            match.record([[1.]])
        play(match)
        with self.assertRaises(ValueError):
            match.record([[1.]])


class TournamentRouteTests(unittest.TestCase):
    def clients(self):
        """The CPU and GPU servers with fake models that score candidates by reading the prompt."""
        args = Namespace(base="missing-config", adapter=None, token_budget=65536, max_batch=64, max_wait_ms=4,
                         run_dir=None, snapshots="snapshots", epoch_examples=1, admin_token="")
        with patch("scripts.serve.GGUFBackend", PromptBackend):
            cpu = serve.create_app("missing-config", backend="gguf", gguf="mock.gguf")
        with patch("scripts.serve_gpu.load_tokenizer", return_value=Tokenizer()), \
             patch("scripts.serve_gpu.load_model", return_value=MagicMock()), \
             patch("scripts.serve_gpu.letter_tokens"), patch("scripts.serve_gpu.Engine", InlineEngine):
            gpu = serve_gpu.create_app(args)
        return {"cpu": TestClient(cpu), "gpu": TestClient(gpu)}

    def test_both_servers_run_the_same_tournament(self):
        results = {}
        for name, client in self.clients().items():
            with self.subTest(server=name), client:
                result = client.post("/v1/tournament", json=body(seed=3)).json()
                info = client.get("/v1/info").json()
                self.assertEqual(result["answer"]["choice"], TARGET)
                self.assertEqual(result["decisions"], sum(len(r["groups"]) for r in result["rounds"]) + 1)
                self.assertEqual(result["usage"]["output_tokens"], 0)
                self.assertGreater(result["usage"]["input_tokens"], 0)
                self.assertEqual(info["limits"]["max_tournament_candidates"], MAX_CANDIDATES)
                self.assertEqual(info["stats"]["decisions"], result["decisions"])
                results[name] = result
        # Same groups and winners; probabilities differ in the last bits (float64 vs float32 softmax).
        advanced = {name: [[g["advanced"] for g in r["groups"]] for r in result["rounds"]]
                    for name, result in results.items()}
        self.assertEqual(advanced["cpu"], advanced["gpu"])
        self.assertEqual(list(results["cpu"]["answer"]["probabilities"]),
                         list(results["gpu"]["answer"]["probabilities"]))

    def test_a_small_tournament_matches_a_plain_choice_question(self):
        criteria = {key: CANDIDATES[key] for key in ("k3", TARGET, "k9", "k1")}
        for name, client in self.clients().items():
            with self.subTest(server=name), client:
                single = client.post("/v1/systemone", json={"state": body()["state"],
                                                            "questions": {"q": body(criteria)["question"]}}).json()
                result = client.post("/v1/tournament", json=body(criteria)).json()
                self.assertEqual(result["rounds"], [])
                self.assertEqual(result["answer"], single["answers"]["q"])
                self.assertEqual(result["usage"], single["usage"])

    def test_invalid_and_overlong_tournaments_return_422(self):
        for name, client in self.clients().items():
            with self.subTest(server=name), client:
                score = {**body(), "question": {"type": "score", "instructions": "几级？", "criteria": ["低", "高"]}}
                self.assertEqual(client.post("/v1/tournament", json=score).status_code, 422)
                overlong = client.post("/v1/tournament", json={**body(), "state": "长" * 6000})
                self.assertEqual(overlong.status_code, 422)
                self.assertIn("round 1 group", overlong.json()["detail"])


if __name__ == "__main__":
    unittest.main()
