import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from decision.processing import options_for

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_unified import create_app  # noqa: E402

QUESTION = {"type": "choice", "instructions": "选哪个？", "criteria": {"A": "甲", "B": "乙"}}


def client(can_extract=True):
    pipe = Mock(can_extract=can_extract)
    pipe.model = SimpleNamespace(config=SimpleNamespace(temperature=1.1, max_length=16384))
    pipe.decide.return_value = {"q": {"A": 0.8, "B": 0.2}}
    pipe.extract.return_value = [{"text": "张三", "label": "人名", "start": 0, "end": 2, "score": 0.9}]
    pipe.decide_and_extract.return_value = (pipe.decide.return_value, pipe.extract.return_value)
    return TestClient(create_app(pipe, "test")), pipe


class FakeProcessor:
    """Encodes a question into ids whose count is the length of its instructions."""

    def encode_question(self, state, question):
        options = options_for(question)
        return {"input_ids": list(range(len(str(question["instructions"])))), "options": options,
                "target": [1 / len(options)] * len(options), "kind": question["type"]}

    def build_prompt(self, state, instructions, options):
        return f"{state}|{instructions}|" + ",".join(key for key, _ in options)


class FakePipe:
    """Scores the first candidate highest by 2 logits: probabilities are known without a model."""

    can_extract = True
    processor = FakeProcessor()
    model = SimpleNamespace(config=SimpleNamespace(temperature=1.0, max_length=16384))

    def candidate_logits(self, items, batch_size=128, token_budget=32768):
        return [[2.0] + [0.0] * (len(item["options"]) - 1) for item in items]


def gpu_style():
    return TestClient(create_app(FakePipe(), "fake", {"step": 15304, "calibrated": True, "base": "models/Qwen3.5-2B"}))


class ServeGpuApiTests(unittest.TestCase):
    """The decision API of scripts/serve_gpu.py, served by the unified model."""

    def test_systemone_answers_in_the_system_one_format_and_debug_adds_logits(self):
        http = gpu_style()
        body = {"state": "s", "questions": {"q": QUESTION, "n": {"type": "noul", "instructions": "是吗？"}}}
        plain = http.post("/v1/systemone", json=body).json()
        self.assertEqual((plain["model"], plain["usage"]["output_tokens"], plain["answers"]["q"]["choice"]), ("fake", 0, "A"))
        self.assertNotIn("logits", plain["answers"]["q"])
        self.assertEqual(plain["usage"]["input_tokens"], len("选哪个？") + len("是吗？"))
        debug = http.post("/api/alpha/decisions?debug=1", json=body).json()["answers"]["q"]
        self.assertEqual((debug["logits"], debug["input_tokens"]), ({"A": 2.0, "B": 0.0}, 4))
        self.assertAlmostEqual(debug["probabilities"]["A"], 0.8808, places=4)

    def test_systemone_rejects_bad_requests_with_422(self):
        http = gpu_style()
        for body in ({"questions": {"q": QUESTION}}, {"state": "s"}, {"state": "s", "questions": {}},
                     {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"A": "甲"}}}}):
            self.assertEqual(http.post("/v1/systemone", json=body).status_code, 422, body)

    def test_batch_keeps_one_result_per_request_and_isolates_errors(self):
        http = gpu_style()
        body = {"requests": [{"state": "a", "questions": {"q": QUESTION}}, {"state": "b"},
                             {"state": "c", "questions": {"q": QUESTION, "r": QUESTION}}], "debug": True}
        out = http.post("/v1/batch", json=body).json()
        self.assertEqual([("error" in r, len(r.get("answers", {}))) for r in out["results"]], [(False, 1), (True, 0), (False, 2)])
        self.assertEqual((out["decisions"], out["usage"]["output_tokens"], "logits" in out["results"][0]["answers"]["q"]), (3, 0, True))
        self.assertEqual(http.post("/v1/batch", json={"requests": []}).status_code, 422)

    def test_tournament_splits_a_large_choice_question(self):
        http = gpu_style()
        criteria = {f"c{i}": f"候选{i}" for i in range(40)}
        out = http.post("/v1/tournament", json={"state": "s", "question": {"type": "choice", "instructions": "选哪个？",
                                                                         "criteria": criteria}}).json()
        self.assertEqual((out["candidates"], len(out["rounds"]), out["decisions"]), (40, 1, 4))
        self.assertEqual(len(out["answer"]["probabilities"]), 6)
        self.assertEqual(http.post("/v1/tournament", json={"state": "s", "question": {**QUESTION, "type": "score"}}).status_code, 422)

    def test_compile_returns_the_prompt_letters_and_token_counts(self):
        out = gpu_style().post("/v1/compile", json={"state": "s", "questions": {"q": QUESTION}}).json()
        question = out["questions"]["q"]
        self.assertEqual((question["prompt"], question["input_tokens"]), ("s|选哪个？|A,B", 4))
        self.assertEqual([o["letter"] for o in question["options"]], ["A", "B"])

    def test_info_and_health_carry_what_studio_reads(self):
        http = gpu_style()
        http.post("/v1/systemone", json={"state": "s", "questions": {"q": QUESTION}})
        info = http.get("/v1/info").json()
        self.assertEqual((info["checkpoint"]["step"], info["checkpoint"]["calibrated"], info["can_extract"]), (15304, True, True))
        self.assertEqual((info["stats"]["requests"], info["stats"]["decisions"], info["limits"]["max_candidates"]), (1, 1, 26))
        health = http.get("/health").json()
        self.assertEqual((health["step"], health["calibrated"], health["can_extract"]), (15304, True, True))
        self.assertEqual(http.get("/v1/training").json(), {"available": False})


class ServeUnifiedTests(unittest.TestCase):
    def test_decide_returns_the_system_one_answer_format(self):
        http, pipe = client()
        body = http.post("/v1/decide", json={"state": "s", "question": QUESTION}).json()
        answer = body["answers"]["q"]
        self.assertEqual((body["task"], answer["choice"], answer["probabilities"]), ("decision", "A", {"A": 0.8, "B": 0.2}))
        pipe.decide.assert_called_once_with("s", {"q": QUESTION})

    def test_extract_and_predict_route_by_request_shape(self):
        http, pipe = client()
        body = http.post("/v1/predict", json={"text": "张三来了", "types": {"人名": "人物的姓名"}, "threshold": 0.3}).json()
        self.assertEqual((body["task"], body["entities"][0]["text"]), ("extraction", "张三"))
        pipe.extract.assert_called_once_with("张三来了", {"人名": "人物的姓名"}, 0.3)
        self.assertEqual(http.post("/v1/predict", json={"state": "s", "questions": {"q": QUESTION}}).json()["task"], "decision")

    def test_predict_with_questions_and_types_answers_both_about_one_input(self):
        http, pipe = client()
        body = http.post("/v1/predict", json={"state": "张三来了", "questions": {"q": QUESTION},
                                              "types": {"人名": "人物的姓名"}, "threshold": 0.3}).json()
        self.assertEqual((body["task"], body["answers"]["q"]["choice"], body["text"], body["entities"][0]["text"]),
                         ("decision+extraction", "A", "张三来了", "张三"))
        pipe.decide_and_extract.assert_called_once_with("张三来了", {"q": QUESTION}, "张三来了", {"人名": "人物的姓名"},
                                                        0.3, token_budget=32768)
        pipe.decide.assert_not_called()
        pipe.extract.assert_not_called()
        self.assertEqual(http.get("/v1/info").json()["requests"]["both"], 1)

    def test_bad_requests_are_400_and_counted(self):
        http, _ = client()
        self.assertEqual(http.post("/v1/predict", json={"types": ["人名"], "question": QUESTION}).status_code, 400)
        self.assertEqual(http.post("/v1/decide", json={"state": "s"}).status_code, 400)
        self.assertEqual(http.post("/v1/extract", json={"text": "x", "types": []}).status_code, 400)
        self.assertEqual(client(can_extract=False)[0].post("/v1/extract", json={"text": "x", "types": ["a"]}).status_code, 400)
        self.assertEqual(http.get("/v1/info").json()["requests"]["errors"], 3)
        self.assertEqual(http.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
