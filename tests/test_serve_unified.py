import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_unified import create_app  # noqa: E402

QUESTION = {"type": "choice", "instructions": "选哪个？", "criteria": {"A": "甲", "B": "乙"}}


def client(can_extract=True):
    pipe = Mock(can_extract=can_extract)
    pipe.model = SimpleNamespace(config=SimpleNamespace(temperature=1.1, max_length=16384))
    pipe.decide.return_value = {"q": {"A": 0.8, "B": 0.2}}
    pipe.extract.return_value = [{"text": "张三", "label": "人名", "start": 0, "end": 2, "score": 0.9}]
    return TestClient(create_app(pipe, "test")), pipe


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

    def test_bad_requests_are_400_and_counted(self):
        http, _ = client()
        self.assertEqual(http.post("/v1/predict", json={"text": "x", "types": ["人名"], "question": QUESTION}).status_code, 400)
        self.assertEqual(http.post("/v1/decide", json={"state": "s"}).status_code, 400)
        self.assertEqual(http.post("/v1/extract", json={"text": "x", "types": []}).status_code, 400)
        self.assertEqual(client(can_extract=False)[0].post("/v1/extract", json={"text": "x", "types": ["a"]}).status_code, 400)
        self.assertEqual(http.get("/v1/info").json()["requests"]["errors"], 2)
        self.assertEqual(http.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
