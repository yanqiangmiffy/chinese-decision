"""HTTP server for a unified model directory: decisions and span extraction from one set of weights.

    CUDA_VISIBLE_DEVICES=2 python scripts/serve_unified.py --model runs/cj-ie-0.8b-from-base/model --port 8011

    GET  /health
    GET  /v1/info
    POST /v1/decide   {"state": ..., "questions": {"q": {"type", "instructions", "criteria"}}}  (or "question")
    POST /v1/extract  {"text": "...", "types": ["人名", ...] | {"人名": "人物的姓名", ...}, "threshold": 0.0}
    POST /v1/predict  either shape; `questions`/`question` -> decision, `types` -> extraction

Decision answers use the hmm / System One format of scripts/serve_gpu.py (`probabilities`, `confidence`,
`choice` / `noul` / `score`), calibrated with the model's own temperature. One forward pass runs at a
time (a lock around the GPU); requests are served in a thread pool.
"""
import argparse
import threading
import time

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from decision.pipeline import DecisionPipeline
from decision.processing import options_for
from decision.serving.api import build_answer


def create_app(pipe, model_name="decision"):
    app = FastAPI(title="chinese-decision unified", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    lock, stats, started = threading.Lock(), {"decide": 0, "extract": 0, "errors": 0}, time.time()

    def run(kind, work):
        try:
            with lock:
                begin = time.perf_counter()
                result = work()
            stats[kind] += 1
            return result | {"latency_ms": round(1000 * (time.perf_counter() - begin), 1)}
        except (ValueError, KeyError, TypeError) as error:
            stats["errors"] += 1
            raise HTTPException(status_code=400, detail=f"{type(error).__name__}: {error}") from error

    def decide(body):
        questions = body.get("questions") or ({"q": body["question"]} if "question" in body else None)
        if "state" not in body or not questions:
            raise ValueError("a decision request needs `state` and `questions` (or `question`)")
        answers = {}
        for name, probabilities in pipe.decide(body["state"], questions).items():
            question = questions[name]
            options = options_for(question)
            answers[name] = build_answer(question["type"], options, [probabilities[key] for key, _ in options])
        return {"task": "decision", "model": model_name, "answers": answers}

    def extract(body):
        if "text" not in body or "types" not in body:
            raise ValueError("an extraction request needs `text` and `types`")
        if not pipe.can_extract:
            raise ValueError("this model has no span head")
        types = body["types"]
        if not types or not isinstance(types, (list, dict)):
            raise ValueError("`types` must be a non-empty list of names or {name: description}")
        entities = pipe.extract(body["text"], types, float(body.get("threshold", 0.0)))
        return {"task": "extraction", "model": model_name, "text": body["text"], "entities": entities}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/info")
    def info():
        config = pipe.model.config
        return {"model": model_name, "temperature": config.temperature, "can_extract": pipe.can_extract,
                "max_length": config.max_length, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "uptime_seconds": round(time.time() - started), "requests": dict(stats)}

    @app.post("/v1/decide")
    def post_decide(body: dict = Body(...)):
        return run("decide", lambda: decide(body))

    @app.post("/v1/extract")
    def post_extract(body: dict = Body(...)):
        return run("extract", lambda: extract(body))

    @app.post("/v1/predict")
    def post_predict(body: dict = Body(...)):
        if "types" in body and ("questions" in body or "question" in body):
            raise HTTPException(status_code=400, detail="send either questions (decision) or types (extraction), not both")
        if "types" in body:
            return run("extract", lambda: extract(body))
        return run("decide", lambda: decide(body))

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="exported model directory")
    parser.add_argument("--name", default="", help="name reported in responses (default: the directory name)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    pipe = DecisionPipeline.from_pretrained(args.model)
    name = args.name or pipe.model.config.model_name or args.model
    uvicorn.run(create_app(pipe, name), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
