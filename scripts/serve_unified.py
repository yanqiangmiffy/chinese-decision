"""HTTP server for a unified model directory: decisions and span extraction from one set of weights.

    CUDA_VISIBLE_DEVICES=2 python scripts/serve_unified.py --model runs/cj-ie-0.8b-from-base/model --port 8011

One service for both tasks. It also speaks the decision API of scripts/serve_gpu.py, so a client written
for that service (Decision Studio, the hmm / System One SDKs) works against this one unchanged and a second
service is not needed:

    GET  /health
    GET  /v1/info
    POST /v1/systemone        decisions with the hmm / System One response (?debug=1 adds logits and token counts)
    POST /api/alpha/decisions same
    POST /v1/batch            {"requests": [{state, questions}, ...], "debug": bool}
    POST /v1/compile          the prompts the questions compile to
    POST /v1/tournament       a choice question with more candidates than one prompt holds
    POST /v1/decide   {"state": ..., "questions": {"q": {"type", "instructions", "criteria"}}}  (or "question")
    POST /v1/extract  {"text": "...", "types": ["人名", ...] | {"人名": "人物的姓名", ...}, "threshold": 0.0}
    POST /v1/predict  either shape; `questions`/`question` -> decision, `types` -> extraction

Decision answers use the hmm / System One format (`probabilities`, `confidence`, `choice` / `noul` / `score`),
calibrated with the model's own temperature. One forward pass runs at a time (a lock around the GPU); requests
are served in a thread pool. Request errors are 400 on /v1/decide, /v1/extract, /v1/predict and 422 on the
serve_gpu-compatible routes, as in the services they mirror.
"""
import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from decision.configuration import LETTERS, PROMPT_VERSION
from decision.pipeline import DecisionPipeline
from decision.processing import options_for
from decision.serving.api import build_answer
from decision.serving.backends import candidate_probabilities
from decision.serving.tournament import MAX_CANDIDATES
from decision.serving.tournament import from_request as tournament_from

MAX_QUESTIONS, MAX_BATCH_REQUESTS, MAX_BATCH_QUESTIONS = 64, 512, 4096
TOKEN_BUDGET, MAX_BATCH = 32768, 128   # per forward pass, see DecisionPipeline.candidate_logits


class RequestError(ValueError):
    pass


class Stats:
    """Counters and recent latencies for /v1/info (the shape the serve_gpu clients read)."""

    def __init__(self):
        self.started = time.time()
        self.kinds = {"decide": 0, "extract": 0, "errors": 0}
        self.requests = self.decisions = self.input_tokens = 0
        self.latencies = deque(maxlen=4096)

    def record(self, kind, latency_ms, decisions=0, tokens=0):
        self.kinds[kind] += 1
        self.requests += 1
        self.decisions += decisions
        self.input_tokens += tokens
        self.latencies.append(latency_ms)

    def percentile(self, q):
        values = sorted(self.latencies)
        return round(values[min(len(values) - 1, int(q * len(values)))], 1) if values else None

    def describe(self):
        return {"requests": self.requests, "decisions": self.decisions, "input_tokens": self.input_tokens,
                "batches": self.requests, "forward_passes": self.requests, "errors": self.kinds["errors"],
                "uptime_s": round(time.time() - self.started), "queue_depth": 0,
                "latency_p50_ms": self.percentile(0.5), "latency_p95_ms": self.percentile(0.95),
                "latency_p99_ms": self.percentile(0.99), "avg_forward_batch": None, "forward_p50_ms": None,
                "compile_p50_ms": None, "recent_forwards": []}


def gpu_info():
    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    return {"name": torch.cuda.get_device_name(index),
            "physical_index": os.environ.get("CUDA_VISIBLE_DEVICES", str(index)).split(",")[0],
            "memory_allocated_mb": round(torch.cuda.memory_allocated(index) / 2**20),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(index) / 2**20),
            "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(index) / 2**20),
            "device_free_mb": round(free / 2**20), "device_total_mb": round(total / 2**20)}


def model_meta(model_dir, config):
    """What /health and /v1/info report about the loaded weights, from the directory's training_summary.json."""
    summary = {}
    path = Path(model_dir) / "training_summary.json"
    if path.is_file():
        summary = json.loads(path.read_text(encoding="utf-8"))
    selected = summary.get("selected") or {}
    checkpoint = str(selected.get("checkpoint", ""))
    step = int(checkpoint.rsplit("-", 1)[1]) if checkpoint.rsplit("-", 1)[-1].isdigit() else None
    calibrated = bool(getattr(config, "calibration_hard_n", 0)) or abs(float(config.temperature) - 1.0) > 1e-9
    return {"base": summary.get("base"), "step": step, "calibrated": calibrated,
            "selection_metric": "ie_score" if "ie_score" in selected else None,
            "selection_value": selected.get("ie_score"), "validation": {}}


def create_app(pipe, model_name="decision", meta=None):
    meta = meta or {}
    app = FastAPI(title="chinese-decision unified", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    lock, stats, loaded_at = threading.Lock(), Stats(), time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def temperature():
        return float(pipe.model.config.temperature)

    def run(kind, work, status=400):
        try:
            with lock:
                begin = time.perf_counter()
                result = work()
            latency = round(1000 * (time.perf_counter() - begin), 1)
            usage = result.get("usage") or {}
            stats.record(kind, latency, result.get("decisions", len(result.get("answers") or {})),
                         usage.get("input_tokens", 0))
            return result | {"latency_ms": latency}
        except (ValueError, KeyError, TypeError) as error:
            stats.kinds["errors"] += 1
            raise HTTPException(status_code=status, detail=f"{type(error).__name__}: {error}") from error

    # ------------------------------------------------------------------ decisions
    def encode(state, questions):
        """[(name, question, item)] for questions about one state; errors name the question."""
        encoded = []
        for name, question in questions.items():
            try:
                encoded.append((name, question, pipe.processor.encode_question(state, question)))
            except (KeyError, TypeError, ValueError) as error:
                raise RequestError(f"question {name}: {error}") from error
        return encoded

    def encode_request(body):
        if not isinstance(body, dict) or body.get("state") is None:
            raise RequestError("state is required")
        questions = body.get("questions") or ({"q": body["question"]} if "question" in body else None)
        if not isinstance(questions, dict) or not questions:
            raise RequestError("questions must be a nonempty object")
        if len(questions) > MAX_QUESTIONS:
            raise RequestError(f"at most {MAX_QUESTIONS} questions per request")
        return encode(body["state"], questions)

    def response_for(encoded, logits, debug):
        answers = {}
        for (name, question, item), row in zip(encoded, logits, strict=True):
            answer = build_answer(question["type"], item["options"], candidate_probabilities(row, temperature()))
            if debug:
                answer["logits"] = {key: round(value, 4) for (key, _), value in zip(item["options"], row, strict=True)}
                answer["input_tokens"] = len(item["input_ids"])
            answers[name] = answer
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": sum(len(item["input_ids"]) for _, _, item in encoded), "output_tokens": 0}}

    def system_one(body, debug):
        encoded = encode_request(body)
        return response_for(encoded, pipe.candidate_logits([item for _, _, item in encoded], MAX_BATCH, TOKEN_BUDGET), debug)

    def batch(body):
        requests = body.get("requests")
        if not isinstance(requests, list) or not requests:
            raise RequestError("requests must be a nonempty list")
        if len(requests) > MAX_BATCH_REQUESTS:
            raise RequestError(f"at most {MAX_BATCH_REQUESTS} requests per batch")
        encoded = []
        for request in requests:
            try:
                encoded.append(encode_request(request))
            except RequestError as error:
                encoded.append(error)
        total = sum(len(e) for e in encoded if not isinstance(e, Exception))
        if total > MAX_BATCH_QUESTIONS:
            raise RequestError(f"at most {MAX_BATCH_QUESTIONS} questions per batch")
        items = [item for e in encoded if not isinstance(e, Exception) for _, _, item in e]
        logits = pipe.candidate_logits(items, MAX_BATCH, TOKEN_BUDGET) if items else []
        results, cursor, tokens = [], 0, 0
        for e in encoded:
            if isinstance(e, Exception):
                results.append({"error": str(e)})
                continue
            response = response_for(e, logits[cursor:cursor + len(e)], bool(body.get("debug")))
            cursor += len(e)
            tokens += response["usage"]["input_tokens"]
            results.append(response)
        return {"model": model_name, "results": results, "usage": {"input_tokens": tokens, "output_tokens": 0},
                "decisions": total}

    def tournament(body):
        match, tokens = tournament_from(body), 0
        while not match.done:
            encoded = encode(match.state, match.questions())
            items = [item for _, _, item in encoded]
            tokens += sum(len(item["input_ids"]) for item in items)
            match.record([candidate_probabilities(row, temperature())
                          for row in pipe.candidate_logits(items, MAX_BATCH, TOKEN_BUDGET)])
        return {"model": model_name, **match.result(), "usage": {"input_tokens": tokens, "output_tokens": 0},
                "decisions": match.decisions}

    def compile_only(body):
        encoded = encode_request(body)
        return {"prompt_version": PROMPT_VERSION, "questions": {
            name: {"prompt": pipe.processor.build_prompt(body["state"], question["instructions"], item["options"]),
                   "input_tokens": len(item["input_ids"]),
                   "options": [{"letter": LETTERS[i], "key": key, "description": description}
                               for i, (key, description) in enumerate(item["options"])]}
            for name, question, item in encoded}}

    # the first version of this server: /v1/decide takes the same request, answers with the same format
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

    # ------------------------------------------------------------------ status
    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "temperature": temperature(),
                "calibrated": meta.get("calibrated", False), "step": meta.get("step"), "can_extract": pipe.can_extract}

    @app.get("/v1/info")
    def info():
        config = pipe.model.config
        return {"model": model_name, "version": app.version, "prompt_version": PROMPT_VERSION,
                "can_extract": pipe.can_extract, "temperature": temperature(), "max_length": config.max_length,
                "limits": {"max_length": config.max_length, "max_questions": MAX_QUESTIONS, "max_candidates": len(LETTERS),
                           "max_batch_requests": MAX_BATCH_REQUESTS, "max_tournament_candidates": MAX_CANDIDATES},
                "checkpoint": {"base": meta.get("base"), "adapter": None, "step": meta.get("step"),
                               "selection_metric": meta.get("selection_metric"),
                               "selection_value": meta.get("selection_value"), "temperature": temperature(),
                               "calibrated": meta.get("calibrated", False), "max_length": config.max_length,
                               "loaded_at": loaded_at, "load_seconds": meta.get("load_seconds", 0)},
                "validation": meta.get("validation", {}),
                "engine": {"token_budget": TOKEN_BUDGET, "max_batch": MAX_BATCH, "max_wait_ms": 0},
                "stats": stats.describe(), "gpu": gpu_info(), "latest_best_step": None,
                "uptime_seconds": round(time.time() - stats.started), "requests": dict(stats.kinds)}

    @app.get("/v1/training")
    def training():
        return {"available": False}

    # ------------------------------------------------------------------ routes
    @app.post("/v1/systemone")
    @app.post("/api/alpha/decisions")
    def post_system_one(request: Request, body: dict = Body(...)):
        debug = request.query_params.get("debug", "").lower() in ("1", "true", "yes")
        return run("decide", lambda: system_one(body, debug), 422)

    @app.post("/v1/batch")
    def post_batch(body: dict = Body(...)):
        return run("decide", lambda: batch(body), 422)

    @app.post("/v1/tournament")
    def post_tournament(body: dict = Body(...)):
        return run("decide", lambda: tournament(body), 422)

    @app.post("/v1/compile")
    def post_compile(body: dict = Body(...)):
        return run("decide", lambda: compile_only(body), 422)

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
    started = time.perf_counter()
    pipe = DecisionPipeline.from_pretrained(args.model)
    name = args.name or pipe.model.config.model_name or args.model
    meta = model_meta(args.model, pipe.model.config) | {"load_seconds": round(time.perf_counter() - started, 1)}
    uvicorn.run(create_app(pipe, name, meta), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
