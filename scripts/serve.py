"""hmm / System One-compatible HTTP API: one forward pass per question, no generated tokens.

Also serves the routes Decision Studio uses (/v1/batch, /v1/compile, /v1/info, /v1/training) and
/v1/tournament (choice questions with more candidates than one prompt holds) with the same request and
response shapes as scripts/serve_gpu.py, so either can back the platform. Questions are
evaluated one at a time; for dynamic batching and checkpoint hot reload on a GPU, use serve_gpu.py.
"""
import argparse
import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from decision.api import build_answer
from decision.inference import GGUFBackend, TransformersBackend, candidate_probabilities
from decision.schema import LETTERS, MAX_LENGTH, PROMPT_VERSION, compile_prompt, options_for
from decision.tournament import MAX_CANDIDATES as MAX_TOURNAMENT_CANDIDATES
from decision.tournament import from_request as tournament_from
from decision.utils import model_name_for

MAX_QUESTIONS = 64
MAX_BATCH_REQUESTS = 512
MAX_BATCH_QUESTIONS = 4096


class RequestError(ValueError):
    """An invalid decision request; its message is returned to the client."""


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def iso_mtime(path):
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


class Stats:
    """Rolling serving statistics in the shape Decision Studio reads from /v1/info."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.counters = dict.fromkeys(("requests", "decisions", "input_tokens", "forward_passes", "errors"), 0)
        self.latencies = deque(maxlen=2000)  # end-to-end ms per HTTP request
        self.compile_ms = deque(maxlen=2000)  # prompt compilation and tokenization per HTTP request
        self.forwards = deque(maxlen=500)  # recent forward passes; each evaluates exactly one prompt

    def compiled(self, ms):
        with self.lock:
            self.compile_ms.append(ms)

    def forward(self, tokens, ms):
        with self.lock:
            self.counters["forward_passes"] += 1
            self.forwards.append({"items": 1, "padded_len": tokens, "ms": round(ms, 2)})

    def error(self):
        with self.lock:
            self.counters["errors"] += 1

    def record(self, latency_ms, decisions, tokens, requests=1):
        with self.lock:
            self.counters["requests"] += requests
            self.counters["decisions"] += decisions
            self.counters["input_tokens"] += tokens
            self.latencies.append(latency_ms)

    def snapshot(self):
        with self.lock:
            counters, latencies = dict(self.counters), sorted(self.latencies)
            compile_ms, forwards = sorted(self.compile_ms), list(self.forwards)
        forward_ms = sorted(f["ms"] for f in forwards)

        def percentile(values, q):
            return round(values[min(len(values) - 1, int(q * len(values)))], 2) if values else None

        return {**counters, "uptime_s": round(time.time() - self.started_at, 1),
                "latency_p50_ms": percentile(latencies, .5), "latency_p95_ms": percentile(latencies, .95),
                "latency_p99_ms": percentile(latencies, .99), "avg_forward_batch": 1. if forwards else None,
                "forward_p50_ms": percentile(forward_ms, .5), "compile_p50_ms": percentile(compile_ms, .5),
                "recent_forwards": forwards[-40:]}


def create_app(base, adapter=None, *, backend="transformers", gguf=None, device="cuda", threads=8,
               context_size=MAX_LENGTH):
    started = time.perf_counter()
    root = Path(base)
    config_path = root / "decision_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    model_name = model_name_for(base, config)
    temperature = config.get("temperature", 1.)
    max_length = config.get("max_length", MAX_LENGTH)
    candidate_probabilities([0., 0.], temperature)
    if config.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
        raise ValueError("saved model uses a different prompt compiler")
    if backend == "gguf":
        if adapter or not gguf:
            raise ValueError("GGUF backend requires --gguf and does not accept --adapter")
        engine = GGUFBackend(base, gguf, threads=threads, context_size=context_size)
    elif backend == "transformers":
        if gguf:
            raise ValueError("--gguf requires --backend gguf")
        engine = TransformersBackend(base, adapter, device=device, threads=threads)
    else:
        raise ValueError(f"unknown backend: {backend}")
    if engine.max_length is not None:
        max_length = min(max_length, engine.max_length)
    tokenizer = engine.tokenizer
    lock = threading.Lock()
    stats = Stats()
    # Optional release metadata next to the weights; older exports and tests have none.
    summary = read_json(root / "training_summary.json")
    validation = read_json(root / "validation_report.json").get("after", {})
    selected = summary.get("selected_checkpoint", {})
    checkpoint = {"base": str(base), "adapter": str(adapter) if adapter else None, "gguf": str(gguf) if gguf else None,
                  "backend": engine.name, "device": engine.device, "step": selected.get("step"),
                  "selection_metric": selected.get("metric"), "selection_value": selected.get("value"),
                  "temperature": temperature, "calibrated": config_path.exists(), "max_length": max_length,
                  "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "load_seconds": round(time.perf_counter() - started, 2)}

    def compile_request(body):
        """Validate and compile one request into [(name, question, options, input_ids, prompt)]."""
        if not isinstance(body, dict) or body.get("state") is None or not isinstance(body.get("questions"), dict) \
                or not body["questions"]:
            raise RequestError("state and nonempty questions are required")
        if len(body["questions"]) > MAX_QUESTIONS:
            raise RequestError(f"at most {MAX_QUESTIONS} questions per request")
        return compile_questions(body["state"], body["questions"])

    def compile_questions(state, questions):
        """Compile any number of questions about one state."""
        compiled = []
        for name, question in questions.items():
            try:
                options = options_for(question)
                prompt = compile_prompt(tokenizer, state, question["instructions"], options)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) > max_length:
                    raise ValueError(f"input has {len(ids)} tokens; limit {max_length}; input was not truncated")
                if hasattr(engine, "validate_tokens"):
                    engine.validate_tokens(prompt, ids)
            except (KeyError, TypeError, ValueError) as exc:
                raise RequestError(f"question {name}: {exc}") from exc
            compiled.append((name, question, options, ids, prompt))
        return compiled

    def evaluate(compiled):
        """Candidate logits per question. Questions run serially, which bounds memory; the lock is taken per
        question so a long batch does not hold up single requests until it finishes."""
        rows = []
        for _, _, options, ids, _ in compiled:
            with lock:
                begun = time.perf_counter()
                try:
                    rows.append(engine.logits(ids, len(options)))
                except Exception:
                    stats.error()
                    raise
                elapsed = time.perf_counter() - begun
            stats.forward(len(ids), 1000 * elapsed)
        return rows

    def respond(compiled, rows, latency_ms, debug):
        answers = {}
        for (name, question, options, ids, _), logits in zip(compiled, rows):
            answer = build_answer(question["type"], options, candidate_probabilities(logits, temperature))
            if debug:
                answer["logits"] = {key: round(value, 4) for (key, _), value in zip(options, logits)}
                answer["input_tokens"] = len(ids)
            answers[name] = answer
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": sum(len(c[3]) for c in compiled), "output_tokens": 0},
                "latency_ms": latency_ms}

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title=model_name, version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "temperature": temperature,
                "calibrated": bool(config_path.exists()), "backend": engine.name,
                "device": engine.device, "max_length": max_length, "step": checkpoint["step"]}

    @app.post("/api/alpha/decisions")
    @app.post("/v1/systemone")
    def decide(body: dict, debug: str = ""):
        """`?debug=1` adds each candidate's raw logit and each question's input token count."""
        started = time.perf_counter()
        try:
            compiled = compile_request(body)
        except RequestError as exc:
            raise HTTPException(422, str(exc)) from exc
        stats.compiled(1000 * (time.perf_counter() - started))
        rows = evaluate(compiled)
        latency = 1000 * (time.perf_counter() - started)
        response = respond(compiled, rows, latency, debug.lower() in ("1", "true", "yes"))
        stats.record(latency, len(compiled), response["usage"]["input_tokens"])
        return response

    @app.post("/v1/batch")
    def batch(body: dict):
        """{"requests": [...], "debug": bool}: one result per request; an invalid request yields
        {"error": ...} in its slot without failing the others."""
        started = time.perf_counter()
        requests = body.get("requests")
        if not isinstance(requests, list) or not requests:
            raise HTTPException(422, "requests must be a nonempty list")
        if len(requests) > MAX_BATCH_REQUESTS:
            raise HTTPException(422, f"at most {MAX_BATCH_REQUESTS} requests per batch")
        compiled = []
        for item in requests:
            try:
                compiled.append(compile_request(item))
            except RequestError as exc:
                compiled.append(exc)
        valid = [c for c in compiled if not isinstance(c, RequestError)]
        total = sum(map(len, valid))
        if total > MAX_BATCH_QUESTIONS:
            raise HTTPException(422, f"at most {MAX_BATCH_QUESTIONS} questions per batch")
        stats.compiled(1000 * (time.perf_counter() - started))
        rows = evaluate([question for c in valid for question in c])
        latency = 1000 * (time.perf_counter() - started)
        results, cursor, tokens = [], 0, 0
        for c in compiled:
            if isinstance(c, RequestError):
                results.append({"error": str(c)})
                continue
            response = respond(c, rows[cursor:cursor + len(c)], latency, bool(body.get("debug")))
            cursor += len(c)
            tokens += response["usage"]["input_tokens"]
            response.pop("latency_ms")
            results.append(response)
        stats.record(latency, total, tokens, requests=len(requests))
        return {"model": model_name, "results": results, "usage": {"input_tokens": tokens, "output_tokens": 0},
                "decisions": total, "latency_ms": latency}

    @app.post("/v1/tournament")
    def tournament(body: dict):
        """{state, question: choice, group_size=16, keep=2, seed=0}: groups of at most group_size candidates,
        the top `keep` of each advance until the rest fit one question. Answer probabilities cover finalists."""
        started = time.perf_counter()
        try:
            match = tournament_from(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        tokens = 0
        while not match.done:
            try:
                compiled = compile_questions(match.state, match.questions())
            except RequestError as exc:
                raise HTTPException(422, str(exc)) from exc
            tokens += sum(len(c[3]) for c in compiled)
            match.record([candidate_probabilities(logits, temperature) for logits in evaluate(compiled)])
        latency = 1000 * (time.perf_counter() - started)
        stats.record(latency, match.decisions, tokens)
        return {"model": model_name, **match.result(), "usage": {"input_tokens": tokens, "output_tokens": 0},
                "decisions": match.decisions, "latency_ms": latency}

    @app.post("/v1/compile")
    def compile_only(body: dict):
        """Compile prompts and count tokens without running the model."""
        try:
            compiled = compile_request(body)
        except RequestError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"prompt_version": PROMPT_VERSION, "questions": {
            name: {"prompt": prompt, "input_tokens": len(ids),
                   "options": [{"letter": LETTERS[i], "key": key, "description": description}
                               for i, (key, description) in enumerate(options)]}
            for name, _, options, ids, prompt in compiled}}

    @app.get("/v1/info")
    def info():
        return {"model": model_name, "version": app.version, "prompt_version": PROMPT_VERSION,
                "limits": {"max_length": max_length, "max_questions": MAX_QUESTIONS,
                           "max_candidates": len(LETTERS), "max_batch_requests": MAX_BATCH_REQUESTS,
                           "max_tournament_candidates": MAX_TOURNAMENT_CANDIDATES},
                "checkpoint": checkpoint, "validation": validation,
                # No dynamic batching: every forward pass is one prompt of at most max_length tokens.
                "engine": {"backend": engine.name, "device": engine.device, "threads": threads,
                           "token_budget": max_length, "max_batch": 1, "max_wait_ms": 0},
                "stats": stats.snapshot(), "gpu": None, "latest_best_step": None}

    @app.get("/v1/training")
    def training():
        """The finished training run recorded in the release metadata; this server does not train."""
        progress = summary.get("training_progress")
        if not progress:
            return {"available": False}
        seen, done = progress.get("examples_seen"), progress.get("status") == "completed"
        return {"available": True, "status": progress.get("status"), "step": progress.get("step"),
                "examples_seen": seen, "epoch_examples": seen if done else None, "progress": 1. if done else None,
                "updated_at": iso_mtime(root / "training_summary.json"), "best": selected or None,
                "trained_by_stratum": progress.get("stats", {})}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="models/chinese-decision-0.8b")
    parser.add_argument("--adapter")
    parser.add_argument("--backend", choices=["transformers", "gguf"], default="transformers")
    parser.add_argument("--gguf", help="GGUF weights; --base still supplies the tokenizer and decision_config.json")
    parser.add_argument("--device", default="cuda", help="Transformers device, e.g. cpu or cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=MAX_LENGTH, help="GGUF context capacity in tokens")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    uvicorn.run(create_app(args.base, args.adapter, backend=args.backend, gguf=args.gguf, device=args.device,
                           threads=args.threads, context_size=args.context_size), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
