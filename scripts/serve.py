"""hmm / System One-compatible HTTP API: one forward pass per question, no generated tokens."""
import argparse
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from decision.api import build_answer
from decision.inference import GGUFBackend, TransformersBackend, candidate_probabilities
from decision.schema import MAX_LENGTH, PROMPT_VERSION, compile_prompt, options_for

MAX_QUESTIONS = 64


def create_app(base, adapter=None, *, backend="transformers", gguf=None, device="cuda", threads=8,
               context_size=MAX_LENGTH):
    config_path = Path(base) / "decision_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
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

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title="Chinese Decision 0.8B", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "chinese-decision-0.8b", "temperature": temperature,
                "calibrated": bool(config_path.exists()), "backend": engine.name,
                "device": engine.device, "max_length": max_length}

    @app.post("/api/alpha/decisions")
    @app.post("/v1/systemone")
    def decide(body: dict):
        started = time.perf_counter()
        if body.get("state") is None or not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise HTTPException(422, "state and nonempty questions are required")
        if len(body["questions"]) > MAX_QUESTIONS:
            raise HTTPException(422, f"at most {MAX_QUESTIONS} questions per request")
        compiled, option_lists, names = [], [], list(body["questions"])
        try:
            for name, question in body["questions"].items():
                options = options_for(question)
                prompt = compile_prompt(tokenizer, body["state"], question["instructions"], options)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) > max_length:
                    raise ValueError(f"question {name}: input has {len(ids)} tokens; limit {max_length}; input was not truncated")
                if hasattr(engine, "validate_tokens"):
                    engine.validate_tokens(prompt, ids)
                compiled.append({"input_ids": ids, "target": [0.] * len(options)})
                option_lists.append(options)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        answers = {}
        # Serial questions bound memory even when long states contain many questions.
        with lock:
            for name, item, options in zip(names, compiled, option_lists):
                logits = engine.logits(item["input_ids"], len(options))
                probabilities = candidate_probabilities(logits, temperature)
                answers[name] = build_answer(body["questions"][name]["type"], options, probabilities)
        return {"model": "chinese-decision-0.8b", "answers": answers,
                "usage": {"input_tokens": sum(len(e["input_ids"]) for e in compiled), "output_tokens": 0},
                "latency_ms": 1000 * (time.perf_counter() - started)}

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
