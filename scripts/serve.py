"""hmm / System One-compatible HTTP API: one forward pass per question, no generated tokens."""
import argparse
import json
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException

from decision.api import build_answer
from decision.model import forward_batch, letter_tokens, load_model, load_tokenizer
from decision.schema import MAX_LENGTH, compile_prompt, options_for

MAX_QUESTIONS = 64


def create_app(base, adapter=None):
    tokenizer = load_tokenizer(base)
    model = load_model(base, adapter).eval()
    letters = letter_tokens(tokenizer, "cuda")
    config_path = Path(base) / "decision_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    temperature = config.get("temperature", 1.)
    max_length = config.get("max_length", MAX_LENGTH)
    lock = threading.Lock()
    app = FastAPI(title="Chinese Decision 0.8B")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "chinese-decision-0.8b", "temperature": temperature,
                "calibrated": bool(config_path.exists())}

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
                compiled.append({"input_ids": ids, "target": [0.] * len(options)})
                option_lists.append(options)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        answers = {}
        # Serial questions bound memory even when long states contain many questions.
        with lock, torch.inference_mode():
            for name, item, options in zip(names, compiled, option_lists):
                logits, _ = forward_batch(model, tokenizer, letters, [item])
                probabilities = (logits[0, :len(options)] / temperature).softmax(-1).cpu().tolist()
                answers[name] = build_answer(body["questions"][name]["type"], options, probabilities)
        return {"model": "chinese-decision-0.8b", "answers": answers,
                "usage": {"input_tokens": sum(len(e["input_ids"]) for e in compiled), "output_tokens": 0},
                "latency_ms": 1000 * (time.perf_counter() - started)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="models/chinese-decision-0.8b")
    parser.add_argument("--adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    torch.set_num_threads(8)
    uvicorn.run(create_app(args.base, args.adapter), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
