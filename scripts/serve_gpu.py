"""Chinese Decision GPU 推理服务：hmm / System One 兼容接口 + Decision Studio 需要的扩展接口。

GPU 生产部署用这个（部署脚本见 deploy/gpu/）：动态批处理、训练进度、热更新。CPU / GGUF 或单机调试用
scripts/serve.py，两者的接口和返回格式一致。

兼容接口：
  POST /v1/systemone            一个 state + 多个问题
  POST /api/alpha/decisions     同上，兼容 hmm 路由
  GET  /health

扩展接口：
  POST /v1/batch                一次提交多个决策请求，跨请求合批（评测、解释、鲁棒性测试用）
  POST /v1/compile              只编译提示词并计数 token，不跑模型
  GET  /v1/info                 模型、检查点、GPU、运行统计、验证集指标
  GET  /v1/training             训练目录的实时进度（只读）
  POST /v1/admin/reload         把训练目录里最新的 best 快照下来并热加载（需要 X-Admin-Token）

所有前向都走同一个动态批处理队列：并发请求在几毫秒的窗口内自动合批，按 token 预算切成
micro-batch，GPU 上同一时刻只有一个前向。提示词编译器直接复用本仓库的 decision 包，
训练、评测、服务三者输入完全一致。
"""

import argparse
import asyncio
import json
import math
import os
import queue
import shutil
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import anyio
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from decision.api import build_answer
from decision.model import forward_batch, letter_tokens, load_model, load_tokenizer
from decision.schema import LETTERS, PROMPT_VERSION, compile_prompt, options_for

MODEL_NAME = "chinese-decision-0.8b"
MAX_LENGTH = 16384
MAX_QUESTIONS = 64
MAX_BATCH_REQUESTS = 512
MAX_BATCH_QUESTIONS = 4096
PAD_MULTIPLE = 64  # 与 decision.model.forward_batch 的 pad_to_multiple_of 一致
IDLE_RELEASE_SECONDS = 20  # 空闲多久后释放缓存的显存


def now_iso():
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def padded(length):
    return math.ceil(length / PAD_MULTIPLE) * PAD_MULTIPLE


class Checkpoint:
    """一份已加载的模型：基础模型 + LoRA adapter（或合并后的模型）。"""

    def __init__(self, base, adapter, device):
        started = time.perf_counter()
        self.base, self.adapter = str(base), str(adapter) if adapter else None
        self.tokenizer = load_tokenizer(base)
        self.model = load_model(base, adapter, device=device).eval()
        self.letters = letter_tokens(self.tokenizer, device)
        config = read_json(Path(base) / "decision_config.json", {}) or {}
        self.temperature = float(config.get("temperature", 1.0))
        self.calibrated = bool(config)
        self.max_length = int(config.get("max_length", MAX_LENGTH))
        self.selection = read_json(Path(adapter) / "selection.json", {}) if adapter else {}
        self.validation = read_json(Path(adapter) / "metrics.json", {}) if adapter else {}
        self.loaded_at = now_iso()
        self.load_seconds = time.perf_counter() - started

    def describe(self):
        return {"base": self.base, "adapter": self.adapter, "step": self.selection.get("step"),
                "selection_metric": self.selection.get("metric"), "selection_value": self.selection.get("value"),
                "temperature": self.temperature, "calibrated": self.calibrated, "max_length": self.max_length,
                "loaded_at": self.loaded_at, "load_seconds": round(self.load_seconds, 2)}


class Engine:
    """动态批处理：后台线程从队列里取任务，在 max_wait 窗口内合批，按 token 预算切 micro-batch。"""

    def __init__(self, checkpoint, token_budget, max_batch, max_wait_ms):
        self.checkpoint = checkpoint
        self.token_budget = token_budget
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue = queue.Queue()
        self.gpu_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.started_at = time.time()
        self.counters = {"requests": 0, "decisions": 0, "input_tokens": 0, "batches": 0, "forward_passes": 0,
                         "errors": 0, "oom_retries": 0}
        self.latencies = deque(maxlen=2000)  # 每个 HTTP 请求的端到端毫秒数
        self.batch_sizes = deque(maxlen=2000)  # 每个前向的样本数
        self.forwards = deque(maxlen=500)  # 最近的前向：样本数、padding 后长度、耗时
        self.compile_ms = deque(maxlen=2000)  # 提示词编译 + 分词（含线程排队）
        threading.Thread(target=self._loop, name="decision-engine", daemon=True).start()

    # ---------------------------------------------------------------- 提交与统计
    def submit(self, items):
        """items: [{"input_ids": [...], "k": 候选数}]，返回 Future → 每个 item 的候选 logits 列表。"""
        future = Future()
        self.queue.put((items, future))
        return future

    def record(self, latency_ms, decisions, tokens, requests=1):
        with self.stats_lock:
            self.counters["requests"] += requests
            self.counters["decisions"] += decisions
            self.counters["input_tokens"] += tokens
            self.latencies.append(latency_ms)

    def stats(self):
        with self.stats_lock:
            latencies = sorted(self.latencies)
            sizes = list(self.batch_sizes)
            counters = dict(self.counters)
            forwards = list(self.forwards)
            compile_ms = sorted(self.compile_ms)

        def pct(q):
            return round(latencies[min(len(latencies) - 1, int(q * len(latencies)))], 2) if latencies else None

        forward_ms = sorted(f["ms"] for f in forwards)
        return {**counters, "uptime_s": round(time.time() - self.started_at, 1), "queue_depth": self.queue.qsize(),
                "latency_p50_ms": pct(0.5), "latency_p95_ms": pct(0.95), "latency_p99_ms": pct(0.99),
                "avg_forward_batch": round(sum(sizes) / len(sizes), 2) if sizes else None,
                "forward_p50_ms": forward_ms[len(forward_ms) // 2] if forward_ms else None,
                "compile_p50_ms": round(compile_ms[len(compile_ms) // 2], 2) if compile_ms else None,
                "recent_forwards": forwards[-40:]}

    def swap(self, checkpoint):
        with self.gpu_lock:
            previous, self.checkpoint = self.checkpoint, checkpoint
        del previous
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 后台线程
    def _loop(self):
        while True:
            try:
                jobs = [self.queue.get(timeout=IDLE_RELEASE_SECONDS)]
            except queue.Empty:
                # GPU 是和别人共用的：空闲一段时间后把缓存的显存还给驱动，只保留权重本身
                if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 256 * 2**20:
                    with self.gpu_lock:
                        torch.cuda.empty_cache()
                continue
            count = len(jobs[0][0])
            deadline = time.perf_counter() + self.max_wait
            while count < self.max_batch * 4:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    job = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                jobs.append(job)
                count += len(job[0])
            self._run(jobs)

    def _chunks(self, order, flat):
        chunk, longest = [], 0
        for index in order:
            length = padded(len(flat[index][2]["input_ids"]))
            if chunk and (len(chunk) >= self.max_batch or (len(chunk) + 1) * max(longest, length) > self.token_budget):
                yield chunk
                chunk, longest = [], 0
            chunk.append(index)
            longest = max(longest, length)
        if chunk:
            yield chunk

    def _forward(self, checkpoint, flat, chunk):
        started = time.perf_counter()
        batch = [{"input_ids": flat[i][2]["input_ids"], "target": [0.0] * flat[i][2]["k"]} for i in chunk]
        logits, _ = forward_batch(checkpoint.model, checkpoint.tokenizer, checkpoint.letters, batch)
        logits = logits.cpu()
        elapsed = 1000 * (time.perf_counter() - started)
        length = padded(max(len(b["input_ids"]) for b in batch))
        with self.stats_lock:
            self.counters["forward_passes"] += 1
            self.batch_sizes.append(len(chunk))
            self.forwards.append({"items": len(chunk), "padded_len": length, "ms": round(elapsed, 2)})
        return logits

    def _run(self, jobs):
        flat = [(j, i, item) for j, (items, _) in enumerate(jobs) for i, item in enumerate(items)]
        results = [[None] * len(items) for items, _ in jobs]
        order = sorted(range(len(flat)), key=lambda x: len(flat[x][2]["input_ids"]))
        try:
            with self.gpu_lock, torch.inference_mode():
                checkpoint = self.checkpoint
                for chunk in self._chunks(order, flat):
                    try:
                        rows = [(chunk, self._forward(checkpoint, flat, chunk))]
                    except torch.cuda.OutOfMemoryError:
                        # 显存不够就退化成逐条前向，慢一点但不失败
                        torch.cuda.empty_cache()
                        with self.stats_lock:
                            self.counters["oom_retries"] += 1
                        rows = [([i], self._forward(checkpoint, flat, [i])) for i in chunk]
                    for indices, logits in rows:
                        for row, index in zip(logits, indices, strict=True):
                            j, i, item = flat[index]
                            results[j][i] = row[: item["k"]].tolist()
            with self.stats_lock:
                self.counters["batches"] += 1
        except Exception as exc:  # noqa: BLE001
            with self.stats_lock:
                self.counters["errors"] += 1
            for _, future in jobs:
                future.set_exception(exc)
            return
        for (_, future), result in zip(jobs, results, strict=True):
            future.set_result(result)


# -------------------------------------------------------------------- 请求编译
class RequestError(ValueError):
    pass


def compile_request(checkpoint, body):
    """校验并编译一个决策请求，返回 [(name, question, options, input_ids)]。"""
    if not isinstance(body, dict) or body.get("state") is None:
        raise RequestError("state is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a nonempty object")
    if len(questions) > MAX_QUESTIONS:
        raise RequestError(f"at most {MAX_QUESTIONS} questions per request")
    compiled = []
    for name, question in questions.items():
        try:
            options = options_for(question)
            prompt = compile_prompt(checkpoint.tokenizer, body["state"], question["instructions"], options)
        except (KeyError, TypeError, ValueError) as exc:
            raise RequestError(f"question {name}: {exc}") from exc
        ids = checkpoint.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > checkpoint.max_length:
            raise RequestError(f"question {name}: input has {len(ids)} tokens; limit {checkpoint.max_length}; "
                               "input was not truncated")
        compiled.append((name, question, options, ids, prompt))
    return compiled


def make_response(checkpoint, compiled, logits_rows, latency_ms, debug=False):
    answers = {}
    for (name, question, options, ids, _), logits in zip(compiled, logits_rows, strict=True):
        scaled = torch.tensor(logits) / checkpoint.temperature
        probabilities = scaled.softmax(-1).tolist()
        answer = build_answer(question["type"], options, probabilities)
        if debug:
            answer["logits"] = {key: round(value, 4) for (key, _), value in zip(options, logits, strict=True)}
            answer["input_tokens"] = len(ids)
        answers[name] = answer
    return {"model": MODEL_NAME, "answers": answers,
            "usage": {"input_tokens": sum(len(c[3]) for c in compiled), "output_tokens": 0},
            "latency_ms": latency_ms}


# -------------------------------------------------------------------- 训练目录（只读）
def training_status(run_dir, epoch_examples):
    run_dir = Path(run_dir)
    progress = read_json(run_dir / "progress.json")
    if progress is None:
        return {"available": False}
    latest = read_json(run_dir / "latest_metrics.json", {}) or {}
    best = read_json(run_dir / "best" / "selection.json", {}) or {}
    config = read_json(run_dir / "config.json", {}) or {}
    seen = progress.get("examples_seen") or 0
    try:
        updated = datetime.fromtimestamp((run_dir / "progress.json").stat().st_mtime, UTC).isoformat(timespec="seconds")
    except OSError:
        updated = None
    keys = ("all",) + tuple(k for k in latest if k.startswith(("domain:", "type:")))
    return {"available": True, "status": progress.get("status"), "step": progress.get("step"),
            "examples_seen": seen, "epoch_examples": epoch_examples,
            "progress": min(1.0, seen / epoch_examples) if epoch_examples else None,
            "world_size": progress.get("world_size"), "updated_at": updated, "best": best,
            "latest_eval": {k: latest[k] for k in keys if k in latest},
            "latest_eval_step": latest.get("step"),
            "trained_by_stratum": progress.get("stats", {}),
            "config": {k: config.get(k) for k in ("base", "lr", "lora_r", "global_batch_size", "max_length", "epochs",
                                                   "prompt_version", "world_size")}}


def snapshot_best(run_dir, snapshots):
    """把训练目录的 best 复制成不可变快照；复制前后 selection.json 不一致说明正在被替换，重试。"""
    source = Path(run_dir) / "best"
    for _ in range(5):
        before = read_json(source / "selection.json")
        if not before:
            raise RuntimeError(f"{source} has no selection.json")
        target = Path(snapshots) / f"step-{before['step']}"
        if target.exists() and read_json(target / "selection.json") == before:
            return target
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(source, partial)
        if read_json(source / "selection.json") == before == read_json(partial / "selection.json"):
            shutil.rmtree(target, ignore_errors=True)
            partial.rename(target)
            return target
        time.sleep(2)
    raise RuntimeError("best checkpoint kept changing while copying; try again later")


# -------------------------------------------------------------------- 应用
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


def create_app(args):
    checkpoint = Checkpoint(args.base, args.adapter, "cuda")
    engine = Engine(checkpoint, args.token_budget, args.max_batch, args.max_wait_ms)
    reload_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_):
        # 每个请求都要在线程里编译提示词；默认 40 个线程会把压测并发卡在 40
        anyio.to_thread.current_default_thread_limiter().total_tokens = 256
        # 预热一次，避免第一个真实请求吃到 CUDA kernel 编译的延迟；预热不计入统计
        await decide_one({"state": "预热", "questions": {"q": {"type": "noul", "instructions": "这是预热吗？"}}}, False)
        with engine.stats_lock:
            for key in ("requests", "decisions", "input_tokens", "batches", "forward_passes"):
                engine.counters[key] = 0
            engine.latencies.clear()
            engine.batch_sizes.clear()
            engine.forwards.clear()
            engine.compile_ms.clear()
        yield

    app = FastAPI(title="Chinese Decision 0.8B", version="0.2.0", lifespan=lifespan,
                  description="一次前向，直接输出候选项概率。兼容 hmm / System One 接口。")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    async def decide_one(body, debug):
        started = time.perf_counter()
        current = engine.checkpoint
        try:
            compiled = await asyncio.to_thread(compile_request, current, body)
        except RequestError as exc:
            raise HTTPException(422, str(exc)) from exc
        engine.compile_ms.append(1000 * (time.perf_counter() - started))
        logits = await asyncio.wrap_future(engine.submit([{"input_ids": c[3], "k": len(c[2])} for c in compiled]))
        latency = 1000 * (time.perf_counter() - started)
        response = make_response(current, compiled, logits, latency, debug)
        engine.record(latency, len(compiled), response["usage"]["input_tokens"])
        return response

    @app.get("/health")
    def health():
        current = engine.checkpoint
        return {"status": "ok", "model": MODEL_NAME, "temperature": current.temperature,
                "calibrated": current.calibrated, "step": current.selection.get("step")}

    @app.post("/api/alpha/decisions")
    @app.post("/v1/systemone")
    async def decide(body: dict, request: Request):
        debug = request.query_params.get("debug", "").lower() in ("1", "true", "yes")
        return await decide_one(body, debug)

    @app.post("/v1/batch")
    async def batch(body: dict):
        """{"requests": [{state, questions}, ...], "debug": bool} → 每个请求一个结果，单个请求出错不影响其他请求。"""
        started = time.perf_counter()
        requests = body.get("requests")
        if not isinstance(requests, list) or not requests:
            raise HTTPException(422, "requests must be a nonempty list")
        if len(requests) > MAX_BATCH_REQUESTS:
            raise HTTPException(422, f"at most {MAX_BATCH_REQUESTS} requests per batch")
        debug = bool(body.get("debug"))
        current = engine.checkpoint

        def compile_all():
            out = []
            for item in requests:
                try:
                    out.append(compile_request(current, item))
                except RequestError as exc:
                    out.append(exc)
            return out

        compiled = await asyncio.to_thread(compile_all)
        total = sum(len(c) for c in compiled if not isinstance(c, Exception))
        if total > MAX_BATCH_QUESTIONS:
            raise HTTPException(422, f"at most {MAX_BATCH_QUESTIONS} questions per batch")
        items = [{"input_ids": q[3], "k": len(q[2])} for c in compiled if not isinstance(c, Exception) for q in c]
        logits = await asyncio.wrap_future(engine.submit(items)) if items else []
        latency = 1000 * (time.perf_counter() - started)
        results, cursor, tokens = [], 0, 0
        for c in compiled:
            if isinstance(c, Exception):
                results.append({"error": str(c)})
                continue
            response = make_response(current, c, logits[cursor: cursor + len(c)], latency, debug)
            cursor += len(c)
            tokens += response["usage"]["input_tokens"]
            response.pop("latency_ms")
            results.append(response)
        engine.record(latency, total, tokens, requests=len(requests))
        return {"model": MODEL_NAME, "results": results, "usage": {"input_tokens": tokens, "output_tokens": 0},
                "decisions": total, "latency_ms": latency}

    @app.post("/v1/compile")
    async def compile_only(body: dict):
        current = engine.checkpoint
        try:
            compiled = await asyncio.to_thread(compile_request, current, body)
        except RequestError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"prompt_version": PROMPT_VERSION, "questions": {
            name: {"prompt": prompt, "input_tokens": len(ids),
                   "options": [{"letter": LETTERS[i], "key": k, "description": d} for i, (k, d) in enumerate(options)]}
            for name, _, options, ids, prompt in compiled}}

    @app.get("/v1/info")
    def info():
        current = engine.checkpoint
        return {"model": MODEL_NAME, "version": app.version, "prompt_version": PROMPT_VERSION,
                "limits": {"max_length": current.max_length, "max_questions": MAX_QUESTIONS,
                           "max_candidates": len(LETTERS), "max_batch_requests": MAX_BATCH_REQUESTS},
                "checkpoint": current.describe(), "validation": current.validation,
                "engine": {"token_budget": engine.token_budget, "max_batch": engine.max_batch,
                           "max_wait_ms": engine.max_wait * 1000},
                "stats": engine.stats(), "gpu": gpu_info(),
                "latest_best_step": (read_json(Path(args.run_dir) / "best" / "selection.json", {}) or {}).get("step")
                if args.run_dir else None}

    @app.get("/v1/training")
    def training():
        if not args.run_dir:
            return {"available": False}
        return training_status(args.run_dir, args.epoch_examples)

    @app.post("/v1/admin/reload")
    async def reload(x_admin_token: str = Header(default="")):
        if not args.admin_token or x_admin_token != args.admin_token:
            raise HTTPException(403, "reload requires a valid X-Admin-Token")
        if not args.run_dir:
            raise HTTPException(400, "server was started without --run-dir")
        if not reload_lock.acquire(blocking=False):
            raise HTTPException(409, "a reload is already in progress")
        try:
            target = await asyncio.to_thread(snapshot_best, args.run_dir, args.snapshots)
            if engine.checkpoint.adapter == str(target):
                return {"reloaded": False, "checkpoint": engine.checkpoint.describe()}
            fresh = await asyncio.to_thread(Checkpoint, args.base, target, "cuda")
            await asyncio.to_thread(engine.swap, fresh)
            return {"reloaded": True, "checkpoint": fresh.describe()}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            reload_lock.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="基础模型或合并后的模型目录")
    parser.add_argument("--adapter", help="LoRA adapter 目录（合并模型不需要）")
    parser.add_argument("--run-dir", help="训练目录（只读）：提供训练进度和热更新")
    parser.add_argument("--snapshots", default="snapshots", help="热更新时 best 快照的存放目录")
    parser.add_argument("--epoch-examples", type=int, default=13_048_000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--token-budget", type=int, default=65536, help="单个 micro-batch padding 后的 token 上限")
    parser.add_argument("--max-batch", type=int, default=64, help="单个 micro-batch 的最大样本数")
    parser.add_argument("--max-wait-ms", type=float, default=4.0, help="合批等待窗口")
    parser.add_argument("--admin-token", default=os.environ.get("DECISION_ADMIN_TOKEN", ""))
    args = parser.parse_args()
    torch.set_num_threads(8)
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
