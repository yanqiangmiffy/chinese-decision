"""Measure independent CPU decisions; compare probabilities across isolated workers.

These are diagnostic prompts, not a labeled accuracy/calibration evaluation set.
Each backend runs in a separate process to avoid holding multiple models in RAM.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from decision.utils import model_name_for

ROOT = Path(__file__).resolve().parents[1]


def cases():
    example = json.loads((ROOT / "examples/request.json").read_text(encoding="utf-8"))
    rows = [{"name": f"ticket_{name}", "state": example["state"], "question": question}
            for name, question in example["questions"].items()]
    samples = [
        ("normal", "谢谢，发票已经收到，问题解决了。", "客户是否仍然需要人工处理？", "服务态度是否友好？"),
        ("medical", "患者描述昨晚开始咳嗽，今天有发热；未提供体温和既往病史。", "上下文是否提供了具体体温？", "上下文信息是否充分？"),
        ("legal", "双方约定每月十日前支付租金。承租人本月十五日仍未支付。", "是否已超过约定付款日期？", "合同履行风险有多高？"),
        ("finance", "本月收入两万元，支出一万八千元，没有其他收支。", "本月收入是否大于支出？", "结余占收入的比例属于哪个等级？"),
    ]
    for name, state, yes_no, score in samples:
        rows.extend([
            {"name": name + "_noul", "state": state, "question": {"type": "noul", "instructions": yes_no}},
            {"name": name + "_score", "state": state, "question": {"type": "score", "instructions": score,
                                                              "criteria": ["低", "中", "高"]}},
        ])
    rows.append({"name": "structured_choice", "state": {"message": "请帮我重置密码 🔑", "locale": "zh-CN"},
                 "question": {"type": "choice", "instructions": "应由哪个部门处理？", "criteria": {
                     "billing": "支付与发票", "technical": "账户和技术问题", "sales": "购买咨询"}}})
    rows.append({"name": "choice_26", "state": "用户希望了解第十三号产品。", "question": {
        "type": "choice", "instructions": "用户想了解哪个产品？",
        "criteria": {str(i): f"第{i}号产品" for i in range(1, 27)}}})
    for size in (20, 60):
        rows.append({"name": f"long_{size}", "state": "历史记录：客户曾咨询订单进度，客服已正常回复。" * size
                     + "最新留言：现在已收到商品，请关闭工单。", "question": {
                         "type": "noul", "instructions": "根据最新留言，客户是否希望关闭工单？"}})
    return rows


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def backend_summary(report):
    short = [sample for case in report["cases"] if case["input_tokens"] < 100
             for sample in case["samples_ms"]]
    return {"short_input_token_limit_exclusive": 100, "short_samples": len(short),
            "short_p50_ms": statistics.median(short) if short else None,
            "short_p95_ms": percentile(short, .95) if short else None,
            "peak_rss_gib": report["peak_rss_bytes"] / 1024 ** 3,
            "repeat_max_probability_delta": max(c["repeat_max_probability_delta"] for c in report["cases"])}


def worker(args):
    import importlib.metadata
    import psutil
    from decision.serving.backends import GGUFBackend, TransformersBackend, candidate_probabilities
    from decision.processing import compile_prompt, options_for

    started = time.perf_counter()
    engine = (TransformersBackend(args.base, device="cpu", threads=args.threads) if args.worker == "transformers"
              else GGUFBackend(args.base, args.gguf, threads=args.threads, context_size=args.context_size))
    load_seconds = time.perf_counter() - started
    temperature = json.loads((Path(args.base) / "decision_config.json").read_text(encoding="utf-8"))["temperature"]
    prepared = []
    for case in cases():
        options = options_for(case["question"])
        prompt = compile_prompt(engine.tokenizer, case["state"], case["question"]["instructions"], options)
        ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(engine, "validate_tokens"):
            engine.validate_tokens(prompt, ids)
        prepared.append((case, options, ids))
    cold_start = time.perf_counter()
    engine.logits(prepared[0][2], len(prepared[0][1]))
    cold_ms = 1000 * (time.perf_counter() - cold_start)
    engine.logits(prepared[0][2], len(prepared[0][1]))
    results = []
    for case, options, ids in prepared:
        results.append({"name": case["name"], "type": case["question"]["type"], "input_tokens": len(ids),
                        "token_ids": ids, "candidate_keys": [k for k, _ in options], "samples_ms": [],
                        "repeat_max_probability_delta": 0.})
    for repeat in range(args.repeats):
        for result, (case, options, ids) in zip(results, prepared):
            started = time.perf_counter()
            values = engine.logits(ids, len(options))
            probabilities = candidate_probabilities(values, temperature)
            result["samples_ms"].append(1000 * (time.perf_counter() - started))
            if repeat == 0:
                result.update(logits=values, probabilities=probabilities,
                              top_candidate=options[max(range(len(options)), key=probabilities.__getitem__)][0])
            else:
                result["repeat_max_probability_delta"] = max(result["repeat_max_probability_delta"],
                    max(abs(p - q) for p, q in zip(probabilities, result["probabilities"])))
            print(json.dumps({"backend": args.worker, "repeat": repeat + 1, "case": case["name"],
                              "tokens": len(ids), "ms": round(result["samples_ms"][-1], 2)}), flush=True)
    for result in results:
        result["median_ms"] = statistics.median(result["samples_ms"])
        result["p95_ms"] = percentile(result["samples_ms"], .95)
    memory = psutil.Process().memory_info()
    versions = {}
    for package in ("torch", "transformers", "llama-cpp-python", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {"backend": args.worker, "gguf": args.gguf, "device": "cpu", "threads": args.threads,
              "context_size": args.context_size, "load_seconds": load_seconds, "cold_forward_ms": cold_ms,
              "rss_bytes": memory.rss, "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "temperature": temperature, "versions": versions, "cases": results}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(ROOT / "models/chinese-decision-0.8b"))
    parser.add_argument("--gguf-dir", help="defaults to the matching <model-name>-gguf sibling directory")
    parser.add_argument("--report", default=str(ROOT / "reports/cpu-benchmark.json"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=["transformers", "gguf"])
    parser.add_argument("--gguf")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    config_path = Path(args.base) / "decision_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    model_name = model_name_for(args.base, config)
    gguf_dir = Path(args.gguf_dir) if args.gguf_dir else Path(args.base).parent / (model_name + "-gguf")
    reports = {}
    for label, backend, filename in (("transformers_fp32", "transformers", None),
                                     ("gguf_bf16", "gguf", model_name + "-BF16.gguf"),
                                     ("gguf_q8_0", "gguf", model_name + "-Q8_0.gguf")):
        child_report = report_path.with_name(report_path.stem + "-" + label + ".json")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", backend, "--base", args.base,
                   "--report", str(child_report), "--threads", str(args.threads), "--repeats", str(args.repeats),
                   "--context-size", str(args.context_size)]
        if filename:
            command += ["--gguf", str(gguf_dir / filename)]
        env = os.environ.copy()
        env.update(TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        subprocess.run(command, check=True, cwd=ROOT, env=env)
        reports[label] = json.loads(child_report.read_text(encoding="utf-8"))
    comparisons = {}
    for left, right in (("transformers_fp32", "gguf_bf16"), ("gguf_bf16", "gguf_q8_0"),
                        ("transformers_fp32", "gguf_q8_0")):
        deltas, agreements, rows = [], 0, []
        for a, b in zip(reports[left]["cases"], reports[right]["cases"]):
            if a["name"] != b["name"] or a["token_ids"] != b["token_ids"]:
                raise ValueError("benchmark inputs or tokenizer IDs differ")
            values = [abs(p - q) for p, q in zip(a["probabilities"], b["probabilities"])]
            deltas.extend(values)
            agreements += a["top_candidate"] == b["top_candidate"]
            rows.append({"name": a["name"], "max_probability_delta": max(values),
                         "left_median_ms": a["median_ms"], "right_median_ms": b["median_ms"]})
        comparisons[left + "_vs_" + right] = {"max_probability_delta": max(deltas),
             "mean_probability_delta": statistics.mean(deltas), "top_candidate_agreement": agreements,
             "questions": len(rows), "cases": rows}
    report = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
              "scope": "warm single-question input-ID evaluation + candidate softmax; no HTTP/tokenization; no prefix reuse",
              "limitation": "15 diagnostic prompts; no accuracy or calibration claims; sequential single-client CPU measurements",
              "backends": reports, "summary": {key: backend_summary(value) for key, value in reports.items()},
              "comparisons": comparisons}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
