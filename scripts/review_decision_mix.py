"""Review a built decision mix: per-source sizes, losses, label balance, candidate counts, token lengths, samples.

    python scripts/review_decision_mix.py --raw datasets/decision-mix-v1/raw --out docs/DECISION_MIX_V1_REVIEW.md

Token lengths are measured with the model's tokenizer in the pointer layout on a sample of lines per source.
"""
import argparse
import gzip
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from decision.configuration import DecisionConfig, PointerConfig
from decision.processing import DecisionProcessor, examples_for, request_for


def lines(path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def gold_key(case, name):
    gold = case["gold"][name]
    if isinstance(gold, dict):
        probabilities = gold["probabilities"]
        return max(probabilities, key=probabilities.get), max(probabilities.values()) < 0.999
    return str(gold), False


def cut(value, n):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= n else text[:n] + "…"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="datasets/decision-mix-v1/raw")
    parser.add_argument("--tokenizer", default="models/chinese-decision-4b")
    parser.add_argument("--out", default="docs/DECISION_MIX_V1_REVIEW.md")
    parser.add_argument("--token-sample", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    raw = Path(args.raw)
    staging = raw / "_staging"
    report = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(staging.glob("*.json"))}
    processor = DecisionProcessor.from_pretrained(args.tokenizer)
    processor.config = DecisionConfig(pointer=PointerConfig(hidden_size=0), max_length=args.max_length)
    rows, samples, domain_totals = [], {}, defaultdict(Counter)
    for name, stats in report.items():
        if "error" in stats or "missing" in stats:
            rows.append({"source": name, "problem": stats.get("error") or stats.get("missing")})
            continue
        cases = list(lines(staging / f"{name}.train.jsonl.gz"))
        gold, soft, kinds, options, langs, questions_per_line = Counter(), 0, Counter(), [], Counter(), []
        for case in cases:
            questions_per_line.append(len(case["questions"]))
            langs[case["_meta"].get("lang")] += 1
            for q, question in case["questions"].items():
                key, is_soft = gold_key(case, q)
                soft += is_soft
                kinds[question["type"]] += 1
                gold[(question["type"], key if question["type"] != "choice" or len(question["criteria"]) <= 12 else "*")] += 1
                criteria = question.get("criteria")
                options.append(2 if question["type"] == "noul" else len(criteria))
        total_q = sum(kinds.values())
        # Majority share of the gold answer within each kind (choice with >12 options pooled as "*").
        balance = {}
        for kind in kinds:
            counts = {k: v for (t, k), v in gold.items() if t == kind}
            if kind == "choice" and set(counts) == {"*"}:
                continue
            balance[kind] = round(max(counts.values()) / sum(counts.values()), 3)
        rng = random.Random(name)
        sample = rng.sample(cases, min(args.token_sample, len(cases)))
        lengths, over = [], 0
        for case in sample:
            item = processor.encode_request(request_for(case, stats["domain"], processor.max_options))
            if item is None:
                over += 1
                continue
            lengths.append(len(item["input_ids"]))
        lengths.sort()
        p = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else 0
        row = {"source": name, "domain": stats["domain"], "lang": "/".join(f"{k}" for k, _ in langs.most_common(2)),
               "seen": stats.get("seen", 0), "overlap": stats.get("overlap_dropped", 0), "invalid": stats.get("invalid", 0),
               "train_q": stats.get("train", 0), "val_q": stats.get("validation", 0), "lines": stats.get("train_lines", 0),
               "q_per_line": round(statistics.mean(questions_per_line), 2) if questions_per_line else 0,
               "kinds": dict(kinds), "soft": round(soft / total_q, 3) if total_q else 0, "majority": balance,
               "options_mean": round(statistics.mean(options), 1) if options else 0, "options_max": max(options or [0]),
               "tok_p50": p(.5), "tok_p95": p(.95), "tok_mean": round(statistics.mean(lengths)) if lengths else 0,
               "overlength_in_sample": over}
        rows.append(row)
        domain_totals[stats["domain"]]["questions"] += row["train_q"]
        domain_totals[stats["domain"]]["lines"] += row["lines"]
        domain_totals[stats["domain"]]["tokens"] += row["tok_mean"] * row["lines"]
        samples[name] = rng.sample(cases, min(2, len(cases)))
    ok = [r for r in rows if "problem" not in r]
    out = [f"# {Path(args.raw).resolve().parent.name} 数据审阅", "",
           f"来源 {len(rows)} 个（成功 {len(ok)}）；训练题 {sum(r['train_q'] for r in ok):,}，训练行 {sum(r['lines'] for r in ok):,}，"
           f"验证题 {sum(r['val_q'] for r in ok):,}；估算训练 token {sum(r['tok_mean'] * r['lines'] for r in ok) / 1e6:,.0f}M（按抽样平均长度）。", ""]
    out += ["## 按领域", "", "| 领域 | 训练题 | 训练行 | 估算 token (M) |", "|---|---:|---:|---:|"]
    for domain, t in sorted(domain_totals.items(), key=lambda x: -x[1]["questions"]):
        out.append(f"| {domain} | {t['questions']:,} | {t['lines']:,} | {t['tokens'] / 1e6:,.1f} |")
    out += ["", "## 按来源", "",
            "| 来源 | 领域 | 语言 | 读取 | 去重删除 | 无效 | 训练题 | 验证题 | 每行题数 | 题型 | 软标签 | 多数类占比 | 候选均值/最大 | token p50/p95 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|"]
    for r in rows:
        if "problem" in r:
            out.append(f"| {r['source']} | **失败：{cut(r['problem'], 80)}** |||||||||||||")
            continue
        kinds = " ".join(f"{k}:{v}" for k, v in r["kinds"].items())
        majority = " ".join(f"{k}:{v:.0%}" for k, v in r["majority"].items())
        out.append(f"| {r['source']} | {r['domain']} | {r['lang']} | {r['seen']:,} | {r['overlap']:,} | {r['invalid']:,} | "
                   f"{r['train_q']:,} | {r['val_q']:,} | {r['q_per_line']} | {kinds} | {r['soft']:.0%} | {majority} | "
                   f"{r['options_mean']}/{r['options_max']} | {r['tok_p50']}/{r['tok_p95']} |")
    out += ["", "## 抽样（每个来源 2 条，截断显示）", ""]
    for name, cases in samples.items():
        out.append(f"### {name}")
        for case in cases:
            for q, question in list(case["questions"].items())[:2]:
                key, _ = gold_key(case, q)
                criteria = question.get("criteria")
                shown = (list(criteria.items())[:6] if isinstance(criteria, dict) else list(enumerate(criteria or []))[:6])
                out.append(f"- **状态**：{cut(case['state'], 220)}  ")
                out.append(f"  **问题**（{question['type']}）：{cut(question['instructions'], 160)}  ")
                out.append(f"  **候选**：{cut(json.dumps(shown, ensure_ascii=False), 260)}  ")
                out.append(f"  **答案**：`{cut(key, 80)}`")
        out.append("")
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    (raw / "review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"sources": len(rows), "ok": len(ok), "train_questions": sum(r["train_q"] for r in ok),
                      "report": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
