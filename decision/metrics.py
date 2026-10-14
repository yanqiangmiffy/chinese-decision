"""Decision metrics: validation summaries for training, streaming metrics for CJ-Bench.

The two ECE definitions differ on purpose and are kept separate:
- `summarize` (validation during training) uses 10 equal-width bins on float32 probabilities;
  checkpoints are selected by distribution NLL, not ECE.
- `Accumulator` / `BenchmarkMetrics` (CJ-Bench) use the benchmark's official 15 bins.
ACC and ECE count hard targets only; NLL and Brier include soft targets.
"""
import math
from collections import defaultdict

import torch


def summarize(rows, temperature=1.0):
    groups = defaultdict(list)
    for row in rows:
        for group in ("all", "domain:" + row["domain"], "type:" + row["kind"], "source:" + row["source"]):
            groups[group].append(row)
    result = {}
    for key, items in groups.items():
        soft_nll, brier, soft_count, hard_hits, confidences, hard_nll = 0., 0., 0, [], [], 0.
        for item in items:
            logits, target = torch.tensor(item["logits"]) / temperature, torch.tensor(item["target"])
            logp = logits.log_softmax(-1)
            p = logp.exp()
            soft_nll += float(-(target * logp).sum())
            brier += float(((p - target) ** 2).sum())
            soft_count += int(not item["hard"])
            if item["hard"]:
                hard_hits.append(float(p.argmax() == target.argmax()))
                confidences.append(float(p.max()))
                hard_nll += float(-logp[target.argmax()])
        n, nh = len(items), len(hard_hits)
        ece = 0.
        for b in range(10):
            selected = [i for i, c in enumerate(confidences) if b / 10 <= c < (b + 1) / 10 or b == 9 and c == 1]
            if selected:
                ece += len(selected) / max(nh, 1) * abs(sum(hard_hits[i] for i in selected) / len(selected)
                                                      - sum(confidences[i] for i in selected) / len(selected))
        result[key] = {"n": n, "hard_n": nh, "soft_n": soft_count,
                       "accuracy": sum(hard_hits) / nh if nh else None,
                       "ece": ece if nh else None, "hard_nll": hard_nll / nh if nh else None,
                       "distribution_nll": soft_nll / n, "brier": brier / n}
    return result


class Accumulator:
    """Bounded-memory metrics for one CJ-Bench group."""

    def __init__(self, bins=15):
        self.bins = bins
        self.n = self.hard_n = self.hits = self.score_n = 0
        self.nll = self.hard_nll = self.brier = self.score_mae = 0.0
        self.bin_n = [0] * bins
        self.bin_hits = [0] * bins
        self.bin_confidence = [0.0] * bins

    def add(self, target, logp, kind, hard):
        if len(target) != len(logp) or not all(math.isfinite(x) for x in logp):
            raise ValueError("invalid benchmark prediction")
        probabilities = [math.exp(x) for x in logp]
        mass = sum(probabilities)
        if abs(mass - 1) > 1e-5:
            raise ValueError(f"prediction probability mass is {mass}")
        self.n += 1
        nll = -sum(t * lp for t, lp in zip(target, logp))
        self.nll += nll
        self.brier += sum((p - t) ** 2 for p, t in zip(probabilities, target))
        if hard:
            winner = max(range(len(logp)), key=logp.__getitem__)
            label = max(range(len(target)), key=target.__getitem__)
            hit = int(winner == label)
            confidence = min(1.0, probabilities[winner])
            index = min(self.bins - 1, int(confidence * self.bins))
            self.hard_n += 1
            self.hits += hit
            self.hard_nll += nll
            self.bin_n[index] += 1
            self.bin_hits[index] += hit
            self.bin_confidence[index] += confidence
        if kind == "score":
            expected = sum(i * p for i, p in enumerate(probabilities))
            target_expected = sum(i * t for i, t in enumerate(target))
            self.score_n += 1
            self.score_mae += abs(expected - target_expected)

    def result(self):
        ece = sum(abs(hits - confidence) for hits, confidence in
                  zip(self.bin_hits, self.bin_confidence)) / max(self.hard_n, 1)
        return {"n": self.n, "hard_n": self.hard_n, "soft_n": self.n - self.hard_n,
                "accuracy": self.hits / self.hard_n if self.hard_n else None,
                "ece": ece if self.hard_n else None, "ece_bins": self.bins,
                "distribution_nll": self.nll / self.n if self.n else None,
                "hard_nll": self.hard_nll / self.hard_n if self.hard_n else None,
                "brier": self.brier / self.n if self.n else None,
                "score_n": self.score_n,
                "expected_score_mae": self.score_mae / self.score_n if self.score_n else None}


class BenchmarkMetrics:
    """CJ-Bench metrics grouped by domain, type, source and task, plus domain macro averages."""

    def __init__(self, bins=15):
        self.groups = defaultdict(lambda: Accumulator(bins))

    def add(self, row, logp):
        domain, kind, source = row["domain"], row["kind"], row["source"]
        keys = ["all", f"domain:{domain}", f"type:{kind}", f"domain_type:{domain}:{kind}",
                f"source:{domain}:{source}"]
        if row.get("task") and row["task"] != "unknown":
            keys.append(f"task:{domain}:{row['task']}")
        for key in keys:
            self.groups[key].add(row["target"], logp, kind, row["hard"])

    def result(self):
        groups = {key: value.result() for key, value in sorted(self.groups.items())}
        domains = [v for k, v in groups.items() if k.startswith("domain:")]
        strata = [v for k, v in groups.items() if k.startswith("domain_type:")]
        def average(items, key):
            values = [v[key] for v in items if v[key] is not None]
            return sum(values) / len(values) if values else None
        return {"groups": groups,
                "macro_domain_accuracy": average(domains, "accuracy"),
                "macro_domain_distribution_nll": average(domains, "distribution_nll"),
                "macro_domain_type_accuracy": average(strata, "accuracy")}
