"""Metrics: decision validation summaries, streaming CJ-Bench metrics, and span extraction scores.

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


def entity_metrics(items, predictions, bins=10):
    """Micro precision/recall/F1 over exact (start, end, type) matches, per type as well, plus ECE of predictions.

    `items` need `gold_entities` (a set of (start, end, type)) and `unrepresented` (gold spans the
    encoder could not represent), as `DecisionProcessor.encode_extraction` gives them.
    """
    tp = fp = fn = unrepresented = 0
    confidences, hits, per_type = [], [], {}
    for item, predicted in zip(items, predictions):
        gold = item["gold_entities"]
        keys = {(s, e, t) for s, e, t, _ in predicted}
        tp += len(keys & gold)
        fp += len(keys - gold)
        fn += len(gold - keys)
        unrepresented += item["unrepresented"]
        for kind, found in (("tp", keys & gold), ("fp", keys - gold), ("fn", gold - keys)):
            for _, _, label in found:
                per_type.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})[kind] += 1
        for s, e, t, probability in predicted:
            confidences.append(probability)
            hits.append(float((s, e, t) in gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ece = 0.0
    for b in range(bins):
        chosen = [i for i, c in enumerate(confidences) if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1)]
        if chosen:
            ece += len(chosen) / len(confidences) * abs(sum(hits[i] for i in chosen) / len(chosen)
                                                        - sum(confidences[i] for i in chosen) / len(chosen))
    types = {label: {"gold": c["tp"] + c["fn"], "f1": 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"])}
             for label, c in sorted(per_type.items())}
    return {"rows": len(items), "gold": tp + fn, "predicted": tp + fp, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "entity_ece": ece if confidences else None, "unrepresented_gold": unrepresented, "per_type": types}
