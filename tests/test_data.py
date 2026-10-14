import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from decision.data import greedy_batches, prepare_domain, prepared_examples, shape_batches, token_batches


def padded_size(batch):
    return len(batch) * ((max(len(e["input_ids"]) for e in batch) + 63) // 64) * 64


class BatchingTests(unittest.TestCase):
    def test_shape_batches_respect_padded_budget_and_visit_all_rows(self):
        rows = [{"input_ids": [1] * n, "id": n} for n in (65, 129, 131, 135, 191, 200, 220)]
        output = shape_batches(rows, batch_size=8, token_budget=512)
        self.assertEqual([r["id"] for batch in output for r in batch], [r["id"] for r in rows])
        for batch in output:
            self.assertEqual(len(batch) & (len(batch) - 1), 0)
            self.assertLessEqual(padded_size(batch), 512)

    def test_greedy_batches_respect_row_and_token_limits(self):
        rows = [{"input_ids": [1] * n, "id": i} for i, n in enumerate((10, 10, 10, 40, 40, 90))]
        output = greedy_batches(rows, batch_size=2, token_budget=100)
        self.assertEqual([[r["id"] for r in batch] for batch in output], [[0, 1], [2, 3], [4], [5]])

    def test_token_batches_are_deterministic_and_lossless(self):
        class Item:
            kind, domain, source, hard = "noul", "general", "test", True
            def __init__(self, i):
                self.case_id, self.length = str(i), 65 + i * 17
            def render(self, tokenizer, rng):
                return str(self.length), [1., 0.]
        class Tokenizer:
            def encode(self, value, **kwargs):
                return list(range(int(value)))
        def batches():
            return list(token_batches(iter(Item(i) for i in range(60)), Tokenizer(), 2048, 32, 4096,
                                      42, Counter(), pool_size=64))
        a, b = batches(), batches()
        self.assertEqual({e["case_id"] for batch in a for e in batch}, {str(i) for i in range(60)})
        self.assertEqual(sum(len(batch) for batch in a), 60)
        self.assertEqual([[e["case_id"] for e in batch] for batch in a], [[e["case_id"] for e in batch] for batch in b])
        for batch in a:
            self.assertEqual(len(batch) & (len(batch) - 1), 0)
            self.assertLessEqual(padded_size(batch), 4096)


class CorpusMixTests(unittest.TestCase):
    def test_all_training_rows_preserved_once_and_seed_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "source", root / "prepared"
            (source / "general").mkdir(parents=True)
            rows = []
            for i in range(18):
                rows.append({"id": str(i), "state": "内容" + str(i),
                             "questions": {"decision": {"type": "noul", "instructions": "判断"}},
                             "gold": {"decision": {"probabilities": {"false": float(i % 2 == 0), "true": float(i % 2 == 1)}}},
                             "_meta": {"source_family": "source-" + str(i // 6)}})
            with gzip.open(source / "general/train.cases.jsonl.gz", "wt", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            result = prepare_domain((str(source), str(out), "general", len(rows)))
            (out / "manifest.json").write_text(json.dumps({"validation_included": False, "strata": result["strata"]}))
            def order():
                return list(prepared_examples(out, ["general"], 42, Counter(), buffer_size=2))
            a, b = order(), order()
            self.assertEqual(len(a), 18)
            self.assertEqual(len({e.case_id for e in a}), 18)
            self.assertEqual([e.case_id for e in a], [e.case_id for e in b])
            self.assertEqual(Counter(e.source for e in a), {"source-0": 6, "source-1": 6, "source-2": 6})
            self.assertEqual({e.case_id.split(":")[0] for e in a}, {str(i) for i in range(18)})


if __name__ == "__main__":
    unittest.main()
