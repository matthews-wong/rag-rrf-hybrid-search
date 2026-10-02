import unittest
from pathlib import Path

from hybrid_search.evaluate import evaluate, recall_at_k, reciprocal_rank
from hybrid_search.search import HybridSearcher, load_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"


class MetricTest(unittest.TestCase):
    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"b", "z"}, 2), 0.5)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["a", "b", "c"], {"c"}), 1 / 3)
        self.assertEqual(reciprocal_rank(["a"], {"z"}), 0.0)


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.docs = {r["id"]: r["text"] for r in load_jsonl(DATA / "docs.jsonl")}
        self.queries = load_jsonl(DATA / "queries.jsonl")

    def test_hybrid_finds_every_golden_answer_in_top_3(self):
        table = evaluate(self.docs, self.queries)
        self.assertEqual(table["hybrid"]["recall@3"], 1.0)

    def test_poisoned_chunk_is_returned_as_plain_data(self):
        # Retrieval must not interpret or filter content; callers delimit it as untrusted.
        hits = HybridSearcher(self.docs).search("credentials note", k=3)
        self.assertIn("poisoned-note", [d for d, _ in hits])


if __name__ == "__main__":
    unittest.main()
