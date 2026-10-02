import unittest

from hybrid_search.bm25 import BM25Index
from hybrid_search.fusion import reciprocal_rank_fusion
from hybrid_search.tfidf import TfidfIndex
from hybrid_search.tokenize import tokenize

DOCS = {
    "a": "rotate the api key every ninety days",
    "b": "restore the database from a nightly backup",
    "c": "api rate limits return http 429",
}


class TokenizeTest(unittest.TestCase):
    def test_lowercases_and_drops_stopwords(self):
        self.assertEqual(tokenize("How do I Rotate the API-Key?"), ["rotate", "api", "key"])


class RankerTest(unittest.TestCase):
    def test_bm25_ranks_matching_doc_first(self):
        self.assertEqual(BM25Index(DOCS).search("restore backup")[0][0], "b")

    def test_tfidf_ranks_matching_doc_first(self):
        self.assertEqual(TfidfIndex(DOCS).search("rotate key")[0][0], "a")

    def test_no_overlap_returns_empty(self):
        self.assertEqual(BM25Index(DOCS).search("kubernetes"), [])
        self.assertEqual(TfidfIndex(DOCS).search("kubernetes"), [])

    def test_empty_corpus_is_safe(self):
        self.assertEqual(BM25Index({}).search("anything"), [])


class FusionTest(unittest.TestCase):
    def test_agreement_beats_single_ranker(self):
        first = [("x", 9.0), ("y", 5.0)]
        second = [("y", 0.9), ("z", 0.1)]
        self.assertEqual(reciprocal_rank_fusion([first, second])[0][0], "y")

    def test_ignores_raw_score_scale(self):
        small = [("x", 0.001), ("y", 0.0005)]
        large = [("x", 1000.0), ("y", 500.0)]
        self.assertEqual(reciprocal_rank_fusion([small]), reciprocal_rank_fusion([large]))

    def test_rejects_non_positive_k(self):
        with self.assertRaisesRegex(ValueError, "got 0"):
            reciprocal_rank_fusion([], k=0)


if __name__ == "__main__":
    unittest.main()
