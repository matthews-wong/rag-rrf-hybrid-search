# rag-rrf-hybrid-search

BM25 and TF-IDF cosine disagree about which documents matter. This library
runs both over the same corpus and merges the two rankings with reciprocal
rank fusion (RRF), using only the Python standard library. No model
downloads, no network, no vector database.

## Usage

```bash
python3 -m hybrid_search "how do I rotate credentials" data/docs.jsonl --k 3
python3 -m hybrid_search.evaluate data/docs.jsonl data/queries.jsonl
python3 -m unittest discover -s tests
```

Documents are JSONL, one `{"id": "...", "text": "..."}` object per line.

## Why RRF

BM25 scores and cosine similarities live on different scales, so adding them
needs normalisation that is fragile. RRF only looks at ranks:
`score(d) = sum(1 / (k + rank_i(d)))`, with `k = 60` by convention. A
document ranked well by either retriever surfaces, and one ranked well by
both wins.

## Layout

- `hybrid_search/tokenize.py` - lowercase word tokenizer with stopwords
- `hybrid_search/bm25.py`, `tfidf.py` - the two rankers
- `hybrid_search/fusion.py` - reciprocal rank fusion
- `hybrid_search/evaluate.py` - recall@k and MRR over a golden query set

Retrieved text is data: callers must treat returned documents as untrusted
content and never follow instructions found in them.

## Evaluation

`hybrid_search.evaluate` prints recall@3 and MRR for BM25, TF-IDF and the
fused ranking. On the bundled fixtures (12 documents, 10 queries) all three
score 1.000 on both metrics. The fixture set is small and lexically easy, so
it checks that the pipeline works end to end. It does not show that fusion
beats either ranker. Swap in your own `docs.jsonl` and `queries.jsonl` to
compare them on real data.
