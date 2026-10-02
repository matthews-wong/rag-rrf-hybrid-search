import sys

from .bm25 import BM25Index
from .search import HybridSearcher, load_jsonl
from .tfidf import TfidfIndex

DEFAULT_CUTOFF = 3


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def evaluate(docs: dict[str, str], queries: list[dict], k: int = DEFAULT_CUTOFF) -> dict[str, dict[str, float]]:
    bm25, tfidf, hybrid = BM25Index(docs), TfidfIndex(docs), HybridSearcher(docs)
    systems = {
        "bm25": lambda q: [d for d, _ in bm25.search(q)],
        "tfidf": lambda q: [d for d, _ in tfidf.search(q)],
        "hybrid": lambda q: [d for d, _ in hybrid.search(q, k=len(docs))],
    }
    table = {}
    for name, rank in systems.items():
        recalls, rrs = [], []
        for row in queries:
            ranked, relevant = rank(row["query"]), set(row["relevant"])
            recalls.append(recall_at_k(ranked, relevant, k))
            rrs.append(reciprocal_rank(ranked, relevant))
        table[name] = {f"recall@{k}": sum(recalls) / len(recalls), "mrr": sum(rrs) / len(rrs)}
    return table


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 -m hybrid_search.evaluate DOCS.jsonl QUERIES.jsonl", file=sys.stderr)
        return 2
    docs = {r["id"]: r["text"] for r in load_jsonl(argv[0])}
    queries = load_jsonl(argv[1])
    print(f"fixture set: {len(docs)} docs, {len(queries)} queries")
    for name, metrics in evaluate(docs, queries).items():
        print(f"{name:<8}" + "  ".join(f"{m}={v:.3f}" for m, v in metrics.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
