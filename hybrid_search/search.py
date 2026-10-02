import json
from pathlib import Path

from .bm25 import BM25Index
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .tfidf import TfidfIndex


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({err})") from err
    return rows


class HybridSearcher:
    def __init__(self, docs: dict[str, str], rrf_k: int = DEFAULT_RRF_K) -> None:
        self._rrf_k = rrf_k
        self._bm25 = BM25Index(docs)
        self._tfidf = TfidfIndex(docs)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        fused = reciprocal_rank_fusion(
            [self._bm25.search(query), self._tfidf.search(query)], self._rrf_k
        )
        return fused[:k]
